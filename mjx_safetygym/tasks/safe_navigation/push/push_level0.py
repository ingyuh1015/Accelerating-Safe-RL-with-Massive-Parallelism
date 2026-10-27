from __future__ import annotations
import jax
import jax.numpy as jnp

from ....bases.base_task import BaseTask
from ....bases.underlying import agent_xy, agent_heading
from ....utils.lidar import pseudo_lidar

class PushLevel0(BaseTask):
    n_rewards = 1
    n_costs = 1
    n_hazards = 0
    n_pillars = 0
    has_push_box = True

    push_keepout: float = 0.2
    pillar_keepout: float = 0.3
    pillar_size: float = 0.2
    null_dist: float = 0.0
    reward_box_dist: float = 1.0
    reward_box_goal: float = 1.0
    reward_goal: float = 1.0

    def __init__(self, world, push_box_size: float = 0.2, **kwargs):
        kwargs.setdefault('extents', (-1.0, -1.0, 1.0, 1.0))
        super().__init__(world, **kwargs)
        self.push_box_size = float(push_box_size)
        if world.push_box_qpos_adr < 0 or len(world.pillar_mocap_ids) != self.n_pillars:
            raise ValueError(
                f'{type(self).__name__} needs a push_box body and {self.n_pillars} '
                f'pillar mocap bodies — build via make_task.')
        self._box_qadr = int(world.push_box_qpos_adr)
        self._box_body = int(world.push_box_body_id)
        self._pillar_geom_ids = jnp.asarray([world.geom_ids[f'pillar{i}'] for i in range(self.n_pillars)], jnp.int32)
        self._pillar_mocap = (jnp.asarray(world.pillar_mocap_ids, jnp.int32)
                              if self.n_pillars > 0 else None)


    def sample_layout(self, key) -> dict:
        m = self.placements_margin
        ka, kg, kb, kh, kp = jax.random.split(key, 5)
        agent_s = self._sample_xy(ka, shrink=self.agent_keepout)
        goal_xy = self._first_valid(
            self._sample_xy_k(kg, 64, shrink=self.goal_keepout), agent_s[None, :],
            jnp.array([self.agent_keepout + self.goal_keepout + m]))
        bk = self.push_keepout
        fixed = jnp.stack([agent_s, goal_xy])
        box_xy = self._first_valid(
            self._sample_xy_k(kb, 64, shrink=bk), fixed,
            jnp.array([self.agent_keepout + bk + m, self.goal_keepout + bk + m]))
        fixed = jnp.concatenate([fixed, box_xy[None, :]], axis=0)
        seps_haz = jnp.array([self.agent_keepout + self.hazard_keepout + m,
                              self.goal_keepout + self.hazard_keepout + m,
                              bk + self.hazard_keepout + m])
        hazards_xy = self._place_objects(kh, self.n_hazards, self.hazard_keepout,
                                         fixed, seps_haz)
        layout = {'agent_xy': agent_s, 'goal_xy': goal_xy, 'box_xy': box_xy,
                  'hazards_xy': hazards_xy, 'box_xy0': box_xy}
        if self.n_pillars > 0:
            fixed2 = jnp.concatenate([fixed, hazards_xy], axis=0)
            seps_p = jnp.concatenate([
                jnp.array([self.agent_keepout + self.pillar_keepout + m,
                           self.goal_keepout + self.pillar_keepout + m,
                           bk + self.pillar_keepout + m]),
                jnp.full((self.n_hazards,), self.hazard_keepout + self.pillar_keepout + m)])
            layout['pillars_xy'] = self._place_objects(
                kp, self.n_pillars, self.pillar_keepout, fixed2, seps_p)
        else:
            layout['pillars_xy'] = jnp.zeros((0, 2), jnp.float32)
        return layout


    def _apply_layout_extras(self, data0, layout):
        adr = self._box_qadr
        qpos = (data0.qpos
                .at[adr + 0].set(layout['box_xy'][0])
                .at[adr + 1].set(layout['box_xy'][1])
                .at[adr + 2].set(self.push_box_size)
                .at[adr + 3].set(1.0)
                .at[adr + 4].set(0.0).at[adr + 5].set(0.0).at[adr + 6].set(0.0))
        data0 = data0.replace(qpos=qpos)
        if self.n_pillars > 0:
            z = jnp.full((self.n_pillars, 1), 0.25)
            mp = data0.mocap_pos.at[self._pillar_mocap].set(
                jnp.concatenate([layout['pillars_xy'], z], axis=1))
            data0 = data0.replace(mocap_pos=mp)
        return data0

    def _post_physics_info(self, data, info) -> dict:
        info['box_xy'] = data.xpos[self._box_body, :2]
        return info


    def is_goal_reached(self, data, info) -> jax.Array:
        d = jnp.linalg.norm(info['box_xy'] - info['goal_xy'])
        return (d <= self.goal_size).astype(jnp.float32)

    def _on_goal_reached(self, key, data, info, reached) -> dict:


        m = self.placements_margin
        existing = jnp.concatenate([
            info['agent_xy0'][None, :], info['box_xy0'][None, :],
            info['hazards_xy'], info['pillars_xy']], axis=0)
        seps = jnp.concatenate([
            jnp.array([self.agent_keepout + self.goal_keepout + m,
                       self.push_keepout + self.goal_keepout + m]),
            jnp.full((self.n_hazards,), self.hazard_keepout + self.goal_keepout + m),
            jnp.full((self.n_pillars,), self.pillar_keepout + self.goal_keepout + m)])
        new_goal = self._first_valid(
            self._sample_xy_k(key, 64, shrink=self.goal_keepout), existing, seps)
        cont = jnp.float32(self.continue_goal)
        info['goal_xy'] = jnp.where((reached * cont)[None].astype(bool),
                                    new_goal, info['goal_xy'])
        return info

    def _update_trackers(self, data, info) -> dict:
        a_xy = agent_xy(data, self._xy_adr)
        info['last_dist_goal'] = jnp.linalg.norm(a_xy - info['goal_xy'])
        info['last_dist_box'] = jnp.linalg.norm(a_xy - info['box_xy'])
        info['last_box_goal'] = jnp.linalg.norm(info['box_xy'] - info['goal_xy'])
        return info

    def reset(self, key):
        state = super().reset(key)
        info = self._update_trackers(state.data, dict(state.info))
        obs = self.observation(state.data, info)
        return state._replace(obs=obs, info=info)


    def reward_vector(self, prev_dist, data, action, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        dist_box = jnp.linalg.norm(a_xy - info['box_xy'])
        dist_box_goal = jnp.linalg.norm(info['box_xy'] - info['goal_xy'])
        gate = (info['last_dist_box'] > self.null_dist * self.push_box_size
                ).astype(jnp.float32)
        r = ((info['last_dist_box'] - dist_box) * self.reward_box_dist * gate
             + (info['last_box_goal'] - dist_box_goal) * self.reward_box_goal
             + self.is_goal_reached(data, info) * self.reward_goal)
        return jnp.stack([r])

    def cost_vector(self, data, info) -> jnp.ndarray:
        return jnp.zeros((self.n_costs,), dtype=jnp.float32)


    def observation(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        heading = self._heading(data)
        sensors = self._sensor_obs(data)

        def lidar(xys):
            return pseudo_lidar(a_xy, heading, xys, n_bins=self.n_bins,
                                max_dist=self.lidar_max_dist)

        parts = [sensors, lidar(info['goal_xy'][None, :]),
                 lidar(info['box_xy'][None, :])]
        if self.n_hazards > 0:
            parts.append(lidar(info['hazards_xy']))
        if self.n_pillars > 0:
            parts.append(lidar(info['pillars_xy']))
        return jnp.concatenate(parts)

    @property
    def obs_dim(self) -> int:
        return self._sensor_obs_dim + self.n_bins * (
            2 + int(self.n_hazards > 0) + int(self.n_pillars > 0))
