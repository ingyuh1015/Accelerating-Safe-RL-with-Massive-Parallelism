from __future__ import annotations
import jax
import jax.numpy as jnp

from ....bases.base_task import BaseTask
from ....bases.underlying import agent_xy, agent_heading
from ....utils.lidar import pseudo_lidar

class ButtonLevel0(BaseTask):
    n_rewards = 1
    n_costs = 1
    n_hazards = 0
    n_buttons = 4
    n_gremlins = 0

    button_size: float = 0.1
    button_keepout: float = 0.2
    gremlin_travel: float = 0.35
    gremlin_keepout: float = 0.4
    gremlin_size: float = 0.1
    resampling_delay: float = 10.0
    reward_distance: float = 1.0
    reward_goal: float = 1.0

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.0, -1.0, 1.0, 1.0))
        super().__init__(world, **kwargs)
        if len(world.button_mocap_ids) != self.n_buttons or \
                len(world.gremlin_mocap_ids) != self.n_gremlins:
            raise ValueError(
                f'{type(self).__name__} needs {self.n_buttons} button / '
                f'{self.n_gremlins} gremlin mocap bodies in the world model — '
                f'build via make_task (passes the task spec to build_world).')
        self._button_mocap = jnp.asarray(world.button_mocap_ids, jnp.int32)
        self._button_geom_ids = jnp.asarray([world.geom_ids[f'button{i}'] for i in range(self.n_buttons)], jnp.int32)
        self._gremlin_geom_ids = jnp.asarray([world.geom_ids[f'gremlin{i}'] for i in range(self.n_gremlins)], jnp.int32)
        self._gremlin_mocap = (jnp.asarray(world.gremlin_mocap_ids, jnp.int32)
                               if self.n_gremlins > 0 else None)


    def sample_layout(self, key) -> dict:
        m = self.placements_margin
        ka, kb, kg, kh, kgr, kpick = jax.random.split(key, 6)
        agent_s = self._sample_xy(ka, shrink=self.agent_keepout)
        buttons_xy = self._place_objects(
            kb, self.n_buttons, self.button_keepout, agent_s[None, :],
            jnp.array([self.agent_keepout + self.button_keepout + m]))
        fixed = jnp.concatenate([agent_s[None, :], buttons_xy], axis=0)


        goal_slot = self._first_valid(
            self._sample_xy_k(kg, 64, shrink=self.goal_keepout), fixed,
            jnp.concatenate([jnp.array([self.agent_keepout + self.goal_keepout + m]),
                             jnp.full((self.n_buttons,),
                                      self.button_keepout + self.goal_keepout + m)]))
        fixed = jnp.concatenate([fixed, goal_slot[None, :]], axis=0)
        seps_haz = jnp.concatenate([
            jnp.array([self.agent_keepout + self.hazard_keepout + m]),
            jnp.full((self.n_buttons,), self.button_keepout + self.hazard_keepout + m),
            jnp.array([self.goal_keepout + self.hazard_keepout + m])])
        hazards_xy = self._place_objects(kh, self.n_hazards, self.hazard_keepout,
                                         fixed, seps_haz)
        layout = {'agent_xy': agent_s, 'buttons_xy': buttons_xy,
                  'hazards_xy': hazards_xy}
        if self.n_gremlins > 0:
            fixed2 = jnp.concatenate([fixed, hazards_xy], axis=0)
            seps_gr = jnp.concatenate([
                jnp.array([self.agent_keepout + self.gremlin_keepout + m]),
                jnp.full((self.n_buttons,), self.button_keepout + self.gremlin_keepout + m),
                jnp.array([self.goal_keepout + self.gremlin_keepout + m]),
                jnp.full((self.n_hazards,), self.hazard_keepout + self.gremlin_keepout + m)])
            layout['gremlins_xy'] = self._place_objects(
                kgr, self.n_gremlins, self.gremlin_keepout, fixed2, seps_gr)
        else:
            layout['gremlins_xy'] = jnp.zeros((0, 2), jnp.float32)

        gidx = jax.random.randint(kpick, (), 0, self.n_buttons)
        layout['goal_onehot'] = (jnp.arange(self.n_buttons) == gidx).astype(jnp.float32)
        layout['goal_xy'] = buttons_xy[gidx]
        layout['gremlins_xy0'] = layout['gremlins_xy']
        layout['btn_timer'] = jnp.float32(self.resampling_delay)
        return layout


    def _apply_layout_extras(self, data0, layout):
        mp = data0.mocap_pos
        z_b = jnp.full((self.n_buttons, 1), self.button_size)
        mp = mp.at[self._button_mocap].set(
            jnp.concatenate([layout['buttons_xy'], z_b], axis=1))
        if self.n_gremlins > 0:
            z_g = jnp.full((self.n_gremlins, 1), self.gremlin_size)
            mp = mp.at[self._gremlin_mocap].set(
                jnp.concatenate([layout['gremlins_xy'], z_g], axis=1))
        return data0.replace(mocap_pos=mp)


    def _pre_physics(self, data, info):
        if self.n_gremlins == 0:
            return data
        t = data.time
        offs = jnp.stack([jnp.sin(t), jnp.cos(t)]) * self.gremlin_travel
        target = info['gremlins_xy0'] + offs[None, :]
        z = jnp.full((self.n_gremlins, 1), self.gremlin_size)
        mp = data.mocap_pos.at[self._gremlin_mocap].set(
            jnp.concatenate([target, z], axis=1))
        return data.replace(mocap_pos=mp)

    def _post_physics_info(self, data, info) -> dict:
        if self.n_gremlins > 0:
            info['gremlins_xy'] = data.mocap_pos[self._gremlin_mocap, :2]
        return info


    def is_goal_reached(self, data, info) -> jax.Array:
        contacts = self._object_contacts(data, self._button_geom_ids)
        return jnp.any(contacts & (info['goal_onehot'] > 0.5)).astype(jnp.float32)

    def _on_goal_reached(self, key, data, info, reached) -> dict:
        gidx = jax.random.randint(key, (), 0, self.n_buttons)
        new_onehot = (jnp.arange(self.n_buttons) == gidx).astype(jnp.float32)
        hit = reached > 0.5
        info['goal_onehot'] = jnp.where(hit, new_onehot, info['goal_onehot'])
        info['goal_xy'] = jnp.sum(info['buttons_xy'] * info['goal_onehot'][:, None], axis=0)
        info['btn_timer'] = jnp.where(hit, jnp.float32(self.resampling_delay),
                                      jnp.maximum(0.0, info['btn_timer'] - 1.0))
        return info


    def reward_vector(self, prev_dist, data, action, info) -> jnp.ndarray:
        cur_dist = jnp.linalg.norm(agent_xy(data, self._xy_adr) - info['goal_xy'])
        progress = (prev_dist - cur_dist) * self.reward_distance
        r_goal = progress + self.is_goal_reached(data, info) * self.reward_goal
        return jnp.stack([r_goal])

    def cost_vector(self, data, info) -> jnp.ndarray:
        return jnp.zeros((self.n_costs,), dtype=jnp.float32)


    def observation(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        heading = self._heading(data)
        sensors = self._sensor_obs(data)

        def lidar(xys):
            return pseudo_lidar(a_xy, heading, xys, n_bins=self.n_bins,
                                max_dist=self.lidar_max_dist)


        btn_lidar = lidar(info['buttons_xy']) * (info['btn_timer'] == 0.0)
        parts = [sensors, btn_lidar, lidar(info['goal_xy'][None, :])]
        if self.n_hazards > 0:
            parts.append(lidar(info['hazards_xy']))
        if self.n_gremlins > 0:
            parts.append(lidar(info['gremlins_xy']))
        return jnp.concatenate(parts)

    @property
    def obs_dim(self) -> int:
        return self._sensor_obs_dim + self.n_bins * (
            2 + int(self.n_hazards > 0) + int(self.n_gremlins > 0))
