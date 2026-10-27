from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _wrap_pi(x):

    return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi


@dataclass(frozen=True)
class LaikagoGenerator:


    foot_height: float = 0.0
    bottom: float = -0.4
    abduct_length: float = 0.037
    thigh_length: float = 0.25
    knee_length: float = 0.25
    num_leg: int = 4
    default_freq: float = 10.0

    @property
    def init_phi(self) -> jax.Array:

        return jnp.array([0.0, jnp.pi, jnp.pi, 0.0])


    def phi_list(self, t, freq_list=None) -> jax.Array:

        if freq_list is None:
            freq_list = jnp.zeros(self.num_leg)
        return jnp.mod(self.init_phi + (freq_list + self.default_freq) * t,
                       2.0 * jnp.pi)

    def phase_list(self, t, freq_list=None) -> jax.Array:

        phi = self.phi_list(t, freq_list)

        return jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1).ravel()


    def _foot_z(self, phi) -> jax.Array:

        k = 2.0 * (phi - jnp.pi) / jnp.pi


        fh = self.foot_height
        seg_mid = fh * (-2.0 * k**3 + 3.0 * k**2) + self.bottom
        seg_hi = fh * (2.0 * k**3 - 9.0 * k**2 + 12.0 * k - 4.0) + self.bottom
        pos = jnp.where(k < 0.0,
                        self.bottom,
                        jnp.where(k < 1.0, seg_mid, seg_hi))
        return pos


    def _ik_solve(self, base_pos) -> jax.Array:

        a0, b0, c0 = base_pos[0], base_pos[1], base_pos[2]
        x2 = a0

        alpha1 = jnp.arctan2(c0, b0)
        z2 = -jnp.sqrt(jnp.clip(b0**2 + c0**2 - self.abduct_length**2, 0.0, jnp.inf))
        theta1 = alpha1 - jnp.arctan2(z2, -self.abduct_length)
        theta1 = _wrap_pi(theta1)

        a = (self.knee_length**2 - (x2**2 + z2**2 + self.thigh_length**2)) \
            / (2.0 * self.thigh_length)
        alpha = jnp.arctan2(x2, z2)
        acos_arg = jnp.clip(a / jnp.sqrt(x2**2 + z2**2), -1.0, 1.0)


        theta2_a = -alpha + jnp.arccos(acos_arg)
        theta3_a = jnp.arctan2(x2 - self.thigh_length * jnp.sin(theta2_a),
                               -z2 - self.thigh_length * jnp.cos(theta2_a)) - theta2_a
        theta3_a = _wrap_pi(theta3_a)


        theta2_b = -alpha - jnp.arccos(acos_arg)
        theta3_b = jnp.arctan2(x2 - self.thigh_length * jnp.sin(theta2_b),
                               -z2 - self.thigh_length * jnp.cos(theta2_b)) - theta2_b
        theta3_b = _wrap_pi(theta3_b)

        use_b = theta3_a < 0.0
        theta2 = jnp.where(use_b, theta2_b, theta2_a)
        theta3 = jnp.where(use_b, theta3_b, theta3_a)
        theta2 = _wrap_pi(theta2)
        return jnp.array([theta1, theta2, theta3])


    def joint_targets(self, t, base_mat=None, res_foot_pos=None,
                      freq_list=None) -> jax.Array:

        if base_mat is None:
            base_mat = jnp.eye(3)
        if res_foot_pos is None:
            res_foot_pos = jnp.zeros((self.num_leg, 3))
        if freq_list is None:
            freq_list = jnp.zeros(self.num_leg)


        theta = _euler_xyz_yaw(base_mat)
        cz, sz = jnp.cos(theta), jnp.sin(theta)
        H_mat = jnp.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        B_mat = base_mat.T @ H_mat

        phi = self.phi_list(t, freq_list)
        pos_z = self._foot_z(phi)


        is_left = jnp.array([0.0, 1.0, 0.0, 1.0])
        y_off = jnp.where(is_left > 0.5, self.abduct_length, -self.abduct_length)

        lo = jnp.array([-0.1, -0.1, self.bottom - 0.1])
        hi = jnp.array([0.1, 0.1, self.bottom + 0.1])

        def per_leg(leg_idx):
            target_foot = jnp.array([0.0, y_off[leg_idx], pos_z[leg_idx]]) \
                + res_foot_pos[leg_idx]
            target_foot = jnp.clip(target_foot, lo, hi)
            base_pos = B_mat @ target_foot
            left = is_left[leg_idx] > 0.5

            base_pos = base_pos.at[1].set(jnp.where(left, -base_pos[1], base_pos[1]))
            thetas = self._ik_solve(base_pos)
            thetas = thetas.at[0].set(jnp.where(left, -thetas[0], thetas[0]))
            return thetas

        all_thetas = jax.vmap(per_leg)(jnp.arange(self.num_leg))
        return all_thetas.reshape(-1)


def _euler_xyz_yaw(mat):

    return jnp.arctan2(mat[1, 0], mat[0, 0])
