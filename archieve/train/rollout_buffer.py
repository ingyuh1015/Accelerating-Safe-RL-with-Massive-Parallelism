from __future__ import annotations
from flashbax.buffers import trajectory_buffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as RolloutBufferState
import jax
import jax.numpy as jnp
from archieve import OnBatchState

class RolloutBuffer:

    def __init__(self, num_envs: int, sample_length: int, zero_batch_state: OnBatchState) -> None:
        self.num_envs = num_envs
        self.length = sample_length
        self._prototype = jax.tree.map(lambda x: x[0], zero_batch_state)

    def init(self) -> RolloutBufferState:
        return trajectory_buffer.init(self._prototype, add_batch_size=self.num_envs, max_length_time_axis=self.length)

    def add(self, state: RolloutBufferState, batch_state: OnBatchState) -> RolloutBufferState:
        return trajectory_buffer.add(state, jax.tree.map(lambda x: x[:, None], batch_state))

    def add_masked(self, state: RolloutBufferState, batch_state: OnBatchState, active: jnp.ndarray) -> RolloutBufferState:
        slot = state.current_index
        batch_state = jax.tree.map(lambda exp_leaf, trans_leaf: jnp.where(active.reshape(active.shape + (1,) * (trans_leaf.ndim - 1)), trans_leaf, exp_leaf[:, slot]), state.experience, batch_state)
        return self.add(state, batch_state)

    def sample(self, state: RolloutBufferState) -> OnBatchState:
        return jax.tree.map(lambda x: x.swapaxes(0, 1), state.experience)
