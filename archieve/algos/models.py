from __future__ import annotations
from typing import Callable, Sequence, cast
from flax import linen as nn
from flax.linen.initializers import lecun_normal, variance_scaling
import jax
import jax.numpy as jnp
from archieve import get_dtype, RunningMeanStd

def get_activation(name: str) -> Callable[..., jnp.ndarray]:
    activations = {'swish': nn.swish, 'tanh': jnp.tanh, 'relu': nn.relu, 'leaky_relu': lambda x: nn.leaky_relu(x, 0.01)}
    if name not in activations:
        raise ValueError(f'Unknown activation: {name}. Options: {list(activations.keys())}')
    return activations[name]

def get_kernel_init(name: str):
    if name == 'lecun_normal':
        return lecun_normal()
    elif name == 'kaiming_uniform':
        return variance_scaling(scale=1 / 3, mode='fan_in', distribution='uniform')
    elif name == 'kaiming_standard':
        return variance_scaling(scale=2.0, mode='fan_in', distribution='uniform')
    elif name == 'orthogonal':
        from flax.linen.initializers import orthogonal
        return orthogonal()
    else:
        raise ValueError(f'Unknown weight_init: {name}. Options: lecun_normal, kaiming_uniform, kaiming_standard, orthogonal')

def build_mlp(x: jnp.ndarray, hidden_sizes: Sequence[int], *, activation: Callable=nn.swish, kernel_init=nn.initializers.lecun_normal(), layer_norm: bool=False, compute_dtype: jnp.dtype=jnp.float32, param_dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    for (i, size) in enumerate(hidden_sizes):
        fan_in = int(x.shape[-1])
        x = nn.Dense(size, name=f'fc_{i}', kernel_init=kernel_init, bias_init=nn.initializers.uniform(scale=1.0 / float(fan_in) ** 0.5), dtype=compute_dtype, param_dtype=param_dtype)(x)
        if layer_norm:
            x = nn.LayerNorm(dtype=compute_dtype, param_dtype=param_dtype, name=f'ln_{i}')(x)
        x = activation(x)
    return x

class GaussianLearningActor(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    log_std_init: float = 0.0
    activation: str = 'swish'
    weight_init: str = 'lecun_normal'
    layer_norm: bool = False
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs: jnp.ndarray, obs_rms: RunningMeanStd | None=None) -> tuple[jnp.ndarray, jnp.ndarray]:
        del obs_rms
        x = obs.astype(self.compute_dtype)
        x = build_mlp(x, self.hidden_sizes, activation=get_activation(self.activation), kernel_init=get_kernel_init(self.weight_init), layer_norm=self.layer_norm, compute_dtype=self.compute_dtype, param_dtype=self.param_dtype)
        mean = nn.Dense(self.action_dim, name='mean', kernel_init=get_kernel_init(self.weight_init), dtype=self.compute_dtype, param_dtype=self.param_dtype)(x)
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, self.log_std_init), (self.action_dim,))
        return (mean.astype(get_dtype()), jnp.broadcast_to(log_std, mean.shape).astype(get_dtype()))

    def sample_action(self, params, obs, rng, obs_rms=None):
        (mean, log_std) = cast(tuple, self.apply(params, obs, obs_rms))
        std = jnp.exp(log_std)
        pre_tanh = mean + std * jax.random.normal(rng, shape=mean.shape, dtype=mean.dtype)
        action = jnp.tanh(pre_tanh)
        logp = self._log_prob(pre_tanh, mean, log_std)
        return (action, (logp, mean, log_std, pre_tanh))

    def loss(self, params, batch, *, adv, entropy_coef=0.0):
        (mean, log_std) = cast(tuple, self.apply(params, batch.obs))
        logp_new = self._log_prob(batch.pre_tanh, mean, log_std)
        ratio = jnp.exp(logp_new - batch.logp)
        objective = ratio * adv
        entropy = jnp.mean(jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1))
        policy_loss = -jnp.mean(objective) - jnp.asarray(entropy_coef, dtype=get_dtype()) * entropy
        return (policy_loss, {'entropy': entropy, 'ratio_mean': jnp.mean(ratio)})

    @staticmethod
    def _log_prob(pre_tanh: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        standardized = (pre_tanh - mean) * jnp.exp(-log_std)
        log_prob = -0.5 * (standardized ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        logp = jnp.sum(log_prob, axis=-1)
        squash = jnp.sum(2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)), axis=-1)
        return logp - squash

class ValueCritic(nn.Module):
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = 'swish'
    weight_init: str = 'lecun_normal'
    layer_norm: bool = False
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs: jnp.ndarray, obs_rms: RunningMeanStd | None=None) -> jnp.ndarray:
        del obs_rms
        x = obs.astype(self.compute_dtype)
        x = build_mlp(x, self.hidden_sizes, activation=get_activation(self.activation), kernel_init=get_kernel_init(self.weight_init), layer_norm=self.layer_norm, compute_dtype=self.compute_dtype, param_dtype=self.param_dtype)
        raw = nn.Dense(1, name='value', kernel_init=get_kernel_init(self.weight_init), dtype=get_dtype(), param_dtype=self.param_dtype)(x)
        return jnp.squeeze(raw, axis=-1).astype(get_dtype())

    def loss(self, params, batch, *, target, use_critic_norm=False, critic_norm_coef=0.001):
        value = self.apply(params, batch.obs)
        mse_loss = jnp.mean((target - value) ** 2)
        l2 = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y ** 2), params, initializer=0.0)
        return mse_loss + critic_norm_coef * l2 * use_critic_norm

def gaussian_kl(mean_old: jnp.ndarray, log_std_old: jnp.ndarray, mean_new: jnp.ndarray, log_std_new: jnp.ndarray) -> jnp.ndarray:
    variance_ratio = jnp.exp(2.0 * (log_std_old - log_std_new))
    normalized_diff = (mean_old - mean_new) * jnp.exp(-log_std_new)
    kl_per_dim = log_std_new - log_std_old + 0.5 * (variance_ratio + normalized_diff ** 2 - 1.0)
    return jnp.sum(kl_per_dim, axis=-1)
