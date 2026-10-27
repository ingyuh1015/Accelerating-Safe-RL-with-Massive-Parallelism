from __future__ import annotations
from archieve import RawPhaseMixtureTightening, PIDRWeights, PIDRUpper, get_dtype
import os
import jax.numpy as jnp

_PIDR_NORM_ITERS = max(0, int(os.environ.get('PIDR_NORM_ITERS', '3')))

def _as_constraint_vector(value: float | jnp.ndarray, num_constraints: int) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=get_dtype())
    if arr.ndim == 0:
        return jnp.full((num_constraints,), arr, dtype=get_dtype())
    return arr.reshape(-1)[:num_constraints]

def empirical_bernstein_radius(samples_NC: jnp.ndarray, *, bound: float | jnp.ndarray, delta: float) -> jnp.ndarray:
    samples = jnp.asarray(samples_NC, dtype=get_dtype())
    if samples.ndim != 2:
        raise ValueError(f'samples_NC must have shape (N, C), got {samples.shape}')
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f'delta must be in (0, 1), got {delta}')
    N = samples.shape[0]
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    sample_var = jnp.sum(jnp.square(centered), axis=0) / jnp.maximum(float(N - 1), 1.0)
    range_vec = _as_constraint_vector(bound, samples.shape[1]).reshape((samples.shape[1],))
    log_term = jnp.asarray(jnp.log(2.0 / float(delta)), dtype=get_dtype())
    return jnp.where(N > 1, jnp.sqrt(2.0 * sample_var * log_term / float(N)) + 7.0 * range_vec * log_term / (3.0 * jnp.maximum(float(N - 1), 1.0)), range_vec)

def _studentized_radius(samples_NC: jnp.ndarray, *, delta: float) -> jnp.ndarray:
    samples = jnp.asarray(samples_NC, dtype=get_dtype())
    if samples.ndim != 2:
        raise ValueError(f'samples_NC must have shape (N, C), got {samples.shape}')
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f'delta must be in (0, 1), got {delta}')
    N = samples.shape[0]
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    sample_var = jnp.sum(jnp.square(centered), axis=0) / jnp.maximum(float(N - 1), 1.0)
    denom = jnp.asarray(float(N), dtype=get_dtype())
    log_term = jnp.asarray(jnp.log(1.0 / float(delta)), dtype=get_dtype())
    return jnp.where(N > 1, jnp.sqrt(2.0 * sample_var * log_term / jnp.maximum(denom, jnp.asarray(1.0, dtype=get_dtype()))), jnp.zeros((samples.shape[1],), dtype=get_dtype()))

def compute_raw_phase_mixture_tightening(costs_TNc: jnp.ndarray, steps_TN: jnp.ndarray, *, gamma: float, episode_length: int, delta_beta: float, c_max: float | jnp.ndarray=1.0) -> RawPhaseMixtureTightening:
    costs = jnp.asarray(costs_TNc, dtype=get_dtype())
    steps = jnp.asarray(steps_TN, dtype=jnp.int32)
    if costs.ndim != 3:
        raise ValueError(f'costs_TNc must have shape (T, N, C), got {costs.shape}')
    if steps.shape != costs.shape[:2]:
        raise ValueError(f'steps_TN shape {steps.shape} does not match costs prefix {costs.shape[:2]}')
    if int(episode_length) <= 0:
        raise ValueError(f'episode_length must be positive, got {episode_length}')
    if not 0.0 < float(delta_beta) < 1.0:
        raise ValueError(f'delta_beta must be in (0, 1), got {delta_beta}')
    (T, N, nc) = costs.shape
    del T
    H = int(episode_length)
    steps = jnp.clip(steps, 0, H - 1)
    flat_steps = steps.reshape(-1)
    counts = jnp.bincount(flat_steps, length=H).astype(get_dtype())
    sample_counts = counts[flat_steps].reshape(steps.shape)
    gamma_arr = jnp.asarray(gamma, dtype=get_dtype())
    weights = jnp.asarray(N, dtype=get_dtype()) * jnp.power(gamma_arr, steps.astype(get_dtype())) / jnp.maximum(sample_counts, jnp.asarray(1.0, dtype=get_dtype()))
    surrogate = jnp.sum(weights[..., None] * costs, axis=(0, 1)) / jnp.asarray(N, dtype=get_dtype())
    cmax_vec = _as_constraint_vector(c_max, nc)
    per_env_range_scalar = jnp.sum(weights, axis=0)
    per_env_raw = jnp.sum(weights[..., None] * costs, axis=0)
    centered = per_env_raw - jnp.mean(per_env_raw, axis=0, keepdims=True)
    sample_var = jnp.sum(jnp.square(centered), axis=0) / jnp.maximum(float(N - 1), 1.0)
    range_vec = cmax_vec * jnp.max(per_env_range_scalar)
    log_term = jnp.asarray(jnp.log(2.0 / float(delta_beta)), dtype=get_dtype())
    beta = jnp.sqrt(2.0 * sample_var * log_term / float(N)) + 7.0 * range_vec * log_term / (3.0 * jnp.maximum(float(N - 1), 1.0))
    coverage_min_count = jnp.min(counts)
    coverage_complete = coverage_min_count > 0
    if abs(float(gamma) - 1.0) < 1e-09:
        finite_bound_scalar = jnp.asarray(H, dtype=get_dtype())
    else:
        finite_bound_scalar = (1.0 - jnp.power(gamma_arr, jnp.asarray(H, dtype=get_dtype()))) / (1.0 - gamma_arr)
    finite_bound = cmax_vec * finite_bound_scalar
    tightened = jnp.where(coverage_complete, surrogate + beta, finite_bound)
    return RawPhaseMixtureTightening(surrogate=surrogate, beta=beta, tightened=tightened, coverage_min_count=coverage_min_count, coverage_complete=coverage_complete)

def _phase_mean_normalize(w: jnp.ndarray, phases: jnp.ndarray, max_phase: int) -> jnp.ndarray:

    dtype = get_dtype()
    counts = jnp.bincount(phases, length=max_phase).astype(dtype)
    sums = jnp.bincount(phases, weights=w, length=max_phase).astype(dtype)
    means = sums / jnp.maximum(counts, jnp.asarray(1.0, dtype=dtype))
    return w / jnp.maximum(means[phases], jnp.asarray(1e-08, dtype=dtype))

def normalize_pidr_weights(raw_weights_N: jnp.ndarray, *, ratio_clip: float | jnp.ndarray, phase_N: jnp.ndarray, phase_count: int, f_loss: jnp.ndarray | float=0.0, r_loss: jnp.ndarray | float=0.0) -> PIDRWeights:

    dtype = get_dtype()
    clip = jnp.maximum(jnp.asarray(ratio_clip, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    raw = jnp.clip(jnp.maximum(jnp.asarray(raw_weights_N, dtype=dtype).reshape(-1), 0.0), 1.0 / clip, clip)
    N = raw.shape[0]
    phases = jnp.asarray(phase_N, dtype=jnp.int32).reshape((N,))
    max_phase = int(phase_count)
    if max_phase <= 0:
        raise ValueError(f'phase_count must be positive, got {phase_count}')
    clipped_phases = jnp.clip(phases, 0, max_phase - 1)
    for _ in range(_PIDR_NORM_ITERS):
        raw = _phase_mean_normalize(raw, clipped_phases, max_phase)
        raw = jnp.clip(jnp.maximum(raw, 0.0), 1.0 / clip, clip)
    counts = jnp.bincount(clipped_phases, length=max_phase).astype(dtype)
    weights = _phase_mean_normalize(raw, clipped_phases, max_phase)
    phase_weight_sums = jnp.bincount(clipped_phases, weights=weights, length=max_phase).astype(dtype)
    phase_sq_sums = jnp.bincount(clipped_phases, weights=jnp.square(weights), length=max_phase).astype(dtype)
    tv_sums = jnp.bincount(clipped_phases, weights=jnp.abs(weights - 1.0), length=max_phase).astype(dtype)
    chi2_sums = jnp.bincount(clipped_phases, weights=jnp.square(weights - 1.0), length=max_phase).astype(dtype)
    observed = counts > 0
    tv_phi = 0.5 * tv_sums / jnp.maximum(counts, jnp.asarray(1.0, dtype=dtype))
    chi2_phi = chi2_sums / jnp.maximum(counts, jnp.asarray(1.0, dtype=dtype))
    ess_phi = jnp.square(phase_weight_sums) / jnp.maximum(counts * phase_sq_sums, jnp.asarray(1e-08, dtype=dtype))
    observed_count = jnp.maximum(jnp.sum(observed).astype(dtype), jnp.asarray(1.0, dtype=dtype))
    tv_by_phase = jnp.sum(jnp.where(observed, tv_phi, 0.0)) / observed_count
    chi2_by_phase = jnp.sum(jnp.where(observed, chi2_phi, 0.0)) / observed_count
    ess_by_phase = jnp.sum(jnp.where(observed, ess_phi, 0.0)) / observed_count
    norm_error = jnp.max(jnp.where(observed, jnp.abs(phase_weight_sums / counts - 1.0), 0.0))
    ratio_sum = jnp.sum(weights)
    ratio_sq_sum = jnp.sum(jnp.square(weights))
    ess = jnp.square(ratio_sum) / jnp.maximum(jnp.asarray(float(N), dtype=dtype) * ratio_sq_sum, jnp.asarray(1e-08, dtype=dtype))
    return PIDRWeights(weights=weights, raw_weights=raw, ratio_mean=jnp.mean(weights), ratio_std=jnp.std(weights), ratio_min=jnp.min(weights), ratio_max=jnp.max(weights), ratio_ess=ess, tv=0.5 * jnp.mean(jnp.abs(weights - 1.0)), chi2=jnp.mean(jnp.square(weights - 1.0)), tv_by_phase=tv_by_phase, chi2_by_phase=chi2_by_phase, ess_by_phase=ess_by_phase, norm_error=norm_error, f_loss=jnp.asarray(f_loss, dtype=dtype), r_loss=jnp.asarray(r_loss, dtype=dtype))

def compute_pidr_weighted_upper(samples_NC: jnp.ndarray, *, base_surrogate_C: jnp.ndarray, base_beta_C: jnp.ndarray, weights_N: jnp.ndarray, delta: float, baseline_NC: jnp.ndarray) -> PIDRUpper:

    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f'delta must be in (0, 1), got {delta}')
    dtype = get_dtype()
    samples = jnp.asarray(samples_NC, dtype=dtype)
    if samples.ndim != 2:
        raise ValueError(f'samples_NC must have shape (N, C), got {samples.shape}')
    (N, nc) = samples.shape
    weights = jnp.asarray(weights_N, dtype=dtype).reshape((N,))
    base = jnp.asarray(base_surrogate_C, dtype=dtype).reshape((nc,))
    base_beta = jnp.asarray(base_beta_C, dtype=dtype).reshape((nc,))
    centered = samples - jnp.asarray(baseline_NC, dtype=dtype).reshape((N, nc))
    correction_samples = (weights[:, None] - 1.0) * centered

    correction_mean = jnp.mean((weights[:, None] - 1.0) * samples, axis=0)
    correction_uncertainty = _studentized_radius(correction_samples, delta=delta)
    correction_upper = jnp.maximum(correction_mean + correction_uncertainty, jnp.zeros((nc,), dtype=dtype))
    upper = base + base_beta + correction_upper
    return PIDRUpper(surrogate=base, base_beta=base_beta, correction_mean=correction_mean, correction_uncertainty=correction_uncertainty, epsilon=correction_upper, upper=upper)
