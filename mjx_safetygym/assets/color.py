import jax.numpy as jnp


COLOR = {
    'push_box': jnp.array([1.0, 1.0, 0.0, 1.0]),
    'button':   jnp.array([1.0, 0.5, 0.0, 1.0]),
    'goal':     jnp.array([0.0, 1.0, 0.0, 1.0]),
    'vase':     jnp.array([0.0, 1.0, 1.0, 1.0]),
    'hazard':   jnp.array([0.0, 0.0, 1.0, 1.0]),
    'pillar':   jnp.array([0.5, 0.5, 1.0, 1.0]),
    'wall':     jnp.array([0.5, 0.5, 0.5, 1.0]),
    'gremlin':  jnp.array([0.5, 0.0, 1.0, 1.0]),
    'circle':   jnp.array([0.0, 1.0, 0.0, 1.0]),
    'red':      jnp.array([1.0, 0.0, 0.0, 1.0]),
    'apple':    jnp.array([0.835, 0.169, 0.169, 1.0]),
    'orange':   jnp.array([1.0, 0.6, 0.0, 1.0]),
    'sigwall':  jnp.array([1.0, 1.0, 0.0, 1.0]),
}
