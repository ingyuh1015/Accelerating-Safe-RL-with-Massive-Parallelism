from __future__ import annotations

from typing import Any, NamedTuple

import jax


class State(NamedTuple):


    data: Any
    obs: jax.Array
    reward: jax.Array
    cost: jax.Array
    done: jax.Array
    key: jax.Array
    info: dict
