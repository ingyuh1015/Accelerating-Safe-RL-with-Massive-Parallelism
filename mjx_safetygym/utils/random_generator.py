from __future__ import annotations
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from .common_utils import ResamplingError


class RandomGenerator:


    def __init__(self) -> None:
        self.key: Optional[jax.Array] = None
        self.placements: Optional[dict] = None
        self.placements_extents: Optional[list] = None
        self.placements_margin: Optional[float] = None
        self.layout: Optional[dict[str, np.ndarray]] = None


    def set_random_seed(self, seed: int) -> None:
        self.key = jax.random.PRNGKey(int(seed))

    def set_placements_info(
        self,
        placements: dict,
        placements_extents: list,
        placements_margin: float,
    ) -> None:
        self.placements = placements
        self.placements_extents = placements_extents
        self.placements_margin = placements_margin


    def _split(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey


    def uniform(self, low, high, size=None) -> np.ndarray:
        subkey = self._split()
        if size is None:
            v = jax.random.uniform(subkey, shape=(), minval=low, maxval=high)
            return float(v)
        v = jax.random.uniform(subkey, shape=tuple(np.atleast_1d(size)), minval=low, maxval=high)
        return np.asarray(v)

    def randn(self, *shape) -> np.ndarray:
        subkey = self._split()
        v = jax.random.normal(subkey, shape=tuple(shape) if shape else ())
        return np.asarray(v) if shape else float(v)

    def binomial(self, n: int, p: float, size=None) -> np.ndarray:
        subkey = self._split()
        if size is None:
            v = jax.random.bernoulli(subkey, p=p, shape=(int(n),)).sum()
            return int(v)
        v = jax.random.bernoulli(subkey, p=p, shape=tuple(np.atleast_1d(size)) + (int(n),)).sum(-1)
        return np.asarray(v)

    def choice(self, a, size=None, p=None) -> np.ndarray:
        subkey = self._split()
        a_arr = jnp.asarray(a) if not np.isscalar(a) else jnp.arange(int(a))
        p_arr = jnp.asarray(p) if p is not None else None
        v = jax.random.choice(
            subkey,
            a_arr,
            shape=tuple(np.atleast_1d(size)) if size is not None else (),
            p=p_arr,
        )
        return np.asarray(v) if size is not None else int(v) if a_arr.dtype.kind in 'iu' else float(v)

    def random_rot(self) -> float:

        return self.uniform(0.0, 2 * np.pi)

    def generate_rots(self, num: int = 1) -> list[float]:
        return [self.random_rot() for _ in range(num)]


    def build_layout(self) -> dict:

        for _ in range(10000):
            if self.sample_layout():
                return self.layout
        raise ResamplingError('Failed to sample layout of objects')

    def sample_layout(self) -> bool:


        def is_valid(xy, layout):
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = float(np.linalg.norm(xy - other_xy))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout: dict[str, np.ndarray] = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout)
                if is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def sample_goal_position(self) -> bool:

        placements, keepout = self.placements['goal']
        goal_xy = self.draw_placement(placements, keepout)
        for other_name, other_xy in self.layout.items():
            if other_name == 'goal':
                continue
            other_keepout = self.placements[other_name][1]
            dist = float(np.linalg.norm(goal_xy - other_xy))
            if dist < other_keepout + self.placements_margin + keepout:
                return False
        self.layout['goal'] = goal_xy
        return True

    def draw_placement(self, placements, keepout: float) -> np.ndarray:

        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert constrained, 'Failed to find any placements satisfying keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained])
                probs = areas / areas.sum()
                idx = int(self.choice(len(constrained), p=probs))
                choice = constrained[idx]
        xmin, ymin, xmax, ymax = choice
        return np.array([
            self.uniform(xmin, xmax),
            self.uniform(ymin, ymax),
        ], dtype=np.float64)

    @staticmethod
    def constrain_placement(placement, keepout: float) -> tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)
