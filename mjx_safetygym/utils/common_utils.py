from __future__ import annotations

import re
import numpy as np


class ResamplingError(AssertionError):
    pass


class MjxException(Exception):
    pass


def quat2zalign(quat: np.ndarray) -> float:

    a, b, c, d = quat
    return float(a * a - b * b - c * c + d * d)


def rot2quat(theta: float) -> np.ndarray:

    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float64)


def convert(value) -> str:

    if isinstance(value, (int, float, str)):
        return str(value)
    return ' '.join(str(i) for i in np.asarray(value).flatten())


def camel_to_snake(name: str) -> str:

    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()
