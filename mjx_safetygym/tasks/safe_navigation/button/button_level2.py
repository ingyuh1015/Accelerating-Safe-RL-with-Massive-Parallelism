from __future__ import annotations

from .button_level1 import ButtonLevel1


class ButtonLevel2(ButtonLevel1):
    n_hazards = 8
    n_gremlins = 6

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.8, -1.8, 1.8, 1.8))
        super().__init__(world, **kwargs)
