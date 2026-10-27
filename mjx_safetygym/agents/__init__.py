from .point import Point
from .car import Car
from .doggo import Doggo


AGENTS_REGISTER = [Point, Car, Doggo]

AGENT_NAME_TO_CLASS = {
    'point': Point,
    'car':   Car,
    'doggo': Doggo,
}


def make_agent(name: str, **kwargs):

    cls = AGENT_NAME_TO_CLASS[name.lower()]
    return cls(**kwargs)
