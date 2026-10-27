from __future__ import annotations
from typing import Iterable

import xml.etree.ElementTree as ET
import numpy as np

from .common_utils import convert


def make_geom(
    name: str,
    geom_type: str,
    pos: np.ndarray,
    size,
    rgba=None,
    quat=None,
    contype: int = 1,
    conaffinity: int = 1,
    group: int = 1,
    **extra,
) -> ET.Element:

    attrs = {
        'name': name,
        'type': geom_type,
        'pos': convert(np.asarray(pos, dtype=np.float64)),
        'size': convert(size),
        'contype': str(contype),
        'conaffinity': str(conaffinity),
        'group': str(group),
    }
    if rgba is not None:
        attrs['rgba'] = convert(np.asarray(rgba))
    if quat is not None:
        attrs['quat'] = convert(np.asarray(quat))
    attrs.update({k: convert(v) for k, v in extra.items()})
    return ET.Element('geom', attrs)


def make_site(name: str, pos: np.ndarray, size=0.1, rgba=None, **extra) -> ET.Element:
    attrs = {'name': name, 'pos': convert(pos), 'size': convert(size)}
    if rgba is not None:
        attrs['rgba'] = convert(rgba)
    attrs.update({k: convert(v) for k, v in extra.items()})
    return ET.Element('site', attrs)


def make_body(name: str, pos: np.ndarray, children: Iterable[ET.Element] = (), **extra) -> ET.Element:
    attrs = {'name': name, 'pos': convert(pos)}
    attrs.update({k: convert(v) for k, v in extra.items()})
    body = ET.Element('body', attrs)
    for c in children:
        body.append(c)
    return body


def make_freejoint(name: str) -> ET.Element:
    return ET.Element('freejoint', {'name': name})


def wrap_world_xml(
    worldbody_children: Iterable[ET.Element],
    extra_root_children: Iterable[ET.Element] = (),
    timestep: float = 0.002,
    integrator: str = 'RK4',
) -> str:


    root = ET.Element('mujoco', {'model': 'mjx_safetygym'})


    ET.SubElement(root, 'option', {
        'timestep': f'{timestep}',
        'integrator': integrator,
        'gravity': '0 0 -9.81',
    })


    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', {'solref': '0.005 1', 'solimp': '0.99 0.999 0.001'})

    for c in extra_root_children:
        root.append(c)

    worldbody = ET.SubElement(root, 'worldbody')

    ET.SubElement(worldbody, 'geom', {
        'name': 'floor', 'type': 'plane', 'pos': '0 0 0',
        'size': '50 50 0.1', 'rgba': '0.85 0.85 0.85 1',
        'contype': '1', 'conaffinity': '1',
    })
    for c in worldbody_children:
        worldbody.append(c)

    return ET.tostring(root, encoding='unicode')
