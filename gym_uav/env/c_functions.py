import os
import sys

from ctypes import *

import numpy as np
from numpy.ctypeslib import ndpointer


def _get_dll_name():
    if sys.platform == 'linux':
        dll_name = 'libc_functions.so'
    elif sys.platform == 'win32':
        dll_name = 'libc_functions.dll'
    else:
        # Your system's dll name
        dll_name = None
    return dll_name


def _get_c_dijkstra():
    dll_name = _get_dll_name()
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), dll_name))
    dijkstra = lib.Dijkstra
    dijkstra.restype = c_int32
    dijkstra.argtypes = [
        ndpointer(np.uint8, flags='aligned, c_contiguous'),
        c_uint32, c_int32, c_uint32
    ]
    return dijkstra


def _get_c_get_dist():
    dll_name = _get_dll_name()
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), dll_name))
    obs_dist_circle = lib.GetDistance
    obs_dist_circle.restype = c_double
    return obs_dist_circle


_dijkstra = _get_c_dijkstra()
_obs_dist_circle = _get_c_get_dist()


def c_dijkstra(arr: np.ndarray, start: tuple[int, int]):
    res = _dijkstra(arr, arr.shape[0], *start)
    return int(res)


def c_obs_dist_circle(_obs_x: list, _obs_y: list, uav_x: float, uav_y: float, end_x: float, end_y: float, r: float,
                      obs_cnt: int):
    obs_x = (c_double * obs_cnt)()
    obs_y = (c_double * obs_cnt)()
    for i in range(obs_cnt):
        obs_x[i] = _obs_x[i]
        obs_y[i] = _obs_y[i]

    res = _obs_dist_circle(obs_x, obs_y, c_double(uav_x), c_double(uav_y), c_double(end_x), c_double(end_y),
                           c_double(r),
                           c_int(obs_cnt))
    return float(res)
