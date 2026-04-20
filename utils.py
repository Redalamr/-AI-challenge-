import ctypes
import numpy as np


def t2np(tensor):
    # conversion tensor vers numpy via ctypes, plus stable
    t = tensor.contiguous().cpu().float()
    n = t.numel()
    if n == 0:
        return np.zeros(t.shape, dtype=np.float32)
    ptr = t.data_ptr()
    arr = np.ctypeslib.as_array(
        (ctypes.c_float * n).from_address(ptr)
    ).reshape(t.shape).copy()
    return arr
