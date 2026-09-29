"""Which EGL device renders on a given CUDA device.

``MUJOCO_EGL_DEVICE_ID`` is an index into ``eglQueryDevicesEXT()``, whose order is not
the CUDA order (a permutation that differs across hosts). Each EGL device reports the
CUDA-visible ordinal it belongs to, so the index is looked up live.
"""

from __future__ import annotations

import ctypes

from mujoco.egl import egl_ext

EGL_CUDA_DEVICE_NV = 0x323A


def egl_index_for_cuda_ordinal(ordinal: int) -> int:
    """The EGL device index whose CUDA-visible ordinal is ``ordinal``."""
    libegl = ctypes.CDLL("libEGL.so.1")
    libegl.eglGetProcAddress.restype = ctypes.c_void_p
    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    address = libegl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not address:
        raise RuntimeError("EGL has no eglQueryDeviceAttribEXT")
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )(address)
    matches = []
    for index, device in enumerate(egl_ext.eglQueryDevicesEXT()):
        value = ctypes.c_ssize_t(-1)
        ok = query(
            ctypes.c_void_p(int(device.address)),
            EGL_CUDA_DEVICE_NV,
            ctypes.byref(value),
        )
        if ok and value.value == ordinal:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"CUDA ordinal {ordinal} matches EGL devices {matches}")
    return matches[0]
