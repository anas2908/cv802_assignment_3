"""Compact storage of the trained models in results/models/*.npz, so the whole repository stays small.

An array can be stored as 8- or 16-bit codes plus one value range per channel:

    x  ~=  lo + code / (2^bits - 1) * (hi - lo)            (key__q = codes, key__lo / key__hi = range, key__axis)

`unpack` turns such entries back into float32 arrays and passes every other array through unchanged, so the
renderers read packed and plain files alike. training/export_bundle.py measures what packing changes in the renders.
"""
from __future__ import annotations

import numpy as np

_META = ("__lo", "__hi", "__axis")


def quantize(name, x, bits=8, axis=-1):
    """{name__q, name__lo, name__hi, name__axis}: codes with one range per index along `axis`"""
    x = np.asarray(x, np.float32)
    axis = axis % x.ndim
    xm = np.moveaxis(x, axis, 0)
    flat = xm.reshape(xm.shape[0], -1)
    lo, hi = flat.min(1), flat.max(1)
    span = np.where(hi > lo, hi - lo, 1.0)
    levels = 2 ** bits - 1
    codes = np.round((flat - lo[:, None]) / span[:, None] * levels)
    codes = np.moveaxis(codes.reshape(xm.shape), 0, axis).astype(np.uint8 if bits == 8 else np.uint16)
    return {name + "__q": codes, name + "__lo": lo, name + "__hi": hi, name + "__axis": np.asarray(axis)}


def unpack(z):
    """dict of arrays (e.g. an opened .npz) -> dict with every quantised entry restored as float32"""
    out = {}
    for k in z.keys():
        if k.endswith("__q"):
            base = k[:-3]
            q = np.asarray(z[k])
            axis = int(z[base + "__axis"])
            shape = [1] * q.ndim
            shape[axis] = -1
            lo = np.asarray(z[base + "__lo"], np.float32).reshape(shape)
            hi = np.asarray(z[base + "__hi"], np.float32).reshape(shape)
            out[base] = lo + q.astype(np.float32) / float(np.iinfo(q.dtype).max) * (hi - lo)
        elif not k.endswith(_META):
            out[k] = z[k]
    return out
