#!/usr/bin/env python3
"""The tensor index of an HFQ file (engine/core/hfq.cpp's reader, for the tools' checks on real rows): HF names, the
quant type (1 f16, 3 Q8_0, 44 MQ4G256), the shape as stored ([rows, K] for weights) and the bytes, memory-mapped.

usage: hfq_index.py <model.mq4-xts> [name filter]   (prints the matching tensors in file order)
"""
import json
import struct
import sys

import numpy as np


def read(path):
    """[(name, qt, shape, offset, size)] in file order, offsets from the start of the file."""
    with open(path, "rb") as f:
        magic, version, _arch, count, meta_off, data_off = struct.unpack("<4sIIIQQ", f.read(32))
        if version != 1:
            raise ValueError(f"hfq: version {version:#x}")
        f.seek(meta_off)
        head = f.read(data_off - meta_off)
    _, end = json.JSONDecoder().raw_decode(head.decode("latin-1"))
    p = end
    n, = struct.unpack_from("<I", head, p)
    p += 4
    assert n == count
    out, cur = [], data_off
    for _ in range(n):
        ln, = struct.unpack_from("<H", head, p)
        name = head[p + 2:p + 2 + ln].decode()
        p += 2 + ln
        qt, nd = head[p], head[p + 1]
        shape = list(struct.unpack_from(f"<{nd}I", head, p + 2))
        p += 2 + 4 * nd + 4
        size, = struct.unpack_from("<Q", head, p)
        p += 8
        out.append((name, qt, shape, cur, size))
        cur += size
    return out


def raw(path, entry):
    """The tensor's bytes as a read-only u8 memmap."""
    return np.memmap(path, np.uint8, "r", offset=entry[3], shape=(entry[4],))


def mq4_rows(path, entry, rows=None, signed=True):
    """An MQ4G256 tensor as [rows][K / 256][136] u8 (a copy), nibbles xor 8 as the engine's loader stores them."""
    _, qt, shape, _, _ = entry
    assert qt == 44 and len(shape) == 2
    r = raw(path, entry).reshape(shape[0], shape[1] // 256, 136)
    r = np.array(r if rows is None else r[rows])
    if signed:
        r[..., 8:] ^= 0x88
    return r


if __name__ == "__main__":
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    for name, qt, shape, off, size in read(sys.argv[1]):
        if flt in name:
            print(f"{off:>12} {size:>11} qt={qt:<2} {shape} {name}")
