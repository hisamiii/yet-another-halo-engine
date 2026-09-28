#!/usr/bin/env python3
"""GGUF route audit: tensor type histogram grouped by name pattern.

The format histogram alone cannot say whether a format is on a compute route:
a dtype can own only embedding or SSM tensors and never reach the FFN GEMM.
This groups each tensor by its name with the layer index erased, so the FFN,
attention and SSM weight formats are separated.

usage: gguf_route_hist.py <model.gguf> [name-substring ...]
"""
import collections
import re
import struct
import sys

NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
         8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
         18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
         23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16"}
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main():
    path = sys.argv[1]
    filters = sys.argv[2:]
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, = struct.unpack("<I", f.read(4))
    tensor_count, = struct.unpack("<Q", f.read(8))
    kv_count, = struct.unpack("<Q", f.read(8))

    def read_string():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_value(t):
        if t in SCALAR:
            f.read(SCALAR[t])
            return
        if t == 8:
            n, = struct.unpack("<Q", f.read(8))
            f.read(n)
            return
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_value(et)
            return
        raise SystemExit("bad kv type %d" % t)

    for _ in range(kv_count):
        read_string()
        t, = struct.unpack("<I", f.read(4))
        skip_value(t)

    groups = collections.OrderedDict()
    for _ in range(tensor_count):
        name = read_string()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))
        pattern = re.sub(r"\d+", "N", name)
        groups.setdefault(pattern, collections.Counter())[ty] += 1
    print("version %d  tensors %d  kv %d" % (version, tensor_count, kv_count))
    for pattern, hist in groups.items():
        if filters and not any(x in pattern for x in filters):
            continue
        total = sum(hist.values())
        parts = "  ".join("%s x%d" % (NAMES.get(t, "?%d" % t), n)
                          for t, n in hist.most_common())
        print("%-46s total=%-4d %s" % (pattern, total, parts))


if __name__ == "__main__":
    main()