#!/usr/bin/env python3
"""Cut a tokenizer-only GGUF from the model: the header and every metadata pair as they are, no tensors.

The CPU tests (verify_tokenizer.py, yah_serve_test with yah_server_fake) read only the tokenizer and the config, so they
run on this ~10 MB file instead of the 15 GB model. The source is a local path or an http(s) URL: from a URL only the
metadata at the start is fetched (HTTP Range, in growing pieces), not the tensor index or the weights.

usage: make_tokenizer_fixture.py <model.gguf path or URL> <out.gguf>
"""
import struct
import sys
import urllib.request

# GGUF metadata value types: fixed sizes; 8 string, 9 array
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_range(src, start, end):
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise SystemExit(f"make_tokenizer_fixture: {src}: no range support (HTTP {r.status})")
            data = r.read()
    else:
        with open(src, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
    return data


class Head:
    """The file's first bytes, fetched further as the parser needs them."""

    def __init__(self, src):
        self.src, self.buf = src, b""

    def need(self, end):
        while len(self.buf) < end:
            want = max(end, 2 * len(self.buf), 1 << 22)
            more = read_range(self.src, len(self.buf), want)
            if not more:
                raise SystemExit(f"make_tokenizer_fixture: {self.src}: ends inside the metadata")
            self.buf += more

    def take(self, at, n):
        self.need(at + n)
        return self.buf[at:at + n]


def skip_value(h, at, kind):
    """the offset after a value of this kind at at"""
    if kind in SIZES:
        return at + SIZES[kind]
    if kind == 8:
        return at + 8 + struct.unpack("<Q", h.take(at, 8))[0]
    if kind == 9:
        sub, count = struct.unpack("<IQ", h.take(at, 12))
        at += 12
        if sub in SIZES:
            return at + count * SIZES[sub]
        for _ in range(count):
            at = skip_value(h, at, sub)
        return at
    raise SystemExit(f"make_tokenizer_fixture: unknown metadata type {kind}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    src, out = sys.argv[1], sys.argv[2]
    h = Head(src)
    magic, version, _, kv_count = struct.unpack("<4sIQQ", h.take(0, 24))
    if magic != b"GGUF" or version != 3:
        raise SystemExit(f"make_tokenizer_fixture: {src}: not a GGUF v3 file")
    at, alignment = 24, 32
    for _ in range(kv_count):
        klen = struct.unpack("<Q", h.take(at, 8))[0]
        key = h.take(at + 8, klen)
        kind = struct.unpack("<I", h.take(at + 8 + klen, 4))[0]
        value = at + 12 + klen
        if key == b"general.alignment" and kind == 4:
            alignment = struct.unpack("<I", h.take(value, 4))[0]
        at = skip_value(h, value, kind)
    h.need(at)
    blob = struct.pack("<4sIQQ", magic, version, 0, kv_count) + h.buf[24:at]   # 0 tensors, the metadata as it is
    blob += b"\0" * (-len(blob) % alignment)                                     # the (empty) data region's start
    with open(out, "wb") as f:
        f.write(blob)
    print(f"make_tokenizer_fixture: {out} ({len(blob)} bytes, {kv_count} metadata pairs)")


if __name__ == "__main__":
    main()
