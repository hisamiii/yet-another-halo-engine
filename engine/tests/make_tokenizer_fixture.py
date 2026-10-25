#!/usr/bin/env python3
"""Cut a tokenizer-only HFQ from the model: header, the metadata the tokenizer and config need, an empty tensor index.

The CPU tests (verify_tokenizer.py, yah_serve_test with yah_server_fake) read only the tokenizer, so they run on this
~14 MB file instead of the 15 GB model. The source is a local path or an http(s) URL: from a URL only the header and
the metadata ranges are fetched (HTTP Range), not the weights.

usage: make_tokenizer_fixture.py <model path or URL> <out.hfq>
"""
import json
import struct
import sys
import urllib.request

HEADER = struct.Struct("<4sIIIQQ")   # magic, version, arch, tensor count, metadata offset, data offset (core/hfq.cpp)
KEEP = ("architecture", "config", "generation_config", "mq4v2.symmetric", "tokenizer", "tokenizer_config")


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
    if len(data) != end - start:
        raise SystemExit(f"make_tokenizer_fixture: {src}: short read at {start}")
    return data


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    src, out = sys.argv[1], sys.argv[2]
    magic, version, arch, _, meta_off, data_off = HEADER.unpack(read_range(src, 0, HEADER.size))
    if magic != b"HFQM":
        raise SystemExit(f"make_tokenizer_fixture: {src}: not an HFQ file")
    head = read_range(src, meta_off, data_off).decode("utf-8", "surrogateescape")
    meta, _ = json.JSONDecoder().raw_decode(head)
    blob = json.dumps({k: meta[k] for k in KEEP if k in meta}, ensure_ascii=False, separators=(",", ":"))
    blob = blob.encode("utf-8", "surrogateescape") + struct.pack("<I", 0)   # the metadata, then 0 tensors
    with open(out, "wb") as f:
        f.write(HEADER.pack(magic, version, arch, 0, HEADER.size, HEADER.size + len(blob)) + blob)
    print(f"make_tokenizer_fixture: {out} ({HEADER.size + len(blob)} bytes)")


if __name__ == "__main__":
    main()
