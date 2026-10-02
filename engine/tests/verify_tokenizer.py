#!/usr/bin/env python3
"""Check yah-tokenize against golden token-id hashes for Qwen3.8-27B.

Each golden is the SHA-256 of the token ids packed little-endian u32, which is
the exact output a correct byte-level BPE must produce. It is a stronger check
than a token count and it needs no reference implementation at run time.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf")
    parser.add_argument(
        "--binary", default=str(ROOT / "engine" / "build" / "yah-tokenize"))
    parser.add_argument("--goldens", default=str(HERE / "tokenizer_goldens.json"))
    args = parser.parse_args()

    with open(args.goldens, encoding="utf-8") as handle:
        goldens = json.load(handle)

    failures = 0
    for case in goldens["cases"]:
        proc = subprocess.run(
            [args.binary, args.model, "--stdin"],
            input=case["text"].encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            print(f"FAIL {case['name']}: {proc.stderr.decode().strip()}")
            failures += 1
            continue
        # The decoded line can contain raw non-UTF-8 bytes (byte-level tokens decode to single bytes).
        # So stay in bytes and read only the two lines that carry the result.
        lines = proc.stdout.split(b"\n")
        count = int(lines[0].split(b"=")[1])
        ids = [int(value) for value in lines[1].split()] if count else []
        digest = hashlib.sha256(
            b"".join(struct.pack("<I", value) for value in ids)).hexdigest()
        ok = count == case["token_count"] and digest == case["sha256"]
        print(("PASS" if ok else "FAIL"), case["name"],
              f"{count}/{case['token_count']} tokens")
        if not ok:
            print("     got ", digest)
            print("     want", case["sha256"])
            failures += 1

    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
