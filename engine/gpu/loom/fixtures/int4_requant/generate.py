#!/usr/bin/env python3
"""Reference for the Loom int4 requantizer (fixtures/int4_requant).

Builds one 16-token, one-block Q8_1 tile: block scale 0.25 and a -112..105 int8
ramp in the payload, with the matching scale and sum sidecar. Then applies the
kernel transform in place: d4 = d*127/7, round each decoded value to nearest even
and clamp to [-7, 7], rewrite the payload, replace the scale, refresh the sums."""
import numpy as np

BATCH = 16


def main():
    d = np.float32(0.25)
    q = np.array([(i - 16) * 7 for i in range(32)], dtype=np.int8)

    data = np.zeros(704, dtype=np.uint8)
    for tl in range(BATCH):
        for i in range(32):
            half = i // 16
            pos = i % 16
            data[half * 256 + tl * 16 + pos] = q[i].astype(np.uint8)
        data[512 + tl * 4:512 + tl * 4 + 4] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        lo = int(np.sum(q[0:16].astype(np.int64)))
        hi = int(np.sum(q[16:32].astype(np.int64)))
        slot0 = np.float32(d * np.float32(lo + hi))
        slot1 = np.float32(d * np.float32(lo))
        base = 576 + tl * 8
        data[base:base + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        data[base + 4:base + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)

    level = np.float32(d * np.float32(127.0))
    d4 = np.float32(level / np.float32(7.0))
    a = (q.astype(np.float32) * d).astype(np.float32)
    v = np.rint((a / d4).astype(np.float32)).astype(np.int64)
    v = np.clip(v, -7, 7).astype(np.int8)

    expected = data.copy()
    for tl in range(BATCH):
        for i in range(32):
            half = i // 16
            pos = i % 16
            expected[half * 256 + tl * 16 + pos] = v[i].astype(np.uint8)
        expected[512 + tl * 4:512 + tl * 4 + 4] = np.frombuffer(d4.tobytes(), dtype=np.uint8)
        lo = int(np.sum(v[0:16].astype(np.int64)))
        hi = int(np.sum(v[16:32].astype(np.int64)))
        slot0 = np.float32(d4 * np.float32(lo + hi))
        slot1 = np.float32(d4 * np.float32(lo))
        base = 576 + tl * 8
        expected[base:base + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        expected[base + 4:base + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)

    np.save("input_bytes.npy", data.view(np.int8))
    np.save("expected_bytes.npy", expected.view(np.int8))
    print("d4", d4, "v", list(v))


if __name__ == "__main__":
    main()
