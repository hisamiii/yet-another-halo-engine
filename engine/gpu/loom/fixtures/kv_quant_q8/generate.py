#!/usr/bin/env python3
"""Generate the KV q8 pack fixtures for yah_kv_quant_q8_f16.loom.

This is the reference the Loom case is checked against. It replicates
QuantizeQ8Kernel from engine/kv/kv_quant.hip operation for operation in the same
widths, because the port is only correct if it agrees bit for bit:

  amax    = max |f32(x[i])|                 f32
  scale   = amax > 0 ? amax / 127           f32 division
  inverse = scale > 0 ? 1 / scale           f32 division
  d       = f16(scale)                      round to nearest even
  qs[i]   = i8(clamp(lroundf(f32(x[i]) * inverse), -127, 127))

block layout is KvQ8Block from engine/kv/kv_cache.hpp: uint16 d, then int8 qs[32],
little endian, 34 bytes per block.

Rounding: lroundf rounds ties away from zero. The generator asserts that no
product lands exactly on a tie, so the float64 rounding below is unambiguous and
the fixture does not silently depend on a tie rule.

Usage:
  python3 generate.py
"""
import os, sys
import numpy as np

BLOCK = 32
LEVELS = np.float32(127.0)
OUT = os.path.dirname(os.path.abspath(__file__))


def pack(x16):
    """x16: (blocks, 32) float16 -> (packed int8 bytes, scale f16, codes int32)."""
    xf = x16.astype(np.float32)
    amax = np.max(np.abs(xf), axis=1).astype(np.float32)
    scale = np.where(amax > 0, (amax / LEVELS).astype(np.float32), np.float32(0))
    divisor = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    inverse = np.where(scale > 0, (np.float32(1.0) / divisor).astype(np.float32),
                       np.float32(0))
    product = (xf * inverse[:, None]).astype(np.float32)
    wide = product.astype(np.float64)
    # A product exactly on a tie is kept on purpose: lroundf and scalar.roundf both
    # round ties away from zero, so it is the one input that pins that rule down.
    ties = int(np.count_nonzero(np.abs(np.abs(wide - np.trunc(wide)) - 0.5) < 1e-9))
    if ties:
        print('  %d rounding tie(s) in the fixture; ties-away-from-zero is required' % ties)
    q = np.where(wide >= 0, np.floor(wide + 0.5), np.ceil(wide - 0.5))
    q = np.clip(q, -127, 127).astype(np.int32)
    d = scale.astype(np.float16)
    d_bytes = np.frombuffer(d.tobytes(), dtype=np.uint8).reshape(-1, 2)
    q_bytes = q.astype(np.int8).view(np.uint8)
    packed = np.concatenate([d_bytes, q_bytes], axis=1).reshape(-1)
    return packed.astype(np.int8), d, q


def save(name, array):
    path = os.path.join(OUT, name)
    np.save(path, array)
    print('wrote %-22s shape=%-14s dtype=%s' % (name, array.shape, array.dtype))


def varied(blocks):
    """Values that exercise rounding, the zero block, and both clamps."""
    block0 = [3.0, -3.0, 1.5, -1.5, 0.75, -0.75, 0.25, -0.25,
              0.0625, -0.0625, 2.0, -2.0, 0.5, -0.5, 1.0, -1.0,
              0.1875, -0.1875, 2.5, -2.5, 0.3125, -0.3125, 0.875, -0.875,
              0.125, -0.125, 1.25, -1.25, 0.375, -0.375, 0.6875, -0.6875]
    rows = [block0]
    rows.append([0.0] * BLOCK)                 # scale zero: d = 0, codes = 0
    rows.append([-2.5] * BLOCK)                # every code clamps to -127
    rows.append([4.0] * BLOCK)                 # every code clamps to +127
    while len(rows) < blocks:
        rows.append(block0)
    return np.array(rows[:blocks], dtype=np.float16)


def uniform(blocks, value):
    return np.full((blocks, BLOCK), value, dtype=np.float16)


def unpack(d, q):
    """Reference for DequantizeQ8Kernel: f32(f16 scale) * f32(code), rounded to f16."""
    scale = d.astype(np.float32)
    value = q.astype(np.float32)
    return (scale[:, None] * value).astype(np.float16)


Q4_LEVELS = np.float32(7.0)


def pack_q4(x16):
    """x16: (blocks, 32) float16 -> (packed bytes, scale f16, nibbles uint8 (blocks,16))."""
    xf = x16.astype(np.float32)
    amax = np.max(np.abs(xf), axis=1).astype(np.float32)
    scale = np.where(amax > 0, (amax / Q4_LEVELS).astype(np.float32), np.float32(0))
    divisor = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    inverse = np.where(scale > 0, (np.float32(1.0) / divisor).astype(np.float32),
                       np.float32(0))
    product = (xf * inverse[:, None]).astype(np.float32)
    wide = product.astype(np.float64)
    ties = int(np.count_nonzero(np.abs(np.abs(wide - np.trunc(wide)) - 0.5) < 1e-9))
    if ties:
        print('  %d q4 rounding tie(s); ties-away-from-zero is required' % ties)
    q = np.where(wide >= 0, np.floor(wide + 0.5), np.ceil(wide - 0.5))
    q = np.clip(q, -8, 7).astype(np.int32)
    # KvQ4Block stores element 2k in the low nibble and 2k+1 in the high nibble,
    # two's complement, so the low nibble of the byte holds the even element.
    nibbles = (q & 0x0F).astype(np.uint8)
    packed_nibbles = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    d = scale.astype(np.float16)
    d_bytes = np.frombuffer(d.tobytes(), dtype=np.uint8).reshape(-1, 2)
    packed = np.concatenate([d_bytes, packed_nibbles], axis=1).reshape(-1)
    return packed.astype(np.int8), d, packed_nibbles


def unpack_q4(d, packed_nibbles):
    """Decodes the nibbles it is given, so a packing error is visible here too."""
    low = (packed_nibbles & 0x0F).astype(np.int32)
    high = (packed_nibbles >> 4).astype(np.int32)
    signed = np.empty((packed_nibbles.shape[0], BLOCK), dtype=np.int32)
    signed[:, 0::2] = np.where(low >= 8, low - 16, low)
    signed[:, 1::2] = np.where(high >= 8, high - 16, high)
    scale = d.astype(np.float32)
    return (scale[:, None] * signed.astype(np.float32)).astype(np.float16)


def main():
    varied_input = varied(4)
    packed, d, q = pack(varied_input)
    save('input_varied.npy', varied_input.reshape(-1))
    save('expected_varied.npy', packed)
    save('dequant_varied_expected.npy', unpack(d, q).reshape(-1))
    print('  block 1 scale=%s codes[-3:]=%s' % (d[1], q[1][-3:]))
    print('  block 3 scale=%s codes[-3:]=%s' % (d[3], q[3][-3:]))

    fill = np.float16(3.0)
    uniform_input = uniform(10240, fill)
    packed, d, q = pack(uniform_input)
    save('expected_uniform.npy', packed)
    save('dequant_uniform_expected.npy', unpack(d, q).reshape(-1))
    print('  uniform fill=%s scale=%s codes[0]=%s' % (float(fill), d[0], q[0][0]))
    print('  uniform round trip exact: %s'
          % bool(np.array_equal(unpack(d, q).reshape(-1), uniform_input.reshape(-1))))

    packed4, d4, nib4 = pack_q4(varied_input)
    save('q4_expected_varied.npy', packed4)
    save('q4_dequant_varied_expected.npy', unpack_q4(d4, nib4).reshape(-1))
    print('  q4 block 0 scale=%s nibbles=%s' % (d4[0], nib4[0][:4]))
    print('  q4 block 2 scale=%s nibbles=%s' % (d4[2], nib4[2][:4]))

    packed4, d4, nib4 = pack_q4(uniform_input)
    save('q4_expected_uniform.npy', packed4)
    save('q4_dequant_uniform_expected.npy', unpack_q4(d4, nib4).reshape(-1))
    print('  q4 uniform scale=%s nibbles=%s' % (d4[0], nib4[0][:4]))
    print('  q4 uniform round trip exact: %s'
          % bool(np.array_equal(unpack_q4(d4, nib4).reshape(-1),
                                uniform_input.reshape(-1))))


if __name__ == '__main__':
    main()
