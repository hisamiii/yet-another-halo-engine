#!/usr/bin/env python3
"""Generate the vision Patchify fixture for yah_vision_patchify_bf16.loom.

Reference for Patchify in vision/encoder.hip. Element i of the patch tensor is

  patch   = i / 1536;  element = i % 1536
  merge   = patch / 4; y = (merge/(grid_width/2))*2 + (patch%4)/2
                       x = (merge%(grid_width/2))*2 + patch%2
  channel = element / 512; spatial = element % 512
  pixel   = ((y*16 + spatial/16)*width + x*16 + spatial%16)*3 + channel
  out[i]  = Bf16((float(pixels[pixel]) * (1/255) - 0.5) / 0.5)

The pixel buffer is an iota pattern (0..255 repeating) so the Loom case can generate
it with check.generate.iota and only the bf16 expectation needs a fixture. bf16 has
no numpy dtype, so it is written as raw 16-bit patterns in an int16 array and
compared through check.tensor.view.
"""
import os
import numpy as np

COUNT = 4
WIDTH = 32
GRID_WIDTH = 2
PATCH_ELEMS = 1536
PIXEL_BYTES = 4608
OUT = os.path.dirname(os.path.abspath(__file__))


def to_bf16_bits(values):
    u = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def main():
    pixels = (np.arange(PIXEL_BYTES) % 256).astype(np.uint8)
    inv255 = np.float32(1.0) / np.float32(255.0)
    half = GRID_WIDTH // 2
    patches = np.zeros(COUNT * PATCH_ELEMS, dtype=np.float32)
    for patch in range(COUNT):
        p4 = patch % 4
        merge = patch // 4
        my, mx = merge // half, merge % half
        y = my * 2 + p4 // 2
        x = mx * 2 + p4 % 2
        for channel in range(3):
            for spatial in range(512):
                sh, sl = spatial // 16, spatial % 16
                pixel = ((y * 16 + sh) * WIDTH + x * 16 + sl) * 3 + channel
                value = np.float32((np.float32(pixels[pixel]) * inv255
                                    - np.float32(0.5)) / np.float32(0.5))
                patches[patch * PATCH_ELEMS + channel * 512 + spatial] = value
    bits = to_bf16_bits(patches).astype(np.int16)
    np.save(os.path.join(OUT, 'expected_bits.npy'), bits)
    print('wrote expected_bits.npy shape=%s' % (bits.shape,))
    print('  raw values[:6]=%s' % patches[:6])
    print('  distinct values=%d' % int(np.unique(patches).size))


if __name__ == '__main__':
    main()
