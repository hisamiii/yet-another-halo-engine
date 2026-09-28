#!/usr/bin/env python3
"""Reference for the Loom DFlash grouped dynamic conv (fixtures/dflash_conv)."""
import numpy as np


def main():
    num_tokens, hidden, kernel, group_size, side = 2, 4, 2, 2, 0
    groups = hidden // group_size
    coeff = 2 * kernel * groups
    inp = np.ones(num_tokens * hidden, dtype=np.float32)
    dyn = np.arange(num_tokens * coeff, dtype=np.float32)
    base = np.arange(2 * kernel * hidden, dtype=np.float32)
    out = np.zeros(num_tokens * hidden, dtype=np.float32)
    for index in range(num_tokens * hidden):
        token = index // hidden
        channel = index % hidden
        grp = channel // group_size
        s = np.float32(0)
        for tap in range(kernel):
            if token < tap:
                continue
            di = token * coeff + (side * kernel + tap) * groups + grp
            bi = (side * kernel + tap) * hidden + channel
            ii = (token - tap) * hidden + channel
            s = np.float32(s + np.float32(base[bi] + dyn[di]) * np.float32(inp[ii]))
        out[index] = s
    np.save("expected.npy", out)
    print(out)


if __name__ == "__main__":
    main()
