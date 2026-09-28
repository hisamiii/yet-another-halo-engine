#!/usr/bin/env python3
"""Generate the DeltaNet decode resident-recurrence fixture for
yah_deltanet_decode_resident_f32.loom.

Reference for DeltaNetRecurrenceKernel<float, Resident=true, WriteOutput=true>
in models/qwen/hip/kernels/deltanet_decode.hpp (the single-token decode
recurrence reached from LaunchSSMConvRecurrence).

One workgroup per value head h with kh = h % num_key_heads. The q/k/v row
conv_out is laid out q first, then k, then v:

  q[kh][i] = conv[(kh) * 128 + i]
  k[kh][i] = conv[(num_key_heads + kh) * 128 + i]
  v[h][j]  = conv[(2*num_key_heads + h) * 128 + j]

Per (head, value row j):

  alpha_in = alpha[h] + ssm_dt[h]
  alpha_h  = exp(softplus(alpha_in) * ssm_a[h]),  softplus(x) =
             x if x > 20 else log1p(exp(x))
  beta_h   = 1 / (1 + exp(-beta[h]))
  k_norm[i] = k[i] / sqrt(sum_l k[l]^2 + 1e-6)
  q_scale   = (1 / sqrt(128)) / sqrt(sum_l q[l]^2 + 1e-6)
  q_norm[i] = q[i] * q_scale
  u_j = sum_i (s[j][i] * alpha_h) * k_norm[i]
  d_j = (v[h][j] - u_j) * beta_h
  s_new[j][i] = s[j][i] * alpha_h + d_j * k_norm[i]
  o_j = sum_i s_new[j][i] * q_norm[i]
  mean_sq = sum_j o_j^2 / 128
  rms = 1 / sqrt(mean_sq + 1e-6)
  out[h][j] = o_j * rms * ssm_norm[j] * (g * sigmoid(g)),
              g = gate[h*128 + j]

This file computes the whole recurrence in float64 numpy from float32 inputs,
which is the oracle the Loom case compares against with a small tolerance; the
kernel runs in float32 (mul/add, no fma) so the two differ at the ulp scale.

The parameters are chosen so the recurrence is exercised and still bounded:
ssm_a is negative, ssm_dt and the alpha/beta inputs are small, and every
transcendental stays away from its overflow guard.
"""
import os
import numpy as np

HEADS = 4          # num_heads (value heads) = ssm_time_step_rank in production
KEY_HEADS = 2      # num_key_heads = ssm_group_count
KDIM = 128
VDIM = 128
INNER = HEADS * VDIM
QKV = 2 * KEY_HEADS * KDIM + HEADS * VDIM
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(20250519)
    conv = rng.uniform(-1.0, 1.0, QKV).astype(np.float32)
    alpha = rng.uniform(-1.0, 1.0, HEADS).astype(np.float32)
    beta = rng.uniform(-1.0, 1.0, HEADS).astype(np.float32)
    ssm_a = rng.uniform(-1.0, -0.05, HEADS).astype(np.float32)
    ssm_dt = rng.uniform(-0.5, 0.5, HEADS).astype(np.float32)
    ssm_norm = rng.uniform(0.5, 1.5, VDIM).astype(np.float32)
    gate = rng.uniform(-2.0, 2.0, INNER).astype(np.float32)
    state = rng.uniform(-0.5, 0.5, HEADS * KDIM * VDIM).astype(np.float32)

    conv64 = conv.astype(np.float64)
    out = np.zeros(INNER, dtype=np.float64)
    state64 = state.astype(np.float64).reshape(HEADS, VDIM, KDIM).copy()

    for h in range(HEADS):
        kh = h % KEY_HEADS
        q = conv64[kh * KDIM:(kh + 1) * KDIM]
        k = conv64[(KEY_HEADS + kh) * KDIM:(KEY_HEADS + kh + 1) * KDIM]
        v = conv64[(2 * KEY_HEADS + h) * KDIM:(2 * KEY_HEADS + h + 1) * KDIM]

        biased = float(alpha[h]) + float(ssm_dt[h])
        softplus = biased if biased > 20.0 else np.log1p(np.exp(biased))
        alpha_h = float(np.exp(softplus * float(ssm_a[h])))
        beta_h = 1.0 / (1.0 + float(np.exp(-float(beta[h]))))

        inv_k = 1.0 / np.sqrt(float(np.dot(k, k)) + 1e-6)
        inv_q = 1.0 / np.sqrt(float(np.dot(q, q)) + 1e-6)
        k_norm = k * inv_k
        q_norm = q * ((1.0 / np.sqrt(float(KDIM))) * inv_q)

        for j in range(VDIM):
            s_row = state64[h, j]
            u = float(np.dot(s_row * alpha_h, k_norm))
            d = (float(v[j]) - u) * beta_h
            s_new = s_row * alpha_h + d * k_norm
            state64[h, j] = s_new
            out[h * VDIM + j] = float(np.dot(s_new, q_norm))

        o = out[h * VDIM:(h + 1) * VDIM]
        mean_sq = float(np.dot(o, o)) / float(VDIM)
        rms = 1.0 / np.sqrt(mean_sq + 1e-6)
        g = gate[h * VDIM:(h + 1) * VDIM].astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-g))
        out[h * VDIM:(h + 1) * VDIM] = o * rms * ssm_norm.astype(np.float64) * (g * sig)

    np.save(os.path.join(OUT, 'input_conv.npy'), conv)
    np.save(os.path.join(OUT, 'input_alpha.npy'), alpha)
    np.save(os.path.join(OUT, 'input_beta.npy'), beta)
    np.save(os.path.join(OUT, 'input_ssm_a.npy'), ssm_a)
    np.save(os.path.join(OUT, 'input_ssm_dt.npy'), ssm_dt)
    np.save(os.path.join(OUT, 'input_ssm_norm.npy'), ssm_norm)
    np.save(os.path.join(OUT, 'input_gate.npy'), gate)
    np.save(os.path.join(OUT, 'input_state.npy'), state)
    np.save(os.path.join(OUT, 'expected_out.npy'), out.astype(np.float32))
    np.save(os.path.join(OUT, 'expected_state.npy'), state64.reshape(-1).astype(np.float32))

    print('heads=%d key_heads=%d qkv=%d state=%d' % (HEADS, KEY_HEADS, QKV, state.size))
    print('out[0:4]=%s' % out[:4])
    print('state[0,0,:4]=%s' % state64[0, 0, :4])


if __name__ == '__main__':
    main()
