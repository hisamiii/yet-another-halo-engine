# yah for gfx1151

Qwen3.8-27B (the MQ4 HFQ build) served on a Ryzen AI Max (Strix Halo, gfx1151) GPU, with an OpenAI-compatible API. Everything it loads is in this directory: the HRX runtime, AMD's ROCm 10 HSA runtime and the HAL sets. No ROCm install, no root.

Needs:
- a gfx1151 GPU with a recent kernel's amdgpu driver (tested on Linux 7.3)
- glibc 2.39 or later (Ubuntu 24.04, Fedora 40, Debian 13, Arch)
- read and write access to `/dev/kfd` and `/dev/dri/renderD*`: membership in the `render` group on most distros
- about 17 GB of GPU-visible memory: the model (15 GB) and a 32K-token kv8 cache. On a 32 GB machine, set the BIOS GPU carve-out low and let the GTT take it.
- the model file `qwen3.8-27b.mq4-xts`

## Run

```
bin/yah_server --model qwen3.8-27b.mq4-xts [--host 127.0.0.1] [--port 8080]
```

It is ready when `curl localhost:8080/health` answers. The first start takes about a minute.

```
bin/yah_chat --url http://127.0.0.1:8080     # a terminal chat client
```

Any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

The server takes:
- `POST /v1/chat/completions` and `POST /v1/responses`, both plain or with `stream: true`
- `GET /v1/models` and `GET /health`

Reasoning effort: `reasoning_effort` for chat completions, `reasoning.effort` for responses. Sampling: `temperature`, `top_p`, `top_k`, `min_p`, the presence, frequency and repetition penalties, `seed` and `stop`. A follow-up turn reuses the earlier turns' cache.

The HAL sets in `hal/` give 32K tokens of context, with an 8-bit KV cache (`hal/32768-kv8/`; `--ctx` / `--kv` pick another pair a bundle holds). One request runs at a time; others wait.

## The NPU

`--npu` runs part of each layer's FFN on the XDNA2 NPU during prompt processing: long prompts (two chunks, 4096 tokens or more) get about 10% faster. It needs:
- the `amdxdna` driver, with read and write access to `/dev/accel/accel0`
- an unlimited memlock limit (`<user> - memlock unlimited` in `/etc/security/limits.d/90-yah.conf`, then log in again)
- the GPU's DPM level pinned to `high`: `echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level` (`auto` restores it). Clock switches while the NPU runs corrupt its outputs.

Without one of them the server says why (`yah_server: npu off: ...`) and runs on the GPU alone. The start takes about 3 s longer: the NPU's down weights are packed from the model.

## If something is off

- `hfq: cannot lock the model in memory (RLIMIT_MEMLOCK)` is a warning: the model stays mapped but may be paged out under memory pressure. To lock it, raise the memlock limit, for example a line `<user> - memlock unlimited` in `/etc/security/limits.d/90-yah.conf`, then log in again.
- No GPU found: check `ls -l /dev/kfd` and your groups (`id`).
- `IREE_HAL_AMDGPU_LIBHSA_PATH` set in the environment overrides the bundled HSA runtime.

Licenses of the bundled parts are in `licenses/`.
