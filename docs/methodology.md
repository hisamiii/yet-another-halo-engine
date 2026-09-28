# Measurement methodology

These are the rules that make the numbers in [measurements.md](measurements.md) usable. They were learned the hard way: the first version of the token-split comparison read as a wash and was wrong.

## Only adjacent paired deltas are usable

At fixed settings this box drifts 15–35% between sessions, and the absolute throughput of a configuration is therefore not reproducible across time. A comparison is only meaningful when both arms are run in the same session, minutes apart. Do not compare a candidate against a baseline number recorded earlier.

## The controlled protocol

1. Warm up **every** arm before the loop starts, and discard those runs.
2. Run one repetition per arm per rep (`-r 1`), not an averaged run.
3. Alternate which arm goes first each rep, so no arm always follows the idle gap.
4. Put a fixed gap (15 s) before every timed run, and record **SCLK and package
   power** with it, not just the temperature.
5. Parse exactly one number out of the output with a sed expression that has
   been shown to produce a value; an extraction bug returns silence, not an
   error.
6. Report work per clock (TF/MHz) next to the milliseconds. A wall-clock delta
   cannot on its own distinguish a kernel that does less work from one that is
   given more clock.

### The clock is part of the result, not noise

On this part the SMU chooses an operating point from how much work each clock
does, so an efficient kernel is rewarded twice: it finishes sooner *and* it runs
at a higher SCLK. A delta in milliseconds is therefore not attributable to
instructions alone, and a number recorded without the clock it ran at is
incomplete.

Both quantities are readable unprivileged:
`/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input` (SCLK, Hz) and
`power1_average` (W). The `clockrun.sh` sampler polls them at 10 Hz around a
timed run and reports mean and maximum SCLK, mean power, Tctl, TFLOPS and
TF/MHz.

Work per clock is the invariant. A 2048-token prefill on this box measured
**TF/MHz = 0.01406**, reproducing the **0.0144** recorded earlier by a different
harness to 2.4%, while that same run's SCLK sampled up to **2694 MHz** against a
2900 MHz top level and drew only **53.6 W mean against a 130 W cap**. So the cap
was not binding and the part was not near its top level on average — the
operating point was chosen, not clipped. That is what makes the 130 W and 80 W
"power limit" sweeps in [kernel-tuning.md](kernel-tuning.md) hard to read as a
ceiling: neither arm was at the cap.

The DPM table is not fixed either: `pp_dpm_sclk` reported level 1 as 827 MHz and
later as 1027 MHz for the same level, so the level's own frequency is
renegotiated. Sample the instantaneous clock; do not read it once.

`-r 1` is the least state-dependent number: within a multi-repetition run the first repetition is the boosted one, so an average mixes regimes. Pairing removes the drift; averaging does not.

## The first run of a session is boosted

The first timed run after an idle period reads about 20% high. In the first token-split harness the column-split arm ran first, un-warmed, with `-r 3`, and took that boost in every rep; the result was an apparent wash (`-7.6%`, `+10.9%`, `+6.6%`). Adding the warm-up and the alternation separated the arms completely, with no overlap, at `+12.8%`.

## Use traces for mechanism, not just end-to-end

End-to-end throughput says whether a change won; it does not say why. A kernel trace with stream busy and idle time answers the mechanism question in one run. In this work the counter-intuitive fact that the winning configuration has a *larger* sum of ATB wait times only makes sense once the GPU's idle fraction and repack time are in the same table.

## Prefer the trace over the sweep when the question is "is the GPU idle?"

A sweep of one parameter answers where the optimum is; it does not reveal the resource that is saturated. The token-count sweep and the stream timeline answer different questions and were both needed.

## Record the protocol with the number

Prompt length, repetition count, arm order, warm-up and temperature all change the result. A number without its protocol, or compared against a number from a different protocol, is not evidence.

## A GPU kernel cannot read the file mapping

The engine's weight table points straight into the GGUF's `mmap`. That is a
host page-cache mapping, not device memory, and it is not registered with HIP.
A kernel launched against it does not fault cleanly: it stalls the graphics
ring, the driver reports `ring gfx_0.0.0 timeout`, and on this part the reset
often fails, wedging the GPU until reboot. The failure is indistinguishable, in
the kernel log, from a bad kernel.

The reference engine avoids this by copying the tensor region into an anonymous
mapping, pinning it with `hipHostRegister(Mapped | ReadOnly)`, and repointing
every tensor at `hipHostGetDevicePointer`. The engine does the same in
`HipWeightRegion`, and the token gate then passes on the first run. Every hang
in the session that produced this note was the embedding kernel reading the
unregistered mapping, not the recurrence it was first blamed on.

The copy is not actually required. `hipHostRegister` refuses the tensor region
because it starts after the metadata and so is not page aligned, but registering
the *whole file mapping* from its page-aligned base and offsetting the device
pointer by the data region's start works, after `mprotect` widens the read-only
mapping to read-write (a private copy-on-write mapping; the GPU only reads, so
no page is copied). The engine does this by default now and falls back to the
copy only if the driver refuses the registration.

Measured peak RSS on the same 5-token run: the copy path peaks at **24.6 GiB**
because the file's source pages and the anonymous copy are resident at once
during the `memcpy`, and the registration pins only the copy. Direct
registration peaks at **12.4 GiB** for about 0.6 s more load time (7.9 s vs
7.3 s). That 12 GiB of headroom is what makes a long context plausible at all:
16 fp16 attention layers at 128k is 8.6 GiB of cache, and 12.4 + 8.6 fits where
24.6 + 8.6 does not.

The diagnostic that found it in one boot was a stage-synchronous mode
(`YAH_SYNC_STAGES`) writing an unbuffered line to a persistent file after every
kernel group: the last line before the hang names the kernel. `/tmp` is wiped
by the reboot the hang causes, so the log must live outside it.
