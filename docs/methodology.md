# Measurement methodology

These are the rules that make the numbers in [measurements.md](measurements.md) usable. They were learned the hard way: the first version of the token-split comparison read as a wash and was wrong.

## Only adjacent paired deltas are usable

At fixed settings this box drifts 15–35% between sessions, and the absolute throughput of a configuration is therefore not reproducible across time. A comparison is only meaningful when both arms are run in the same session, minutes apart. Do not compare a candidate against a baseline number recorded earlier.

## The controlled protocol

1. Warm up **every** arm before the loop starts, and discard those runs.
2. Run one repetition per arm per rep (`-r 1`), not an averaged run.
3. Alternate which arm goes first each rep, so no arm always follows the idle gap.
4. Put a fixed gap (15 s) before every timed run, and record the temperature with it.
5. Parse exactly one number out of the output with a sed expression that has been shown to produce a value; an extraction bug returns silence, not an error.

`-r 1` is the least state-dependent number: within a multi-repetition run the first repetition is the boosted one, so an average mixes regimes. Pairing removes the drift; averaging does not.

## The first run of a session is boosted

The first timed run after an idle period reads about 20% high. In the first token-split harness the column-split arm ran first, un-warmed, with `-r 3`, and took that boost in every rep; the result was an apparent wash (`-7.6%`, `+10.9%`, `+6.6%`). Adding the warm-up and the alternation separated the arms completely, with no overlap, at `+12.8%`.

## Use traces for mechanism, not just end-to-end

End-to-end throughput says whether a change won; it does not say why. A kernel trace with stream busy and idle time answers the mechanism question in one run. In this work the counter-intuitive fact that the winning configuration has a *larger* sum of ATB wait times only makes sense once the GPU's idle fraction and repack time are in the same table.

## Prefer the trace over the sweep when the question is "is the GPU idle?"

A sweep of one parameter answers where the optimum is; it does not reveal the resource that is saturated. The token-count sweep and the stream timeline answer different questions and were both needed.

## Record the protocol with the number

Prompt length, repetition count, arm order, warm-up and temperature all change the result. A number without its protocol, or compared against a number from a different protocol, is not evidence.
