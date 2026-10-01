#!/usr/bin/env bash
# lhcmd.sh <hsaco> <fmt> <weights> <gx> [iters]: exec loomhip on a kstore tile kernel with signature-derived buffers (for wrapping by profilers)
cd /home/q/yah-scratch/tools; h=$1; fmt=$2; wf=$3; gx=$4; it=${5:-5}
BUFS=$(python3 lhargs.py ${h%.hsaco}.loom $fmt $wf z:142606336) || exit 3
exec /home/q/yet-another-halo-engine/engine/build/loomhip $h ${SYM:-yah_ffn_gemm_$fmt} $gx ${GY:-8} ${BX:-512} $it $BUFS
