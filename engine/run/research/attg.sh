#!/usr/bin/env bash
# attg.sh <tag> <hsaco> <sym> <fmt> <weights> <gx> <bx>: ATT of one GEMM dispatch (gy 8)
cd /home/q/yah-scratch/tools; tag=$1 h=$2 S=$3 f=$4 w=$5 gx=$6 bx=$7
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
rm -rf att-$tag; BUFS=$(LH_GATE=$PWD/chain_G.bin LH_INPUT=$PWD/real_l4_fn.bin python3 lhargs.py ${h%.hsaco}.loom $f $PWD/$w z:142606336) || exit 3
sleep 1; $GR att-$tag -- $TR/bin/rocprofv3 --att --att-library-path $TR/lib --kernel-include-regex "$S\$" -d att-$tag -o run -- $LH $PWD/$h $S $gx 8 $bx 1 $BUFS > att-$tag.log 2>&1
grep -E "\*\*\*|refus" att-$tag.log; dirname $(find att-$tag -name 'code.json' | head -1)
