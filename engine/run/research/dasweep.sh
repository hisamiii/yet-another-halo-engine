#!/usr/bin/env bash
# dasweep.sh: decode-ahead on vs off for every decode-ahead format x kind/shape (real bytes, 1 s gaps)
cd /home/q/yah-scratch/tools
L=/home/q/yet-another-halo-engine/engine/gpu/loom/tools; LC=/home/q/hrx/build/cmake/loom/src/loom/tools/loom-compile/loom-compile
# tag fmt kind mt kb gx weights input
K=( "i4k iq4xs kstore 1088 20 136 real_iq4_gate.bin real_l4_fn.bin"
    "i4sw iq4xs swiglu 1088 20 136 real_iq4_up.bin real_l4_fn.bin"
    "i4r68 iq4xs kres 320 68 40 real_iq4_down33.bin chain_S.bin"
    "q4k q4k kstore 640 20 80 real_q4k_qkv.bin real_l4_fn.bin"
    "q4sw q4k swiglu 1088 20 136 real_q4k_up.bin real_l4_fn.bin"
    "q4r68 q4k kres 320 68 40 real_q4k_down51.bin chain_S.bin"
    "q5k q5k kstore 768 20 96 real_q5k_q27.bin real_l4_fn.bin"
    "q5r24 q5k kres 320 24 40 real_q5k_ssmout0.bin chain_S6144.bin" )
for k in "${K[@]}"; do read -r tag f kind mt kb gx w in <<< "$k"
  S=yah_ffn_gemm_$f; [ $kind != kstore ] && S=${S}_$kind
  r=""
  for da in 1 0; do t=ds_${tag}_$da
    YAH_TG_DECAHEAD=$da YAH_TG_KIND=$kind python3 $L/gen_gemm_tile.py $f $PWD/$t.loom >/dev/null
    $LC $t.loom --root=@$S --target=amdgpu:gfx1151 --format=amdgpu-hsaco --output=$t.hsaco --config=$S.m_tiles=$mt --config=$S.k_blocks=$kb --config=$S.token_tiles=8 2>&1 | grep -m1 " error"
    c=$(SYM=$S FMT=$f WFILE=$w LH_INPUT=$in LH_GATE=chain_G.bin LH_RESID=real_l4_resid_f32.bin bash lcyc.sh $t $t.hsaco $gx 8 256 2>&1 | tail -1 | awk '{print $2}')
    r="$r da$da=$c"
  done
  echo "$tag ($f $kind ${mt}x$kb):$r"
done
