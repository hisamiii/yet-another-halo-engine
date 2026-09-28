#!/usr/bin/env bash
# Builds the HIP tools. The kernel translation units are compiled and linked
# separately, exactly as the reference engine builds them, so no kernel header
# is included in more than one translation unit. Objects are cached by
# timestamp and compiled in parallel; a few kernels need per-file device flags,
# which is why this is not a single hipcc invocation.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ported="$root/engine/gpu/ported"
out="$root/engine/build"
objdir="$out/obj"
mkdir -p "$out" "$objdir"

kernels=(
  prefill_embed prefill_gemm prefill_quant_gemm prefill_fp16
  prefill_quant_wave64 small_batch_wave64 small_batch_quant16_wave64
  prefill_norm prefill_residual prefill_rope prefill_ssm prefill_swiglu
  prefill_unpack embed residual unpack rope sample norm gemv gemv_quant
  fused qkv ssm ssm_decode_recurrence ssm_recurrence ssm_row_split swiglu
  attention_batched attention_decode attention_decode_graph attention_tile
  attention_wmma hipblas_gemm batched_ssm
)

incs=(-I"$ported" -I"$root/engine")
base=(-std=c++20 -O3 -DENGINE_ENABLE_HIP=1 --offload-arch=gfx1151 "${incs[@]}")

extra_flags() {
  case "$1" in
    prefill_quant_wave64|small_batch_wave64) echo "-mwavefrontsize64" ;;
    small_batch_quant16_wave64)
      echo "-mwavefrontsize64 -gline-tables-only -Xarch_device -mllvm=-amdgpu-sched-strategy=iterative-ilp" ;;
    *) echo "" ;;
  esac
}

pids=()
wait_one() {
  wait "${pids[0]}"
  pids=("${pids[@]:1}")
}
source_for() {
  case "$1" in
    batched_ssm) echo "$ported/src/models/qwen/hip/batched_ssm.hip" ;;
    *) echo "$ported/src/models/qwen/hip/kernels/$1.hip" ;;
  esac
}

# Rebuild when any prerequisite is newer than the object, not just the
# translation unit. Comparing against the .hip alone silently reused a stale
# object after an included header changed: an A/B whose whole point was a new
# code path in model/forward.hip ran the previous binary twice and read as
# "the change has no effect". -MMD records the real prerequisite list.
stale() {
  local obj="$1" dep="$2" prereq
  [[ -f "$obj" ]] || return 0
  [[ -f "$dep" ]] || return 0
  while read -r prereq; do
    if [[ -n "$prereq" && -e "$prereq" && "$prereq" -nt "$obj" ]]; then
      return 0
    fi
  done < <(tr ' ' '\n' < "$dep" | tr -d '\\' | grep -v ':' | grep -v '^$')
  return 1
}

compile() {
  local src="$1" name="$2"
  local obj="$objdir/$name.o" dep="$objdir/$name.d"
  if ! stale "$obj" "$dep"; then return 0; fi
  local flags; flags="$(extra_flags "$name")"
  rm -f "$obj"
  # shellcheck disable=SC2086
  hipcc -c "${base[@]}" $flags -MMD -MF "$dep" -MT "$obj" "$src" -o "$obj" &
  pids+=("$!")
  if (( ${#pids[@]} >= 8 )); then wait_one; fi
}

target="${1:-yah-run}"
standalone=0
case "$target" in
  yah-run) main="$root/engine/run/yah_run.hip"; name="yah_run" ;;
  kv-quant-check) main="$root/engine/kv/kv_quant_check.hip"; name="kv_quant_check" ;;
  # The bench translation units include the kernel headers directly, so their
  # objects define symbols the kernel objects and libyah_core.a also define.
  # Linking them the normal way is a duplicate-symbol error; they are their own
  # program and need only their own object plus the quant dequantiser.
  gemm_bench|dot_peak|klook_replica|launch_bench|gridsync_bench)
    main="$root/engine/gpu/$target.hip"; name="$target"; standalone=1 ;;
  *) main="$root/engine/gpu/$target.hip"; name="$target" ;;
esac

if [[ "$standalone" == 0 ]]; then
  for k in "${kernels[@]}"; do
    compile "$(source_for "$k")" "$k"
  done
fi
compile "$ported/src/core/quant/ggml_dequant.cpp" "ggml_dequant"
compile "$main" "$name"
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ "$standalone" == 1 ]]; then
  printf 'build_gpu: linking %s (standalone)\n' "$target"
  hipcc "$objdir/$name.o" "$objdir/ggml_dequant.o" -o "$out/$target" \
    -lhipblas -licuuc -lpthread
else
  objs=()
  for k in "${kernels[@]}"; do objs+=("$objdir/$k.o"); done
  objs+=("$objdir/ggml_dequant.o" "$objdir/$name.o")

  printf 'build_gpu: linking %s (%d objects)\n' "$target" "${#objs[@]}"
  hipcc "${objs[@]}" "$out/libyah_core.a" -o "$out/$target" \
    -lhipblas -lhipblaslt -licuuc -lpthread
fi
