#!/bin/bash
# Phase 3 mutation gate: each one-line change must turn the named tests red.
# A mutant that survives means the test suite does not constrain that code path.
set -u
cd "$(dirname "$0")/.."
G=csrc/quantization/gguf
BK=/tmp/gguf_mutation_backup
rm -rf $BK && mkdir -p $BK && cp $G/*.cuh $G/*.h $BK/

restore() { cp $BK/*.cuh $BK/*.h $G/ ; }
trap restore EXIT

build() {
  rm -f $G/*_hip.cuh $G/*_hip.h $G/gguf_kernel.hip
  python3 setup_rocm.py install > /tmp/mut_build.log 2>&1
}

run_mutant() {  # name, expectation, pytest -k filter
  local name=$1 expect=$2 filter=$3
  if ! build; then
    echo "RESULT $name: BUILD FAILED  (expected: $expect)"
    grep -m2 -E "error:|static_assert" /tmp/mut_build.log | sed 's/^/    /'
    return
  fi
  local out
  out=$(timeout 2400 python3 -m pytest tests/test_gguf_rocm.py -q --timeout=300 -k "$filter" 2>&1 | tail -1)
  echo "RESULT $name: $out   (expected: $expect)"
}

echo "=== baseline (unmutated) is 212 passed ==="

# 1. Halve the mmvq warp reduction: lanes 16-31 never fold in.
sed -i 's|for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)|for (int mask = WARP_SIZE / 4; mask > 0; mask >>= 1)|' $G/mmvq.cuh
run_mutant "mmvq half warp reduction" "mmvq FAILS" "mmvq or permutation"
restore

# 2. Swap the Q4_K nibbles. The plan named dequantize_q4_0, but Q4_0 is neither in
#    Qwen3.8-27B nor in MODEL_TYPES, so that mutant is unconstrained by design.
python3 - <<'PY'
p = 'csrc/quantization/gguf/dequantize.cuh'
s = open(p).read()
a = """    y[l + 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn(q[l] & 0xF)), m1));
    y[l + 32] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn(q[l] >> 4)), m2));"""
b = """    y[l + 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn(q[l] >> 4)), m1));
    y[l + 32] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn(q[l] & 0xF)), m2));"""
assert s.count(a) == 1
open(p, 'w').write(s.replace(a, b))
PY
run_mutant "q4_K nibble swap" "dequantize FAILS" "dequantize"
restore

# 3. Neutralize dp4a: the integer dot product becomes the accumulator.
sed -i 's|  c = __builtin_amdgcn_sdot4(a, b, c, false);|  c = c;|' $G/ggml-common.h
run_mutant "dp4a returns c" "mmvq and mmq FAIL" "mmvq or mmq or permutation"
restore

# 4. WARP_SIZE_GGUF disagrees with WARP_SIZE. Must be caught, at build time by the
#    static_assert in gguf_kernel.cu rather than at runtime.
sed -i 's|#define WARP_SIZE_GGUF 32|#define WARP_SIZE_GGUF 64|' $G/ggml-common.h
run_mutant "WARP_SIZE_GGUF 32->64" "BUILD FAILED" "warp"
restore

# 5. Force the portable fallbacks over the amdgcn intrinsics. This one must change
#    NOTHING -- a failure means __builtin_amdgcn_sdot4 miscompiles on gfx1151.
sed -i 's|#if __has_builtin(__builtin_elementwise_sub_sat)|#if 0  // mutation|' $G/ggml-common.h
sed -i 's|#if __has_builtin(__builtin_amdgcn_sdot4)|#if 0  // mutation|' $G/ggml-common.h
run_mutant "portable vs intrinsic" "ALL PASS (212)" ""
restore

build && echo "=== restored and rebuilt ==="
