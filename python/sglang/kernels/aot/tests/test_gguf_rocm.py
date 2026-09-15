# SPDX-License-Identifier: Apache-2.0
"""GGUF kernel tests for ROCm / gfx1151 (RDNA3.5, wave32).

Separate from test_gguf.py: these are wave32-specific and deliberately not in the
AMD CI allowlist, whose runners are MI300 (gfx942, wave64). Tolerances here are
three orders of magnitude tighter than test_gguf.py's, which accepts an all-zeros
output for every quant type and shape.
"""

import math
import sys

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES, GGUFReader
from gguf import GGMLQuantizationType as T
from gguf import dequantize as gguf_dequantize
from gguf import quants as gguf_quants

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="GGUF-on-ROCm enablement; CUDA is covered by test_gguf.py"
)

# The six ops share one translation unit, so a missing registration means a
# missing m.impl, not a missing kernel. See test_ops_registered.
GGML_OPS = (
    "ggml_dequantize",
    "ggml_mul_mat_a8",
    "ggml_mul_mat_vec_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
)

# Every ggml type present in unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf, by descending
# tensor count. Q8_0 backs ssm_alpha/ssm_beta, the only non-128-aligned weights.
MODEL_TYPES = (T.Q5_K, T.IQ4_XS, T.Q8_0, T.Q4_K, T.Q6_K, T.Q3_K, T.IQ4_NL, T.IQ3_S)
# MMQ_QUANT_TYPES in srt/layers/quantization/gguf.py excludes i-quants.
MODEL_MMQ_TYPES = (T.Q5_K, T.Q8_0, T.Q4_K, T.Q6_K, T.Q3_K)
# dequantize.cuh:286+ does genuine fp32 math for these; the k/legacy paths all
# end in convert_from_half<float>. test_dequantize_half_representable relies on
# the split, so it is a real discriminator rather than a tautology.
IQ_TYPES = (T.IQ4_XS, T.IQ4_NL, T.IQ3_S)
# The only two blocks carrying a separate dmin, so the only two whose dequantize
# evaluates d*q - dmin*m -- and it does so in fp16 (__hmul/__hsub). Q6_K's
# d*sc*(q-32) is symmetric and cancels nothing. See test_dequantize_ulp.
FP16_CANCELLING_TYPES = (T.Q4_K, T.Q5_K)

DTYPES = (torch.float16, torch.bfloat16)
MANTISSA_BITS = {torch.float16: 10, torch.bfloat16: 7}
FIXTURE_HIDDEN = 1024


@pytest.fixture(scope="session")
def gguf_sample_dir():
    # Session-scoped, never module-level: test_gguf.py:21 downloads at import and
    # takes the whole collection down when HF is unreachable.
    from huggingface_hub import snapshot_download

    return snapshot_download("Isotr0py/test-gguf-sample")


def load_fixture(sample_dir, quant_type, hidden_size=FIXTURE_HIDDEN, rows=None):
    """One packed tensor from the HF sample repo, optionally sliced to `rows`.

    GGUF quantizes along K within a row, so raw[:rows] is itself a valid tensor of
    that type -- verified bit-exact against gguf.dequantize for Q4_K/Q6_K/IQ4_XS/
    Q8_0 at rows=48 and 129. This is the only way to reach N not divisible by
    MMQ_Y=128 (mmq.cuh:194-204) for the k-quant and i-quant types, which
    gguf.quants.quantize cannot produce.
    """
    from pathlib import Path

    path = Path(sample_dir) / f"Quant_{quant_type.name}_{hidden_size}.gguf"
    raw = np.asarray(GGUFReader(path).tensors[0].data)
    if rows is not None:
        assert rows <= raw.shape[0], f"{quant_type.name} fixture has only {raw.shape[0]} rows"
        raw = np.ascontiguousarray(raw[:rows])
    ref = torch.from_numpy(gguf_dequantize(raw, quant_type).astype(np.float32))
    return torch.from_numpy(raw).cuda(), ref.cuda()


def make_gguf_weight(rows, cols, quant_type, seed=0):
    """Deterministic packed weight at an arbitrary shape, no network.

    Only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 are supported -- gguf.quants.quantize raises
    NotImplementedError for k-quants and i-quants. Its value over load_fixture is
    freedom in K (the model's real K is 5120; fixtures are 256/1024).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols), dtype=np.float32)
    packed = gguf_quants.quantize(w, quant_type)
    ref = torch.from_numpy(gguf_dequantize(packed, quant_type).astype(np.float32))
    return torch.from_numpy(packed).cuda(), ref.cuda()


def logical_cols(qweight, quant_type):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    return qweight.shape[1] // type_size * block_size


def ulp_diff(a, b):
    """Distance in representable steps between two same-dtype float tensors.

    assert_close cannot express "at most one rounding step"; for a deterministic
    integer unpack plus a scale multiply that is the honest tolerance.
    """
    assert a.dtype == b.dtype and a.shape == b.shape
    ia, ib = (t.contiguous().view(torch.int16).to(torch.int32) for t in (a, b))
    # int16 float bits are sign-magnitude; remap to a monotonic ordering.
    ka, kb = (torch.where(i < 0, -32768 - i, i) for i in (ia, ib))
    return (ka - kb).abs().max().item()


def ulp_at(magnitude, dtype):
    """Spacing of `dtype` at `magnitude` -- the size of one rounding step there."""
    return 2.0 ** (math.floor(math.log2(magnitude)) - MANTISSA_BITS[dtype])


def quantize_q8_1_reference(x):
    """The activation quantization gguf_kernel.cu:23-58 applies to x, in numpy.

    Needed by test_mmvq_permutation, which asserts an exact value rather than a
    statistical bound. d is stored via __float2half and read back as float, so the
    fp16 round is part of the contract, not a tolerance.
    """
    ntok, k = x.shape
    assert k % 32 == 0, "reference assumes blocks align to rows"
    xb = x.reshape(ntok, k // 32, 32)
    amax = np.abs(xb).max(axis=2, keepdims=True)
    d = amax / 127.0
    # roundf is half-away-from-zero; np.round is half-to-even.
    q = np.where(amax == 0.0, 0.0, np.trunc(xb / np.where(d == 0.0, 1.0, d) + np.copysign(0.5, xb)))
    d_half = d[..., 0].astype(np.float16).astype(np.float32)
    return q.astype(np.float32), d_half


def lossless_q8_1_activations(ntok, k, dtype, seed=0):
    """Activations whose q8_1 round-trip is the identity, so the matmul has an exact oracle.

    Values are n * 2^-7 with |n| <= 127, and element 0 of every 32-block is pinned to
    +-127 so gguf_kernel.cu's `d = amax/127` is exactly 2^-7 and `roundf(x/d)` is
    exactly n -- no rounding anywhere in the activation path.

    This matters most for Q4_K/Q5_K. quantize_q8_1 stores ds.y as the sum of the
    *unquantized* activations, so MMQ evaluates the -dmin*m*sum(x) term against exact
    x while the d*sc*dot term sees quantized x. A reference that rounds both mispredicts
    by dmin*m*(sum(q*d) - sum(x)), which is why those two types alone sat at rel_fro
    ~0.012 against a naive q8_1 reference while Q6_K/Q8_0/Q3_K sat at 0.0002.
    """
    # ds.y is stored via __float2half, and fp16 represents j*2^-7 exactly only up to
    # |j| = 2048. Bounding the 31 free slots to 61 caps a block sum at 127+31*61=2018
    # structurally; drawing them from the full +-126 exceeds 2048 in the tail, which
    # at ntok=2050 is ~65k blocks and so happens every run.
    rng = np.random.default_rng(seed)
    n = rng.integers(-61, 62, size=(ntok, k)).astype(np.float32)
    blocks = n.reshape(ntok, k // 32, 32)
    blocks[:, :, 0] = 127.0 * rng.choice([-1.0, 1.0], size=blocks.shape[:2])
    assert np.abs(blocks.sum(axis=2)).max() <= 2048
    return torch.from_numpy(n * 2.0**-7).cuda().to(dtype)


def matmul_metrics(out, ref):
    """Relative Frobenius error, float64 cosine similarity, max absolute error.

    No single scalar separates a correct kernel from a broken one here. Per-element
    relative error legitimately reaches ~1e5 because ref crosses zero -- that is why
    test_gguf.py:160 ended up at rtol=1e4, which then accepts all-zeros. Cosine alone
    returns 1.0000 for a dropped-half-warp-reduction mutant. The conjunction kills
    all seven mutants in the Phase 3 table.
    """
    o, r = out.double().flatten(), ref.double().flatten()
    rel_fro = ((o - r).norm() / r.norm()).item()
    cos = (o @ r / (o.norm() * r.norm())).item()
    max_abs = (o - r).abs().max().item()
    return rel_fro, cos, max_abs


# Measured with lossless_q8_1_activations: rel_fro 2.08e-4 (fp16) / 1.66e-3 (bf16),
# identical across all five MMQ types and equal to rounding `ref` to the output dtype.
# With activation quantization removed the kernel has no error of its own left, so
# these are ~3x the store rounding rather than a statistical allowance.
EXACT_REL_FRO = {torch.float16: 7e-4, torch.bfloat16: 5e-3}


def assert_matmul_exact(out, ref, dtype):
    """Gate for lossless activations: the kernel must be exact up to the output store."""
    rel_fro, cos, _ = matmul_metrics(out, ref)
    assert rel_fro <= EXACT_REL_FRO[dtype], f"relative Frobenius {rel_fro:.5g}"
    assert cos >= 0.99999, f"cosine {cos:.9g}"


def assert_matmul_close(out, ref, k):
    """Gate for Gaussian activations, where q8_1 quantization dominates the error.

    Deliberately loose: assert_matmul_exact is the correctness gate. This one exists
    because lossless activations pin amax to a fixed position in every block and use
    integer values, which could mask a bug in amax reduction or in roundf.
    """
    rel_fro, cos, max_abs = matmul_metrics(out, ref)
    # Worst of 40 seeds at the smallest output shape (7x48, where the ratio is noisiest)
    # is 0.0275, for Q5_K in bf16; without dmin it is 0.0044. An all-zeros output --
    # which test_gguf.py accepts for every type and shape -- gives exactly 1.0.
    assert rel_fro <= 0.05, f"relative Frobenius {rel_fro:.5g}"
    assert cos >= 0.9995, f"cosine {cos:.8g}"
    assert max_abs <= 0.03 * np.sqrt(k), f"max abs {max_abs:.5g} (K={k})"


# ---------------------------------------------------------------------------
# 1.0 Registration gate -- everything below is blocked on this
# ---------------------------------------------------------------------------


def test_ops_registered():
    """Distinguishes 'compiled the .cu' from 'registered the ops' -- two edits.

    Adding csrc/quantization/gguf/gguf_kernel.cu to setup_rocm.py's sources and
    adding the six m.impl lines to common_extension_rocm.cc are independent; doing
    only the first leaves every other test failing with an opaque AttributeError.
    """
    import sgl_kernel  # noqa: F401  -- registers the ops; torch.ops is lazy

    missing = [name for name in GGML_OPS if not hasattr(torch.ops.sgl_kernel, name)]
    assert not missing, (
        f"unregistered GGUF ops: {missing}. Add gguf_kernel.cu to setup_rocm.py "
        f"sources and the m.impl entries to csrc/common_extension_rocm.cc."
    )


def test_warp_size_is_32():
    """mmq.cuh uses WARP_SIZE_GGUF (32, hardcoded) while mmvq.cuh uses WARP_SIZE.

    WARP_SIZE is 64 on any ROCm build without -DSGL_ROCM_WARP_SIZE (utils.h:342-352);
    they agree only because setup_rocm.py special-cases gfx1151. On a wave64 part
    this file's expectations do not hold.
    """
    assert torch.cuda.get_device_properties(0).warp_size == 32


# ---------------------------------------------------------------------------
# 1.1 Dequantize -- exact oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", MODEL_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@torch.inference_mode()
def test_dequantize_ulp(gguf_sample_dir, quant_type, dtype):
    """Dequantize is an integer unpack times a scale, so it has an exact oracle.

    Replaces atol=1e-2/rtol=4e-2 (test_gguf.py:91), which is wide enough to pass a
    kernel that unpacks the wrong nibble.
    """
    from sgl_kernel import ggml_dequantize

    qweight, ref = load_fixture(gguf_sample_dir, quant_type)
    out = ggml_dequantize(qweight, quant_type, qweight.shape[0], logical_cols(qweight, quant_type), dtype)

    assert out.shape == ref.shape
    if quant_type in FP16_CANCELLING_TYPES:
        # d*q and dmin*m are both O(max|ref|) and cancel, so an element whose true
        # value is near zero is unbounded in ULP while still within a rounding step
        # of the operands. Measured 1.0 fp16 / 0.6 bf16 steps at full scale; the
        # bound stays ~20x below one quantization step, which is what a mis-unpacked
        # nibble would cost.
        scale = ref.abs().max().item()
        bound = 3 * ulp_at(scale, torch.float16) + ulp_at(scale, dtype)
        max_abs = (out.float() - ref).abs().max().item()
        assert max_abs <= bound, f"max abs {max_abs:.5g} > {bound:.5g}"
    else:
        # Every other path ends in convert_from_half with no cancelling subtraction,
        # so a correct kernel is at most one rounding step from the numpy reference.
        assert ulp_diff(out, ref.to(dtype)) <= 1
    torch.cuda.synchronize()


@pytest.mark.parametrize("quant_type", MODEL_TYPES, ids=lambda t: t.name)
@torch.inference_mode()
def test_dequantize_half_representable(gguf_sample_dir, quant_type):
    """Reference-free invariant: non-IQ fp32 output must survive an fp16 round-trip.

    The legacy and k-quant paths end in convert_from_half<float>, so every output is
    already an fp16 value. The i-quant paths (dequantize.cuh:286+) do genuine fp32
    math and must not be, which is what makes this discriminating -- a kernel that
    silently routed a k-quant through an fp32 path, or vice versa, fails here with
    no reference tensor involved.
    """
    from sgl_kernel import ggml_dequantize

    qweight, _ = load_fixture(gguf_sample_dir, quant_type)
    out = ggml_dequantize(
        qweight, quant_type, qweight.shape[0], logical_cols(qweight, quant_type), torch.float32
    )

    round_trips = torch.equal(out, out.half().float())
    if quant_type in IQ_TYPES:
        assert not round_trips, f"{quant_type.name} unexpectedly fp16-representable"
    else:
        assert round_trips, f"{quant_type.name} output is not an fp16 value"
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1.2 MMVQ (decode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ntok", (1, 2, 6, 7))
@torch.inference_mode()
def test_mmvq_permutation(ntok):
    """Structured probe: a one-hot Q8_0 weight turns the matmul into a permutation.

    Row r is zero except W[r, r % K] = 127.0, for which Q8_0 picks d = amax/127 = 1
    exactly. The output is then a known rearrangement of the q8_1-quantized input
    rather than a statistical blur, so a swapped lane or a dropped warp-reduction
    term moves a specific element instead of nudging a norm. Statistical tests
    cannot see this -- an index error inside a random matmul still produces a
    plausible-looking distribution.
    """
    from sgl_kernel import ggml_mul_mat_vec_a8

    k, rows = 512, 256
    w = np.zeros((rows, k), dtype=np.float32)
    w[np.arange(rows), np.arange(rows) % k] = 127.0
    packed = gguf_quants.quantize(w, T.Q8_0)
    # Confirm the scale assumption rather than trusting it.
    assert np.array_equal(gguf_dequantize(packed, T.Q8_0), w)

    rng = np.random.default_rng(1234)
    # Bounded away from zero so rtol alone is a meaningful assertion, and away
    # from exact .5 ties so roundf vs np.round cannot disagree.
    x = rng.uniform(0.5, 1.5, size=(ntok, k)).astype(np.float32)
    q, d = quantize_q8_1_reference(x)

    cols = np.arange(rows) % k
    # y[t, r] = d_W * d_x * sum_i(qw_i * qx_i), and only qw[r % K] is nonzero.
    expected = 127.0 * d[:, cols // 32] * q[:, cols // 32, cols % 32]

    out = ggml_mul_mat_vec_a8(
        torch.from_numpy(packed).cuda(), torch.from_numpy(x).cuda(), T.Q8_0, rows
    )
    torch.testing.assert_close(
        out.float().cpu(), torch.from_numpy(expected), rtol=1e-3, atol=0.0
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("quant_type", MODEL_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ntok", (1, 6, 7))
@torch.inference_mode()
def test_mmvq(gguf_sample_dir, quant_type, dtype, ntok):
    """ntok straddles the mmvq_safe boundary at srt/.../gguf.py:194-198 (2/6, 8/16)."""
    from sgl_kernel import ggml_mul_mat_vec_a8

    torch.manual_seed(ntok)  # test_gguf.py:48 declares SEEDS and never uses it
    qweight, w_ref = load_fixture(gguf_sample_dir, quant_type)
    k = logical_cols(qweight, quant_type)
    rows = qweight.shape[0]

    x = lossless_q8_1_activations(ntok, k, dtype, seed=ntok)
    out = ggml_mul_mat_vec_a8(qweight, x, quant_type, rows)
    assert_matmul_exact(out.float(), x.float() @ w_ref.T, dtype)

    x = torch.rand((ntok, k), dtype=dtype, device="cuda") - 0.5
    out = ggml_mul_mat_vec_a8(qweight, x, quant_type, rows)
    assert_matmul_close(out.float(), x.float() @ w_ref.T, k)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1.3 MMQ (prefill)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", MODEL_MMQ_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ntok", (7, 64, 65, 2050))
@pytest.mark.parametrize("rows", (48, 129, 768))
@torch.inference_mode()
def test_mmq(gguf_sample_dir, quant_type, dtype, ntok, rows):
    """rows=48 and 129 are not multiples of MMQ_Y=128, so need_check is true.

    mmq.cuh:194-204 picks need_check from nrows_x % mmq_y; the HF fixtures are all
    N=768, so the out-of-bounds guard path is dead under test today. rows=48 is not
    synthetic -- it is the real shape of ssm_alpha/ssm_beta, 96 of the 866 tensors
    in Qwen3.8-27B-UD-Q4_K_M.gguf, and it is smaller than one tile.
    """
    from sgl_kernel import ggml_mul_mat_a8

    torch.manual_seed(ntok * 31 + rows)
    qweight, w_ref = load_fixture(gguf_sample_dir, quant_type, rows=rows)
    k = logical_cols(qweight, quant_type)

    x = lossless_q8_1_activations(ntok, k, dtype, seed=ntok * 31 + rows)
    out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
    assert out.shape == (ntok, rows)
    assert_matmul_exact(out.float(), x.float() @ w_ref.T, dtype)

    x = torch.rand((ntok, k), dtype=dtype, device="cuda") - 0.5
    out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
    assert_matmul_close(out.float(), x.float() @ w_ref.T, k)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@torch.inference_mode()
def test_mmq_model_hidden_size(dtype):
    """K=5120 with a non-tile-aligned N, the one shape combination fixtures cannot reach.

    Fixtures cap K at 1024; the model's K is 5120/6144/10240/17408. Only Q8_0 among
    the model's types can be built offline, and it is also the type that carries the
    N=48 weights, so this is the real ssm_alpha shape end to end.
    """
    from sgl_kernel import ggml_mul_mat_a8

    torch.manual_seed(5120)
    k, rows, ntok = 5120, 48, 64
    qweight, w_ref = make_gguf_weight(rows, k, T.Q8_0, seed=7)

    x = lossless_q8_1_activations(ntok, k, dtype, seed=5120)
    out = ggml_mul_mat_a8(qweight, x, T.Q8_0, rows)
    assert out.shape == (ntok, rows)
    assert_matmul_exact(out.float(), x.float() @ w_ref.T, dtype)

    x = torch.rand((ntok, k), dtype=dtype, device="cuda") - 0.5
    out = ggml_mul_mat_a8(qweight, x, T.Q8_0, rows)
    assert_matmul_close(out.float(), x.float() @ w_ref.T, k)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1.5 Negatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_type", (0, 1, 15, 24, 30))
@torch.inference_mode()
def test_mul_mat_a8_rejects_unsupported_type(gguf_sample_dir, bad_type):
    """The MMQ switch has no default case, so an unknown type returns torch::empty.

    A fresh allocator block is usually zeroed, and all-zeros passes test_gguf.py's
    tolerances for every type and shape -- green tests, wrong inference output.
    """
    from sgl_kernel import ggml_mul_mat_a8

    qweight, _ = load_fixture(gguf_sample_dir, T.Q4_K)
    x = torch.rand((8, logical_cols(qweight, T.Q4_K)), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="unsupported"):
        ggml_mul_mat_a8(qweight, x, bad_type, qweight.shape[0])


@torch.inference_mode()
def test_mul_mat_a8_rejects_iquant(gguf_sample_dir):
    """i-quants reach fused_mul_mat_gguf's MMQ branch through a set-membership test
    with no kernel-side backstop. MMQ_QUANT_TYPES excludes them today; if that set
    ever gains one, the failure is silent rather than loud."""
    from sgl_kernel import ggml_mul_mat_a8

    qweight, _ = load_fixture(gguf_sample_dir, T.IQ4_XS)
    x = torch.rand((8, logical_cols(qweight, T.IQ4_XS)), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="unsupported"):
        ggml_mul_mat_a8(qweight, x, T.IQ4_XS, qweight.shape[0])


@pytest.mark.parametrize("bad_type", (15, 24, 26))
@torch.inference_mode()
def test_dequantize_rejects_unsupported_type(gguf_sample_dir, bad_type):
    """ggml_get_to_cuda returns nullptr (dequantize.cuh:581) and gguf_kernel.cu:87
    calls it unchecked. It is a host pointer, so this is a SIGSEGV that takes the
    whole pytest process with it rather than something pytest.raises can catch."""
    from sgl_kernel import ggml_dequantize

    qweight, _ = load_fixture(gguf_sample_dir, T.Q4_K)
    with pytest.raises(RuntimeError, match="unsupported"):
        ggml_dequantize(qweight, bad_type, qweight.shape[0], 1024, torch.float16)


@torch.inference_mode()
def test_mul_mat_vec_a8_rejects_mismatched_k(gguf_sample_dir):
    from sgl_kernel import ggml_mul_mat_vec_a8

    qweight, _ = load_fixture(gguf_sample_dir, T.Q4_K)
    x = torch.rand((1, logical_cols(qweight, T.Q4_K) // 2), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        ggml_mul_mat_vec_a8(qweight, x, T.Q4_K, qweight.shape[0])
        torch.cuda.synchronize()


@torch.inference_mode()
def test_mul_mat_vec_a8_rejects_cpu_activation(gguf_sample_dir):
    from sgl_kernel import ggml_mul_mat_vec_a8

    qweight, _ = load_fixture(gguf_sample_dir, T.Q4_K)
    x = torch.rand((1, logical_cols(qweight, T.Q4_K)), dtype=torch.float16)
    with pytest.raises(RuntimeError):
        ggml_mul_mat_vec_a8(qweight, x, T.Q4_K, qweight.shape[0])


# ---------------------------------------------------------------------------
# Mixed-type coverage
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_all_model_types_in_one_process(gguf_sample_dir):
    """Every ggml type in Qwen3.8-27B-UD-Q4_K_M.gguf, dispatched back to back.

    Nothing above covers the mixed-type case, and the model interleaves nine types
    across 866 tensors. A HIP launch failure is sticky and surfaces at the next CUDA
    call, so per-type tests in separate parametrizations can hide it: the poisoned
    context is torn down before the next case runs.
    """
    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

    torch.manual_seed(0)
    for quant_type in MODEL_TYPES:
        qweight, w_ref = load_fixture(gguf_sample_dir, quant_type)
        k = logical_cols(qweight, quant_type)
        for ntok in (1, 64):
            x = torch.rand((ntok, k), dtype=torch.float16, device="cuda") - 0.5
            out = fused_mul_mat_gguf(x, qweight, quant_type)
            assert_matmul_close(out.float(), x.float() @ w_ref.T, k)
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
