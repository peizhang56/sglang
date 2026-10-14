"""Head-tile selection for the DSA fp8 partial decode kernel.

Guards ``_pick_h_per_block``, which replaced a hardcoded ``h_per_block = 16``.
The fp8 partial kernel's KV gather indexes ``indices[b_i, s_i, g_i, ...]`` with
no head dependence, so the ``num_heads / h_per_block`` head-blocks of one token
gather byte-identical KV rows -- a smaller tile multiplies gather traffic by
exactly that factor.  Larger is not free: the grid is
``seq * ceil(num_heads/h) * n_groups``, so a big tile can starve a small decode
batch.

The function is pure Python, so this is a cheap unit test of the policy.  Two
properties matter and are asserted separately:

* **Safety** -- the kernel asserts ``num_heads <= h_per_block or num_heads %
  h_per_block == 0``.  A pick that violates it is a crash, so that is swept
  over a wide range of head counts rather than spot-checked.
* **Policy** -- the largest tile whose grid still fills the GPU.
"""

import sys

import pytest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="tilelang_kernel only imports on ROCm"
)

# gfx950 (MI355X) fp8 path: block_I, threads, block_per_cu, cu = 64, 256, 2, 256
CU = 256
BLOCK_PER_CU = 2
# GLM-5.3-Flash: topk 2048, block_I 64.
TOPK = 2048
BLOCK_I = 64
NUM_HEADS = 64

TILES = (64, 32, 16)


def _pick(seq, num_heads, n_groups, cu=CU, block_per_cu=BLOCK_PER_CU):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_h_per_block

    return _pick_h_per_block(seq, num_heads, n_groups, cu, block_per_cu)


def _grid(seq, num_heads, h, n_groups):
    """The kernel's real grid: T.Kernel(seq_len * head_blocks_per_seq, n_groups)."""
    head_blocks = (num_heads + h - 1) // h
    return seq * head_blocks * n_groups


def _kernel_assert_holds(num_heads, h):
    """The assert inside sparse_mla_fwd_decode_partial_fp8."""
    return num_heads <= h or num_heads % h == 0


# --------------------------------------------------------------------------
# Safety: a pick must never trip the kernel's own assert.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192])
@pytest.mark.parametrize("seq", [1, 12, 80, 512, 14472])
@pytest.mark.parametrize("n_groups", [1, 8, 32])
def test_pick_never_violates_the_kernel_assert(num_heads, seq, n_groups):
    h = _pick(seq, num_heads, n_groups)
    assert h in TILES
    assert _kernel_assert_holds(num_heads, h), (
        f"num_heads={num_heads} h={h} would trip the kernel assert"
    )


def test_every_head_count_up_to_64_is_safe_at_any_grid_size():
    """Regression: the fallback must itself satisfy the kernel assert.

    A bare ``return 16`` is unsafe for a num_heads over 16 that is not a
    multiple of it (24, 40, 56, ...): every tile is skipped by the divisibility
    guard and the fallback trips the assert the guard exists to protect.
    Swept rather than spot-checked because the failure is a hard crash.
    """
    for num_heads in range(1, 65):
        for seq in (1, 2, 12, 48, 80, 256, 1024, 14472):
            for n_groups in (1, 2, 8, 32):
                h = _pick(seq, num_heads, n_groups)
                assert _kernel_assert_holds(num_heads, h), (
                    f"num_heads={num_heads} seq={seq} n_groups={n_groups} -> h={h}"
                )


def test_head_counts_above_64_must_be_a_multiple_of_16():
    """Documents the kernel's inherent limit, which the picker cannot widen.

    For num_heads > 64 that is not a multiple of 16, NO tile in (64, 32, 16)
    satisfies ``num_heads <= h or num_heads % h == 0``.  The geometry is
    unsupported and the kernel's own assert is the right guard; the picker is
    not expected to manufacture a safe tile.
    """
    for num_heads in (65, 70, 100, 130):
        assert not any(_kernel_assert_holds(num_heads, h) for h in TILES)
    for num_heads in (80, 96, 128, 192):
        h = _pick(14472, num_heads, 1)
        assert _kernel_assert_holds(num_heads, h)


# --------------------------------------------------------------------------
# Policy: largest tile that still fills the GPU.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seq", [1, 4, 12, 48, 80, 256, 1024, 14472])
@pytest.mark.parametrize("n_groups", [1, 2, 8, 32])
def test_pick_is_the_largest_tile_that_fills_the_grid(seq, n_groups):
    """Reference implementation of the documented policy."""
    want = 16
    for h in TILES:
        if NUM_HEADS % h and NUM_HEADS > h:
            continue
        if _grid(seq, NUM_HEADS, h, n_groups) >= CU * BLOCK_PER_CU:
            want = h
            break
    assert _pick(seq, NUM_HEADS, n_groups) == want


def test_documented_production_shapes():
    """The three shapes the docstring names, through the real inner-iter pick.

    Recomputing ``n_groups`` the way ``tilelang_sparse_fwd`` does keeps this
    honest: if ``_pick_inner_iter`` changes, this test moves with it.
    """
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter

    ni = TOPK // BLOCK_I
    for seq, expected in ((14472, 64), (80, 64), (12, 32)):
        inner_iter = _pick_inner_iter(seq, ni, CU, BLOCK_PER_CU)
        n_groups = ni // inner_iter
        assert _pick(seq, NUM_HEADS, n_groups) == expected, f"seq={seq}"


def test_large_sequence_takes_the_widest_tile():
    """A prefill chunk fills the grid at any tile, so it must take 64.

    This is the case the patch exists for: at num_heads=64 the old fixed 16
    issued a 4x redundant KV gather.
    """
    assert _pick(14472, NUM_HEADS, 1) == 64


def test_tiny_batch_does_not_starve_the_grid():
    """A single-token decode must not pick a tile that leaves the GPU idle."""
    h = _pick(1, NUM_HEADS, 1)
    assert h == 16  # smallest tile => most blocks
    assert _grid(1, NUM_HEADS, h, 1) == 4


def test_pick_is_monotone_in_sequence_length():
    """More work must never select a smaller tile."""
    picks = [_pick(seq, NUM_HEADS, 8) for seq in (1, 2, 4, 8, 16, 32, 64, 128, 1024)]
    assert picks == sorted(picks), picks


def test_fallback_is_the_stock_tile():
    """When nothing fills the grid the function returns the original 16."""
    # n_groups=1, seq=1 -> grid is 1*head_blocks, far below cu*block_per_cu.
    assert _pick(1, NUM_HEADS, 1) == 16


def test_head_count_below_a_tile_is_admitted_as_one_block():
    """num_heads < h is allowed by the kernel assert (num_heads <= h)."""
    h = _pick(14472, 48, 1)
    assert _kernel_assert_holds(48, h)
    assert (48 + h - 1) // h == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
