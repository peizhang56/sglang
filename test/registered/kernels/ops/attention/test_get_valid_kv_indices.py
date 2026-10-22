"""Compaction parity for ``get_valid_kv_indices``.

The kernel pads to ``next_power_of_2(topk)`` because ``tl.arange`` needs a
power-of-2 extent while the page-table width is one only at ``index_kpool == 1``.
Non-power-of-2 widths are the regression; the power-of-2 ones prove the padding
left the original path alone.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="get_valid_kv_indices is the ROCm DSA path"
)


def _run(page_table_1):
    """Call the kernel under the same contract as dsa_backend.py."""
    from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

    bs, topk = page_table_1.shape
    device = page_table_1.device

    counts = (page_table_1 != -1).sum(dim=1)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(counts, dim=0)
    kv_indices = torch.zeros(bs * topk, dtype=torch.int32, device=device)

    get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)
    return kv_indptr, kv_indices, counts


def _expected(page_table_1):
    """Row-major compaction of the non -1 entries, order preserved."""
    rows = []
    for row in page_table_1.tolist():
        rows.extend([v for v in row if v != -1])
    return rows


def _check(page_table_1):
    kv_indptr, kv_indices, counts = _run(page_table_1)
    total = int(counts.sum().item())
    assert kv_indices[:total].tolist() == _expected(page_table_1)
    # Per-row placement, not just the concatenation.
    for b in range(page_table_1.shape[0]):
        lo, hi = int(kv_indptr[b]), int(kv_indptr[b + 1])
        want = [v for v in page_table_1[b].tolist() if v != -1]
        assert kv_indices[lo:hi].tolist() == want
    return kv_indices, total


def _table(bs, topk, *, valid_per_row, seed=0, device="cuda:0"):
    """Page table whose first ``valid_per_row`` entries are valid, rest -1."""
    gen = torch.Generator().manual_seed(seed)
    vals = torch.randint(0, 100000, (bs, topk), generator=gen, dtype=torch.int32)
    table = torch.full((bs, topk), -1, dtype=torch.int32)
    for b in range(bs):
        n = valid_per_row[b]
        table[b, :n] = vals[b, :n]
    return table.to(device)


# ===== The regression: topk is not a power of 2 =====


@pytest.mark.parametrize("topk", [96, 100, 320, 3072, 2176])
def test_non_power_of_two_topk(topk):
    """kpool > 1 widths. Pre-fix these raised on tl.arange(0, topk)."""
    assert topk & (topk - 1), "this case must NOT be a power of two"
    bs = 8
    valid = [0, 1, topk // 3, topk // 2, topk - 1, topk, 5, topk // 4]
    _check(_table(bs, topk, valid_per_row=valid, seed=topk))


@pytest.mark.parametrize("topk", [64, 256, 2048])
def test_power_of_two_topk_still_works(topk):
    """The original path; padding must be a no-op when topk is already 2^n."""
    bs = 8
    valid = [0, 1, topk // 3, topk // 2, topk - 1, topk, 5, topk // 4]
    _check(_table(bs, topk, valid_per_row=valid, seed=topk))


# ===== Edge cases around the padding sentinel =====


def test_all_entries_valid():
    """No -1 anywhere: padding lanes are the only masked ones."""
    topk = 320
    kv_indices, total = _check(_table(4, topk, valid_per_row=[topk] * 4, seed=1))
    assert total == 4 * topk


def test_all_entries_invalid():
    """Every entry -1: nothing may be written."""
    topk = 320
    table = torch.full((4, topk), -1, dtype=torch.int32, device="cuda:0")
    kv_indptr, kv_indices, counts = _run(table)
    assert int(counts.sum().item()) == 0
    assert int(kv_indptr[-1].item()) == 0
    assert kv_indices.eq(0).all(), "buffer must be untouched"


def test_padding_overhang_is_never_written():
    """A width just past a power of 2 has the largest padding overhang: 2049
    pads to 4096, so 2047 sentinel lanes must not reach the buffer."""
    topk = 2049
    bs = 2
    valid = [7, topk]
    table = _table(bs, topk, valid_per_row=valid, seed=9)
    kv_indices, total = _check(table)
    # Everything past the compacted region must still be zero.
    assert kv_indices[total:].eq(0).all()


def test_interleaved_invalid_entries():
    """-1 scattered through the row, not just as a trailing block."""
    topk = 192
    gen = torch.Generator().manual_seed(21)
    table = torch.randint(0, 100000, (6, topk), generator=gen, dtype=torch.int32)
    holes = torch.rand(6, topk, generator=gen) < 0.5
    table[holes] = -1
    _check(table.to("cuda:0"))


def test_single_row():
    _check(_table(1, 3072, valid_per_row=[1500], seed=4))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
