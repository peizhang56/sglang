"""Top-p prefix-pivot parity for the ROCm ``renorm_triton`` path.

Assertions here are on scale-free quantities -- L1 mass moved and the retained
support -- never a per-element ``atol`` against a vocabulary-sized distribution:
``test_renorm.py``'s ``atol=1e-3`` is ~200x the entire signal of a pivot
regression that redistributes 20% of the mass, so it cannot see one.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="renorm_triton top-p prefix is the ROCm path"
)

# Production geometry: GLM-5.3-Flash ships vocab 154,880 and top_p 0.95.
VOCAB = 154880
TOP_P = 0.95


def _exact_renorm(probs_fp32: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """The pre-patch full-vocabulary ascending sort, kept as the reference."""
    vocab_size = probs_fp32.shape[-1]
    sorted_probs = torch.sort(probs_fp32, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=vocab_size - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1))
    out = torch.where(probs_fp32 >= pivots, probs_fp32, torch.zeros_like(probs_fp32))
    return out / out.sum(dim=-1, keepdim=True)


def _probs(batch, vocab, *, nucleus=None, scale=1.0, seed=0, device="cuda:0"):
    """Deterministic distribution; ``nucleus`` forces a peaky head of that width."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=gen).to(device) * scale
    if nucleus is not None:
        logits[:, :nucleus] += 20.0
    return logits.softmax(-1).float()


def _nucleus_width(probs_fp32, top_p):
    desc = torch.sort(probs_fp32, dim=-1, descending=True).values
    return int((torch.cumsum(desc, dim=-1) <= top_p).sum(dim=-1).max().item()) + 1


def _prefix_width():
    from sglang.kernels.ops.sampling import renorm_triton

    return renorm_triton._TOP_P_PIVOT_PREFIX


# ===== Covered rows: the prefix holds the nucleus, so this must be exact =====


@pytest.mark.parametrize("batch_size", [1, 48])
@pytest.mark.parametrize("nucleus", [1, 15, 512])
def test_covered_rows_are_bit_exact(batch_size, nucleus):
    """A nucleus inside the prefix must reproduce the full sort exactly."""
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    probs = _probs(batch_size, VOCAB, nucleus=nucleus, seed=nucleus)
    top_ps = torch.full((batch_size,), TOP_P, device=probs.device)
    assert _nucleus_width(probs, TOP_P) < _prefix_width(), "row must be covered"

    got = top_p_renorm_probs_triton(probs, top_ps)
    want = _exact_renorm(probs, top_ps)

    # Scale-free: total probability mass moved, not a per-element tolerance.
    assert got.sub(want).abs().sum(-1).max().item() < 1e-6
    # And the same tokens survived.
    assert torch.equal((got > 0), (want > 0))


def test_covered_pivot_equals_full_sort_pivot():
    """The pivot VALUE itself, which is all the sort ever produced."""
    from sglang.kernels.ops.sampling.renorm_triton import _top_p_pivots

    probs = _probs(8, VOCAB, nucleus=64, seed=7)
    top_ps = torch.full((8,), TOP_P, device=probs.device)

    sorted_probs = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=VOCAB - 1)
    want = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)

    assert torch.equal(_top_p_pivots(probs, top_ps), want)


# ===== Uncovered rows: approximation, pinned so it cannot drift silently =====


def test_uncovered_row_keeps_the_whole_row():
    """Nucleus wider than the prefix -> pivot 0.0, i.e. top-p does not truncate.

    ``sampler.py``'s ``_TOP_P_SORT_PREFIX`` takes this case in the OPPOSITE
    direction (it truncates); see ``test/registered/sampling/test_top_p_sort_prefix.py``.
    """
    from sglang.kernels.ops.sampling.renorm_triton import (
        _top_p_pivots,
        top_p_renorm_probs_triton,
    )

    # Near-uniform: nucleus is ~90% of a 154,880 vocabulary, far past the prefix.
    probs = _probs(4, VOCAB, scale=0.1, seed=3)
    top_ps = torch.full((4,), TOP_P, device=probs.device)
    assert _nucleus_width(probs, TOP_P) > _prefix_width(), "row must be uncovered"

    assert torch.equal(
        _top_p_pivots(probs, top_ps), torch.zeros(4, device=probs.device)
    )

    got = top_p_renorm_probs_triton(probs, top_ps)
    assert int((got > 0).sum(-1).min().item()) == VOCAB  # nothing truncated

    # Assert the divergence is large, so making the fallback exact fails loudly here.
    want = _exact_renorm(probs, top_ps)
    assert got.sub(want).abs().sum(-1).max().item() > 1e-2
    assert int((want > 0).sum(-1).max().item()) < VOCAB


# ===== Escape hatch and edge cases =====


def test_prefix_disabled_restores_the_exact_full_sort():
    """SGLANG_TOP_P_PIVOT_PREFIX=0 must be exact even on an uncovered row."""
    import importlib

    from sglang.kernels.ops.sampling import renorm_triton

    saved = renorm_triton._TOP_P_PIVOT_PREFIX
    try:
        renorm_triton._TOP_P_PIVOT_PREFIX = 0
        probs = _probs(4, VOCAB, scale=0.1, seed=3)
        top_ps = torch.full((4,), TOP_P, device=probs.device)
        got = renorm_triton.top_p_renorm_probs_triton(probs, top_ps)
        want = _exact_renorm(probs, top_ps)
        assert got.sub(want).abs().sum(-1).max().item() < 1e-6
        assert torch.equal((got > 0), (want > 0))
    finally:
        renorm_triton._TOP_P_PIVOT_PREFIX = saved
        importlib.reload(renorm_triton)


def test_top_p_one_keeps_every_token():
    """top_p == 1.0 retains the whole row under both paths."""
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    probs = _probs(4, VOCAB, nucleus=32, seed=11)
    got = top_p_renorm_probs_triton(probs, torch.ones(4, device=probs.device))
    assert int((got > 0).sum(-1).min().item()) == VOCAB
    torch.testing.assert_close(got.sum(-1), torch.ones(4, device=probs.device))


def test_rows_are_normalized():
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    probs = _probs(16, VOCAB, nucleus=128, seed=5)
    top_ps = torch.full((16,), TOP_P, device=probs.device)
    got = top_p_renorm_probs_triton(probs, top_ps)
    torch.testing.assert_close(got.sum(-1), torch.ones(16, device=probs.device))


def test_vocab_smaller_than_prefix_uses_the_exact_path():
    """Below the prefix width the patch must fall through to the full sort."""
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    small = min(1024, _prefix_width() // 2)
    probs = _probs(8, small, seed=2)
    top_ps = torch.full((8,), TOP_P, device=probs.device)
    got = top_p_renorm_probs_triton(probs, top_ps)
    want = _exact_renorm(probs, top_ps)
    assert got.sub(want).abs().sum(-1).max().item() < 1e-6
    assert torch.equal((got > 0), (want > 0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
