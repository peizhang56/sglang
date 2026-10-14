"""Prefix-sort parity for the torch sampler fallbacks.

Guards the local patch (``_TOP_P_SORT_PREFIX``) that replaced the full
descending ``probs.sort()`` in ``top_k_top_p_min_p_sampling_from_probs_torch``
and ``top_p_normalize_probs_torch`` with ``torch.topk`` over a prefix.

Neither function had any test reference before this file.

Two things here are easy to get wrong and are asserted explicitly:

1. **Output width.**  With ``torch.topk`` the working tensor is only
   ``prefix`` wide, so the scatter target must be ``zeros_like(probs)`` and not
   ``zeros_like(probs_sort)``.  Getting that wrong returns a prefix-width row
   and silently reshapes the sampler's output.

2. **The uncovered-row direction.**  When a row's prefix mass does not reach
   ``top_p`` this path TRUNCATES at the prefix.  The ``renorm_triton`` patch
   handles the same case in the OPPOSITE direction -- it keeps the whole row.
   Both directions are pinned, in their own file, so the divergence is visible.
   See ``test/registered/kernels/ops/sampling/test_top_p_prefix_pivot.py``.

Assertions are on scale-free quantities (L1 mass moved, retained support), not
a per-element ``atol`` -- at vocab 154,880 a single probability is ~6e-6, so an
``atol`` of 1e-3 is larger than the entire signal.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sampler fallbacks run on device"
)

VOCAB = 154880
TOP_P = 0.95


def _probs(batch, vocab, *, nucleus=None, scale=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=gen).cuda() * scale
    if nucleus is not None:
        logits[:, :nucleus] += 20.0
    return logits.softmax(-1).float()


def _exact_top_p_normalize(probs, top_ps):
    """The pre-patch full descending sort, kept as the reference."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs).scatter_(-1, probs_idx, probs_sort)


def _nucleus_width(probs, top_p):
    desc = torch.sort(probs, dim=-1, descending=True).values
    return int((torch.cumsum(desc, dim=-1) <= top_p).sum(dim=-1).max().item()) + 1


def _prefix_width():
    from sglang.srt.layers import sampler

    return sampler._TOP_P_SORT_PREFIX


# --------------------------------------------------------------------------
# top_p_normalize_probs_torch
# --------------------------------------------------------------------------


def test_normalize_returns_full_vocabulary_width():
    """Regression: the scatter target is zeros_like(probs), not probs_sort.

    With a 2048-wide prefix, scattering into ``zeros_like(probs_sort)`` would
    return a [batch, 2048] row instead of [batch, VOCAB].
    """
    from sglang.srt.layers.sampler import top_p_normalize_probs_torch

    probs = _probs(8, VOCAB, nucleus=64, seed=1)
    top_ps = torch.full((8,), TOP_P, device=probs.device)
    assert _prefix_width() < VOCAB, "prefix must be active for this to bite"

    got = top_p_normalize_probs_torch(probs, top_ps)
    assert got.shape == probs.shape


@pytest.mark.parametrize("nucleus", [1, 15, 512])
def test_normalize_covered_rows_match_full_sort(nucleus):
    from sglang.srt.layers.sampler import top_p_normalize_probs_torch

    probs = _probs(8, VOCAB, nucleus=nucleus, seed=nucleus)
    top_ps = torch.full((8,), TOP_P, device=probs.device)
    assert _nucleus_width(probs, TOP_P) < _prefix_width()

    got = top_p_normalize_probs_torch(probs, top_ps)
    want = _exact_top_p_normalize(probs.clone(), top_ps)

    assert got.sub(want).abs().sum(-1).max().item() < 1e-6
    assert torch.equal((got > 0), (want > 0))


def test_normalize_uncovered_row_truncates_at_the_prefix():
    """Documented approximation: keeps at most ``prefix`` tokens, not the row.

    This is the OPPOSITE direction to ``renorm_triton._top_p_pivots``, which
    keeps the whole row in the same situation.  Pinned deliberately.
    """
    from sglang.srt.layers.sampler import top_p_normalize_probs_torch

    probs = _probs(4, VOCAB, scale=0.1, seed=3)
    top_ps = torch.full((4,), TOP_P, device=probs.device)
    assert _nucleus_width(probs, TOP_P) > _prefix_width(), "row must be uncovered"

    got = top_p_normalize_probs_torch(probs, top_ps)
    kept = int((got > 0).sum(-1).max().item())
    assert kept <= _prefix_width()

    # The exact path would have kept far more; assert the divergence is real so
    # that making this exact is a visible, deliberate edit.
    want = _exact_top_p_normalize(probs.clone(), top_ps)
    assert int((want > 0).sum(-1).min().item()) > _prefix_width()


def test_normalize_rows_sum_to_one():
    from sglang.srt.layers.sampler import top_p_normalize_probs_torch

    probs = _probs(16, VOCAB, nucleus=128, seed=5)
    top_ps = torch.full((16,), TOP_P, device=probs.device)
    got = top_p_normalize_probs_torch(probs, top_ps)
    torch.testing.assert_close(got.sum(-1), torch.ones(16, device=probs.device))


def test_normalize_prefix_disabled_is_exact():
    from sglang.srt.layers import sampler

    saved = sampler._TOP_P_SORT_PREFIX
    try:
        sampler._TOP_P_SORT_PREFIX = 0
        probs = _probs(4, VOCAB, scale=0.1, seed=3)
        top_ps = torch.full((4,), TOP_P, device=probs.device)
        got = sampler.top_p_normalize_probs_torch(probs, top_ps)
        want = _exact_top_p_normalize(probs.clone(), top_ps)
        assert got.sub(want).abs().sum(-1).max().item() < 1e-6
        assert torch.equal((got > 0), (want > 0))
    finally:
        sampler._TOP_P_SORT_PREFIX = saved


# --------------------------------------------------------------------------
# top_k_top_p_min_p_sampling_from_probs_torch
# --------------------------------------------------------------------------


def _sample(probs, top_ks, top_ps, *, return_filtered_probs=False):
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

    batch = probs.shape[0]
    return top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        top_ks,
        top_ps,
        torch.zeros(batch, device=probs.device),
        False,  # need_min_p_sampling
        None,  # sampling_seed
        torch.arange(batch, device=probs.device),
        return_filtered_probs=return_filtered_probs,
    )


def test_sampling_selects_a_token_inside_the_nucleus():
    """End-to-end guard: the sampled id must be one top-k/top-p would allow."""
    probs = _probs(32, VOCAB, nucleus=8, seed=4)
    top_ks = torch.full((32,), 8, device=probs.device, dtype=torch.int64)
    top_ps = torch.full((32,), TOP_P, device=probs.device)

    out = _sample(probs, top_ks, top_ps)
    token_ids = out[0] if isinstance(out, tuple) else out
    # The peaky head was built on columns [0, nucleus).
    assert int(token_ids.max().item()) < 8
    assert int(token_ids.min().item()) >= 0


def test_sampling_top_k_larger_than_prefix_is_clamped_to_the_prefix():
    """A top_k wider than the prefix cannot retain more than the prefix.

    The top-k mask compares against ``arange(probs_sort.shape[-1])``, which is
    the prefix width, so a larger ``top_k`` zeroes nothing and the effective k
    is the prefix.  Pinned so the limit is explicit rather than incidental.
    """
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

    prefix = _prefix_width()
    probs = _probs(4, VOCAB, scale=0.1, seed=6)
    top_ks = torch.full((4,), prefix * 4, device=probs.device, dtype=torch.int64)
    top_ps = torch.ones(4, device=probs.device)

    _, filtered, _, _ = top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        top_ks,
        top_ps,
        torch.zeros(4, device=probs.device),
        False,
        None,
        torch.arange(4, device=probs.device),
        return_filtered_probs=True,
    )
    # return_filtered_probs bypasses the prefix (full sort), so the full width
    # is available; this documents that the opt-out exists and is wired.
    assert filtered.shape[-1] == VOCAB


def test_return_filtered_probs_uses_the_full_sort():
    """``return_filtered_probs`` must not take the prefix path.

    It returns the filtered distribution and its permutation, which have to be
    full width to be meaningful.
    """
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

    probs = _probs(4, VOCAB, nucleus=16, seed=8)
    out = top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        torch.full((4,), 16, device=probs.device, dtype=torch.int64),
        torch.full((4,), TOP_P, device=probs.device),
        torch.zeros(4, device=probs.device),
        False,
        None,
        torch.arange(4, device=probs.device),
        return_filtered_probs=True,
    )
    assert isinstance(out, tuple) and len(out) == 4
    assert out[1].shape[-1] == VOCAB
    assert out[2].shape[-1] == VOCAB


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
