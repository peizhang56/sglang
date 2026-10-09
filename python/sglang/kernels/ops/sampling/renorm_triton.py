"""ROCm-compatible top-k / top-p probability renormalization fallbacks."""

from __future__ import annotations

import os
from typing import Union

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024

# Width of the descending prefix used to find the top-p pivot; 0 = exact full sort.
# See _top_p_pivots.
_TOP_P_PIVOT_PREFIX = int(os.environ.get("SGLANG_TOP_P_PIVOT_PREFIX", "2048"))


@triton.jit
def _mask_and_partial_sum_kernel(
    probs_ptr,
    pivots_ptr,
    out_ptr,
    partial_sums_ptr,
    vocab_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    row_offsets = row.to(tl.int64) * vocab_size + offsets

    probs = tl.load(probs_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    pivot = tl.load(pivots_ptr + row)
    kept = tl.where(mask & (probs >= pivot), probs, 0.0)

    tl.store(out_ptr + row_offsets, kept, mask=mask)
    tl.store(partial_sums_ptr + row * num_chunks + chunk, tl.sum(kept, axis=0))


@triton.jit
def _normalize_kernel(
    out_ptr,
    row_sums_ptr,
    numel,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    row = offsets // vocab_size
    values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    denominator = tl.load(row_sums_ptr + row, mask=mask, other=1.0)
    tl.store(out_ptr + offsets, values / denominator, mask=mask)


def _prepare_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}")
    if not probs.is_cuda:
        raise ValueError("renorm kernels require a CUDA/HIP tensor")
    return probs.float().contiguous()


def _renorm_from_pivots(probs_fp32: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = probs_fp32.shape
    num_chunks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    out = torch.empty_like(probs_fp32)
    partial_sums = torch.empty(
        (batch_size, num_chunks), device=probs_fp32.device, dtype=torch.float32
    )
    _mask_and_partial_sum_kernel[(batch_size, num_chunks)](
        probs_fp32,
        pivots,
        out,
        partial_sums,
        vocab_size=vocab_size,
        num_chunks=num_chunks,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )

    row_sums = partial_sums.sum(dim=1)
    _normalize_kernel[(triton.cdiv(out.numel(), _BLOCK_SIZE),)](
        out,
        row_sums,
        out.numel(),
        vocab_size=vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out


def _top_p_pivots(probs_fp32: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Per-row top-p threshold: the smallest prob whose retained mass covers top_p.

    LOCAL PATCH (GLM-5.3-Flash, 8x MI355X). This is THE hot torch op in the decode
    step. The c96 ptrace counts aten::sort + aten::cumsum at [48, 154880] 263 times
    in a 289-step window -- a full-vocabulary sort on essentially every decode step,
    measured standalone at 915 us against a 33.9 ms step (2.7% of decode). It is
    reached from the EAGLE/dflash verify path (dflash_utils.py, eagle_utils.py),
    which imports this module directly rather than through the kernel registry.

    All the sort+cdf produce is ONE scalar per row, and that scalar always lives in
    the head of the distribution, so a descending top-k prefix finds it exactly.

    Equivalence, for vocabulary V, ascending index i, descending index j = V-1-i:
        cdf_desc[j] == 1 - cdf_asc[i-1]
        the ascending form picks the smallest i with cdf_asc[i] >= 1-p, i.e. the
        largest i with cdf_asc[i-1] < 1-p  <=>  cdf_desc[j] > p
        so j = smallest index with cdf_desc[j] > p = searchsorted(..., right=True).
    Tie handling is unchanged: both forms return a pivot VALUE and the mask that
    consumes it is `probs >= pivot`, so ties at the pivot are retained either way.

    A row is exact whenever its prefix mass exceeds top_p. Rows where it does not
    (a nucleus wider than the prefix, and every row when top_p == 1.0) fall back to
    pivot 0.0, which retains the whole row and renormalizes -- EXACT for top_p == 1
    and, for any other uncovered row, erring toward keeping more of the tail rather
    than truncating it. The fallback is a `torch.where`, so no host synchronisation
    is introduced; that matters because sglang has commits specifically removing
    syncs from this path.

    SGLANG_TOP_P_PIVOT_PREFIX=0 restores the exact full sort.
    """
    vocab_size = probs_fp32.shape[-1]
    k = _TOP_P_PIVOT_PREFIX
    if not 0 < k < vocab_size:
        sorted_probs = torch.sort(probs_fp32, dim=-1).values
        cdf = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.searchsorted(
            cdf, (1.0 - top_ps).unsqueeze(1), right=False
        ).squeeze(1)
        cutoff.clamp_(max=vocab_size - 1)
        return sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1).contiguous()

    top_vals, _ = torch.topk(probs_fp32, k, dim=-1, sorted=True)
    cdf_desc = torch.cumsum(top_vals, dim=-1)
    idx = torch.searchsorted(cdf_desc, top_ps.unsqueeze(1), right=True).squeeze(1)
    idx.clamp_(max=k - 1)
    pivots = top_vals.gather(1, idx.unsqueeze(1)).squeeze(1)
    # Prefix mass did not cover top_p -> keep the whole row instead of truncating.
    return torch.where(
        cdf_desc[:, -1] > top_ps, pivots, torch.zeros_like(pivots)
    ).contiguous()


def top_p_renorm_probs_triton(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Apply exact top-p thresholding and renormalize each probability row.

    Sorting and prefix sums use PyTorch's device kernels because a vocabulary-sized
    in-register Triton sort does not scale to 100K+ vocabularies. Triton performs
    the bandwidth-heavy masking, partial reduction, and normalization.
    """
    probs_fp32 = _prepare_probs(probs)
    batch_size, vocab_size = probs_fp32.shape
    if batch_size == 0 or vocab_size == 0:
        return probs_fp32

    if isinstance(top_p, torch.Tensor):
        top_ps = top_p.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if top_ps.numel() == 1:
            top_ps = top_ps.expand(batch_size)
        elif top_ps.numel() != batch_size:
            raise ValueError(
                f"top_p must be scalar or have one value per row, got "
                f"{top_ps.numel()} values for {batch_size} rows"
            )
    else:
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p values must be in (0, 1]")
        top_ps = torch.full(
            (batch_size,), float(top_p), device=probs.device, dtype=torch.float32
        )

    # Match FlashInfer's threshold semantics: sort ascending, discard the prefix
    # whose cumulative mass is below 1 - p, and retain all ties at the pivot.
    pivots = _top_p_pivots(probs_fp32, top_ps)

    return _renorm_from_pivots(probs_fp32, pivots)


def top_k_renorm_probs_triton(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Apply exact top-k thresholding and renormalize each probability row.

    Sorting uses PyTorch's device kernels because a vocabulary-sized in-register
    Triton sort does not scale to 100K+ vocabularies. Triton performs the
    bandwidth-heavy masking, partial reduction, and normalization.
    """
    probs_fp32 = _prepare_probs(probs)
    batch_size, vocab_size = probs_fp32.shape
    if batch_size == 0 or vocab_size == 0:
        return probs_fp32

    if isinstance(top_k, torch.Tensor):
        top_ks = top_k.to(device=probs.device, dtype=torch.int64).reshape(-1)
        if top_ks.numel() == 1:
            top_ks = top_ks.expand(batch_size)
        elif top_ks.numel() != batch_size:
            raise ValueError(
                f"top_k must be scalar or have one value per row, got "
                f"{top_ks.numel()} values for {batch_size} rows"
            )
    else:
        top_ks = torch.full(
            (batch_size,), int(top_k), device=probs.device, dtype=torch.int64
        )

    # Match FlashInfer's threshold semantics: sort descending, keep the k highest
    # probabilities, and retain all ties at the pivot.
    sorted_probs = torch.sort(probs_fp32, dim=-1, descending=True).values
    cutoff = (top_ks - 1).clamp_(min=0, max=vocab_size - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1).contiguous()

    return _renorm_from_pivots(probs_fp32, pivots)


__all__ = ["top_k_renorm_probs_triton", "top_p_renorm_probs_triton"]
