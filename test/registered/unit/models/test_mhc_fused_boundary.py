"""Dispatch boundary for the mHC fused post+pre path.

``_MHC_FUSED_BOUNDARY_MAX_TOKENS`` was chosen to put the two production shapes
on opposite sides -- a 48-token decode batch fuses, a 16,384-token prefill chunk
does not -- so both are pinned. The kernel is mocked: this tests the policy only.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 4096
HC_MULT = 4

# Decode is bs 12 x 4 draft tokens; prefill is --chunked-prefill-size 16384.
DECODE_TOKENS = 48
PREFILL_TOKENS = 16384


class TestMhcFusedBoundary(CustomTestCase):
    def _layer(self):
        """A bare object carrying only what hc_ffn_post_pre reads."""
        import sglang.srt.models.glm5_next as glm5_next

        layer = object.__new__(glm5_next.Glm5NextDecoderLayer)
        layer.config = SimpleNamespace(
            mhc=True,
            hc_mult=HC_MULT,
            rms_norm_eps=1e-5,
            hc_eps=1e-5,
            hc_sinkhorn_iters=0,
        )
        layer.hc_ffn_fn = None
        layer.hc_ffn_scale = None
        layer.hc_ffn_base = None
        return layer

    def _call(self, layer, num_tokens):
        hidden_states = torch.zeros(num_tokens, HIDDEN)
        residual = torch.zeros(num_tokens, HC_MULT * HIDDEN)
        h_res = torch.zeros(num_tokens, HC_MULT * HC_MULT)
        h_post = torch.zeros(num_tokens, HC_MULT)
        return layer.hc_ffn_post_pre(
            hidden_states,
            residual,
            h_res,
            h_post,
            out_norm_weight=torch.zeros(HIDDEN),
            out_norm_eps=1e-5,
        )

    @staticmethod
    def _fused_result(num_tokens):
        """The 5-tuple the real kernel returns, which the caller unpacks."""
        return (
            torch.zeros(num_tokens, HC_MULT, HIDDEN),  # next_residual
            torch.zeros(num_tokens, HIDDEN),  # layer_input
            torch.zeros(num_tokens, HC_MULT),  # post
            torch.zeros(num_tokens, HC_MULT, HC_MULT),  # comb
            torch.zeros(num_tokens, HIDDEN),  # norm_fused
        )

    def _dispatch(self, num_tokens, kernel_returns_none=False):
        """Returns True if the fused kernel was invoked."""
        import sglang.srt.models.glm5_next as glm5_next

        result = None if kernel_returns_none else self._fused_result(num_tokens)
        with mock.patch.object(
            glm5_next, "apply_mhc_post_pre_boundary", return_value=result
        ) as fused:
            out = self._call(self._layer(), num_tokens)
        if fused.called:
            if kernel_returns_none:
                self.assertIsNone(out)
            else:
                self.assertIsNotNone(out)
                self.assertEqual(len(out), 5)
            return True
        self.assertIsNone(out, "unfused path must return None")
        return False

    def test_kernel_declining_falls_back_even_below_the_cap(self):
        """The kernel may still return None; the caller must propagate it."""
        self.assertTrue(
            self._dispatch(DECODE_TOKENS, kernel_returns_none=True)
        )

    def test_boundary_is_inclusive_at_the_cap(self):
        """`num_tokens > cap` is the guard, so the cap itself still fuses."""
        import sglang.srt.models.glm5_next as glm5_next

        cap = glm5_next._MHC_FUSED_BOUNDARY_MAX_TOKENS
        self.assertTrue(self._dispatch(cap))
        self.assertFalse(self._dispatch(cap + 1))

    def test_decode_batch_fuses(self):
        """48 tokens: measured +24.0% for the fused kernel."""
        self.assertTrue(self._dispatch(DECODE_TOKENS))

    def test_prefill_chunk_stays_unfused(self):
        """16,384 tokens: measured -10.4%, i.e. the fused kernel loses."""
        self.assertFalse(self._dispatch(PREFILL_TOKENS))

    def test_cap_separates_the_two_production_shapes(self):
        """The whole point of the value: decode on one side, prefill the other."""
        import sglang.srt.models.glm5_next as glm5_next

        cap = glm5_next._MHC_FUSED_BOUNDARY_MAX_TOKENS
        self.assertLessEqual(DECODE_TOKENS, cap)
        self.assertGreater(PREFILL_TOKENS, cap)

    def test_cap_stays_below_the_measured_crossover(self):
        """The measured crossover is ~2048; above it the dispatch would start
        choosing the fused kernel in the region where it loses."""
        import sglang.srt.models.glm5_next as glm5_next

        self.assertLess(glm5_next._MHC_FUSED_BOUNDARY_MAX_TOKENS, 2048)

    def test_small_batches_still_fuse(self):
        """The old cap was 16; everything at or below it must be unaffected."""
        for num_tokens in (1, 12, 16, 24):
            with self.subTest(num_tokens=num_tokens):
                self.assertTrue(self._dispatch(num_tokens))

    def test_requires_mhc_config(self):
        layer = self._layer()
        layer.config.mhc = False
        with self.assertRaises(AssertionError):
            self._call(layer, DECODE_TOKENS)


if __name__ == "__main__":
    unittest.main()
