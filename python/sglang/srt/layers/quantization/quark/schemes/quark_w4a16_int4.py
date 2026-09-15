# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import is_hip

__all__ = ["QuarkW4A16Int4"]

# Per-nibble +8 (mod 16), i.e. signed int4 -> unsigned int4 with a +8 bias.
_INT4_SIGN_FLIP = -2004318072  # 0x88888888 as a signed int32

# Rows up to which the fused int4 GEMM beats dequantize-to-bf16 + hipBLAS GEMM.
# Measured on gfx1151 (5120x17408): ~2.9x at M<=16, 1.13x at M=256, and it
# falls behind from M~2048 (prefill), where the dense GEMM is compute-bound.
_FUSED_GEMM_MAX_ROWS = 256


class QuarkW4A16Int4(QuarkLinearScheme):
    """Weight-only int4 per-group (W4A16), as produced by Quark's AWQ exporter.

    The checkpoint layout is AWQ's: 8 int4 values packed per int32 along the
    output dim, in AWQ's interleaved nibble order, with a bf16 per-group scale.
    The only difference is that Quark stores *signed* int4 codes against an
    all-zero zero-point, where AWQ stores unsigned codes against a zero-point
    of 8. Since dequantization is ``(w - zp) * scale`` in both, adding 8 to
    every nibble of both tensors is lossless and lets us reuse the AWQ kernels.
    """

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        from sglang.kernels.ops.quantization.awq_triton import (
            AWQ_TRITON_SUPPORTED_GROUP_SIZES,
        )

        self.group_size = weight_config.get("group_size")
        self.pack_factor = 8
        self.out_dtype = torch.get_default_dtype()
        # ROCm has no native AWQ GEMM, so the shared AWQ path dequantizes the
        # whole weight to bf16 on every forward -- more memory traffic than the
        # unquantized model. Use the fused triton int4 GEMM for the small-M
        # (decode) shapes instead, where it is ~3x faster.
        self.use_fused_gemm = (
            is_hip() and self.group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # Turing and up (same as AWQ); on ROCm the triton kernel is used.
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"Input size {input_size_per_partition} is not divisible by "
                f"group size {self.group_size}. This can be caused by too "
                "large a tensor parallel size."
            )
        if output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f"Output size {output_size_per_partition} is not divisible by "
                f"the int4 pack factor {self.pack_factor}. This can be caused "
                "by too large a tensor parallel size."
            )

        layer.logical_widths = output_partition_sizes

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_zero_point = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        flip = torch.tensor(
            _INT4_SIGN_FLIP, dtype=torch.int32, device=layer.weight.device
        )
        layer.weight = Parameter(
            torch.bitwise_xor(layer.weight.data, flip), requires_grad=False
        )
        layer.weight_zero_point = Parameter(
            torch.bitwise_xor(layer.weight_zero_point.data, flip), requires_grad=False
        )
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
            awq_dequantize,
        )

        out_shape = x.shape[:-1] + (layer.weight.shape[-1] * self.pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if reshaped_x.shape[0] <= _FUSED_GEMM_MAX_ROWS and self.use_fused_gemm:
            from sglang.kernels.ops.quantization.awq_triton import awq_gemm_triton

            out = awq_gemm_triton(
                reshaped_x,
                layer.weight,
                layer.weight_scale,
                layer.weight_zero_point,
                split_k_iters=1,
            )
        else:
            weight = awq_dequantize(
                layer.weight, layer.weight_scale, layer.weight_zero_point
            )
            out = torch.matmul(reshaped_x, weight)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
