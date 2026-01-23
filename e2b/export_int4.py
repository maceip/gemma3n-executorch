#!/usr/bin/env python3
"""
[NOT WORKING] Export Gemma3n-E2B-IT to ExecuTorch .pte format with int4 quantization.

STATUS: This script does NOT work. See known issues below.

Model: google/gemma-3n-E2B-it
Text params: 4.46B (30 layers)
Target output: ~2.5 GB .pte file (int4 via XNNPACK)

Uses PT2E quantization with XNNPACKQuantizer (qc4 = 4-bit per-channel weights).

KNOWN ISSUES:
1. After prepare_pt2e inserts observers, the model forward pass fails with:
   "IndexError: tensors used as indices must be long, int, byte or bool tensors"

   Root cause: The Gemma3n model's unique architecture (AltUp mechanism, sparse MLP
   with gaussian_topk) has operations where integer tensors are used for indexing.
   The PT2E observer insertion process somehow converts these to float32, breaking
   subsequent index operations.

2. Alternative approaches tried and failed:
   - torchao int4_weight_only: Requires GPU kernels ("X must be BF16 and contiguous on GPU")
   - torchao Int8DynamicActivationInt4WeightConfig: Works on CPU for forward pass but
     to_executorch fails with "Missing out variants: torchao::dequantize_affine"
   - quanto library: Tensor subclass issues with to_edge
   - Int8DynActInt4WeightQuantizer: Missing out variants for quantized ops

This file is kept as reference for future work when upstream fixes become available.
"""
import os
import gc
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)

from executorch.extension.llm.export.builder import prepare_pt2e, convert_pt2e, to_edge_transform_and_lower
from executorch.extension.llm.export.quantizer_lib import (
    PT2EQuantOptions,
    DynamicQuantLinearOptions,
    get_pt2e_quantizers
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export

MODEL_ID = "google/gemma-3n-E2B-it"
OUTPUT_PATH = "Gemma3n-E2B-text-only-int4.pte"
SEQ_LEN = 32


class TextOnlyWrapper(torch.nn.Module):
    """Wrapper that extracts text-only forward pass from Gemma3n."""

    def __init__(self, language_model, lm_head):
        super().__init__()
        self.lm = language_model
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.lm(input_ids, use_cache=False)
        return self.lm_head(outputs.last_hidden_state)


def main():
    print(f"Loading {MODEL_ID}...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # Extract text-only components
    lm = model.model.language_model
    lm_head = model.lm_head

    num_layers = len(list(lm.layers))
    params = sum(p.numel() for p in lm.parameters()) + sum(p.numel() for p in lm_head.parameters())
    print(f"Text model: {num_layers} layers, {params/1e9:.2f}B params")

    # Free memory from vision/audio towers
    del model.model.vision_tower
    del model.model.audio_tower
    gc.collect()

    # Create wrapper
    wrapper = TextOnlyWrapper(lm, lm_head)
    wrapper.eval()

    # Get int4 quantizers (is_qc4=True for 4-bit per-channel)
    print("\nSetting up PT2E int4 quantization...")
    quant_options = PT2EQuantOptions(
        quantize_linear=DynamicQuantLinearOptions(is_per_channel=True, is_qc4=True)
    )
    quantizers = get_pt2e_quantizers(quant_options)
    print(f"Using quantizer: {type(quantizers[0]).__name__}")

    # Export to get GraphModule
    dummy = torch.ones(1, SEQ_LEN, dtype=torch.long)
    print(f"\nExporting model (seq_len={SEQ_LEN})...")
    ep = export(wrapper, (dummy,), strict=False)
    gm = ep.module()

    # Prepare for quantization
    print("Preparing for PT2E quantization...")
    prepared = prepare_pt2e(gm, quantizers[0])

    # Skip calibration for dynamic quantization (PlaceholderObserver doesn't need it)
    # Note: Dynamic quantization quantizes activations at runtime, not using calibration data
    print("Skipping calibration (using dynamic quantization)...")

    # Convert to quantized
    print("Converting to quantized model...")
    quantized = convert_pt2e(prepared)

    # Re-export quantized model directly without testing
    # Note: The graph module has dtype issues that get resolved when lowered to XNNPACK
    print("\nRe-exporting quantized model...")
    try:
        ep_quantized = export(quantized, (dummy,), strict=False)
    except Exception as e:
        print(f"Re-export failed: {e}")
        print("Trying alternative: export from original then apply quantization in lowering...")
        raise

    # Lower to XNNPACK
    print("Lowering to XNNPACK backend...")
    et = to_edge_transform_and_lower(
        ep_quantized,
        partitioner=[XnnpackPartitioner()],
    )
    print("to_edge_transform_and_lower OK")

    # Convert to executorch
    print("Converting to executorch...")
    et_prog = et.to_executorch()
    print("to_executorch OK")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        et_prog.write_to_file(f)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024**2)
    size_gb = size_mb / 1024
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {size_gb:.2f} GB ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
