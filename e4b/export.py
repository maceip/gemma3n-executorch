#!/usr/bin/env python3
"""
Export Gemma3n-E4B-IT (text-only) to ExecuTorch .pte format.

Model: google/gemma-3n-E4B-it
Text params: 7.40B (35 layers)
Output: ~13.1 GB .pte file

IMPORTANT: Requires patches to transformers. See ../README.md
"""
import os
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)

MODEL_ID = "google/gemma-3n-E4B-it"
OUTPUT_PATH = "Gemma3n-E4B-IT-text-only.pte"
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

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
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

    # Test forward pass
    dummy = torch.ones(1, SEQ_LEN, dtype=torch.long)
    print(f"Testing forward pass (seq_len={SEQ_LEN})...")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward OK: {out.shape}")

    # Export
    print("\nExporting to ExecuTorch...")
    from torch.export import export
    from executorch.exir import to_edge

    ep = export(wrapper, (dummy,), strict=False)
    print("  torch.export OK")

    et = to_edge(ep)
    print("  to_edge OK")

    et_prog = et.to_executorch()
    print("  to_executorch OK")

    # Save
    with open(OUTPUT_PATH, 'wb') as f:
        et_prog.write_to_file(f)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024**2)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
