#!/usr/bin/env python3
"""
Validate a Gemma3n ExecuTorch .pte file.

Usage: python validate.py /path/to/model.pte
"""
import sys
import torch
from executorch.runtime import Runtime


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <path_to_pte>")
        sys.exit(1)

    pte_path = sys.argv[1]
    print(f"Loading {pte_path}...")

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)

    print(f"Methods: {program.method_names}")

    method = program.load_method('forward')
    print("Method 'forward' loaded successfully")

    # Test inference
    dummy = torch.ones(1, 32, dtype=torch.long)
    print(f"\nRunning inference with input shape {dummy.shape}...")

    output = method.execute([dummy])

    print(f"Output type: {type(output)}")
    if output:
        print(f"Output[0] shape: {output[0].shape}")
        print(f"Output[0] dtype: {output[0].dtype}")
        print(f"Sample logits: {output[0][0, 0, :5]}")

        # Basic sanity checks
        logits = output[0]
        assert not torch.isnan(logits).any(), "ERROR: Output contains NaN!"
        assert not torch.isinf(logits).any(), "ERROR: Output contains Inf!"
        print("\nValidation PASSED - no NaN/Inf in output")
    else:
        print("ERROR: No output returned")
        sys.exit(1)


if __name__ == "__main__":
    main()
