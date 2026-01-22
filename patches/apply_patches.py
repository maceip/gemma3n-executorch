#!/usr/bin/env python3
"""
Apply required patches to transformers for Gemma3n ExecuTorch export.

This script modifies the installed transformers package to fix two issues:
1. Missing self.training guard in AltUp (causes "Constant is mutated" error)
2. Dynamic erfinv call in _gaussian_topk (not supported by ExecuTorch)

Usage: python apply_patches.py
"""
import site
import os
import sys


def find_modeling_file():
    """Find the modeling_gemma3n.py file in site-packages."""
    for path in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = os.path.join(path, 'transformers', 'models', 'gemma3n', 'modeling_gemma3n.py')
        if os.path.exists(candidate):
            return candidate
    return None


def apply_patch_1(content: str) -> str:
    """Add self.training guard to correction_coefs.weight.data.clamp_()"""
    old = "        if self.config.altup_coef_clip is not None:\n            self.correction_coefs.weight.data.clamp_"
    new = "        if self.training and self.config.altup_coef_clip is not None:\n            self.correction_coefs.weight.data.clamp_"

    if old in content:
        content = content.replace(old, new)
        print("  Patch 1 applied: Added self.training guard")
    elif new in content:
        print("  Patch 1: Already applied")
    else:
        print("  WARNING: Patch 1 pattern not found - may need manual patching")

    return content


def apply_patch_2(content: str) -> str:
    """Pre-compute std_multiplier in __init__ instead of forward."""

    # Check if already patched
    if '_std_multiplier' in content:
        print("  Patch 2: Already applied")
        return content

    # Add buffer registration in __init__
    old_init = "        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]\n\n    def forward"
    new_init = """        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        # Pre-compute std_multiplier at init time to avoid erfinv during forward pass
        # This enables export to ExecuTorch which doesn't support erfinv
        if self.activation_sparsity > 0.0:
            normal_dist = torch.distributions.normal.Normal(0, 1)
            std_multiplier = normal_dist.icdf(torch.tensor(self.activation_sparsity, dtype=torch.float32))
            self.register_buffer("_std_multiplier", std_multiplier, persistent=False)

    def forward"""

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("  Patch 2a applied: Added _std_multiplier buffer in __init__")
    else:
        print("  WARNING: Patch 2a pattern not found")

    # Replace _gaussian_topk method
    old_gaussian = '''    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)'''

    new_gaussian = '''    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        # Use pre-computed std_multiplier to avoid erfinv during tracing/export
        std_multiplier = self._std_multiplier.to(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)'''

    if old_gaussian in content:
        content = content.replace(old_gaussian, new_gaussian)
        print("  Patch 2b applied: Replaced _gaussian_topk method")
    else:
        print("  WARNING: Patch 2b pattern not found")

    return content


def main():
    print("Gemma3n ExecuTorch Patch Applier")
    print("=" * 40)

    modeling_file = find_modeling_file()
    if not modeling_file:
        print("ERROR: Could not find transformers/models/gemma3n/modeling_gemma3n.py")
        print("Make sure transformers is installed: pip install transformers")
        sys.exit(1)

    print(f"Found: {modeling_file}")

    # Read current content
    with open(modeling_file, 'r') as f:
        content = f.read()

    # Create backup
    backup_file = modeling_file + '.backup'
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"Backup created: {backup_file}")

    # Apply patches
    print("\nApplying patches...")
    content = apply_patch_1(content)
    content = apply_patch_2(content)

    # Write patched content
    with open(modeling_file, 'w') as f:
        f.write(content)

    print("\nDone! Patches applied successfully.")
    print("You can now export Gemma3n models to ExecuTorch.")


if __name__ == "__main__":
    main()
