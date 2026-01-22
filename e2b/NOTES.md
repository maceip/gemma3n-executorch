# gemma3n-e2b-it export notes

## model details

- source: google/gemma-3n-E2B-it
- type: instruction-tuned multimodal model
- full model: 10.6b params (text + vision + audio)
- text-only: 4.46b params (30 transformer layers)

## export configuration

- sequence length: 32 tokens (fixed at export time)
- dtype: float16
- output size: 8.4 gb

## validation results

```
methods: forward
input shape: torch.Size([1, 32])
output shape: torch.Size([1, 32, 262400])
output dtype: torch.float16
sample logits: tensor([-38.3438, -3.1719, -17.7812, -38.2188, -21.0000])
```

## required patches

before running export apply these patches to transformers/models/gemma3n/modeling_gemma3n.py

### patch 1 - line 1119 - add self.training guard

```diff
- if self.config.altup_coef_clip is not None:
+ if self.training and self.config.altup_coef_clip is not None:
```

### patch 2 - gemma3ntextmlp class - pre-compute std_multiplier

in __init__ after self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

```python
if self.activation_sparsity > 0.0:
    normal_dist = torch.distributions.normal.Normal(0, 1)
    std_multiplier = normal_dist.icdf(torch.tensor(self.activation_sparsity, dtype=torch.float32))
    self.register_buffer("_std_multiplier", std_multiplier, persistent=False)
```

replace _gaussian_topk method

```python
def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
    std_multiplier = self._std_multiplier.to(inputs.dtype)
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    cutoff_x = inputs_mean + inputs_std * std_multiplier
    return nn.functional.relu(inputs - cutoff_x)
```

## reproduction steps

1. install dependencies
   ```bash
   pip install torch transformers executorch
   ```

2. apply patches to transformers

3. run export
   ```bash
   python export.py
   ```

4. validate
   ```bash
   python ../validate.py gemma-3n-E2B-it-text-only.pte
   ```
