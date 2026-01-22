# gemma3n-executorch

export google gemma3n instruction-tuned models (text-only) to executorch pte format for mobile deployment

> [!WARNING]
> These exports contain only the text decoder (language_model + lm_head). The vision and audio towers from the full multimodal gemma3n models are not included. Use these exports for text-only inference tasks.

## models

| model | quant | text params | layers | size | download |
|-------|-------|-------------|--------|------|----------|
| e2b-it | bf16 | 4.46b | 30 | 8.4 gb | [huggingface](https://huggingface.co/macmacmacmac/gemma-3n-E2B-it-pte) |
| e2b-it | int8 | 4.46b | 30 | 7.0 gb | [huggingface](https://huggingface.co/macmacmacmac/gemma-3n-E2B-it-pte) |
| e4b-it | bf16 | 7.40b | 35 | 13.1 gb | [huggingface](https://huggingface.co/macmacmacmac/gemma-3n-E4B-it-pte) |
| e4b-it | int8 | 7.40b | 35 | 9.6 gb | [huggingface](https://huggingface.co/macmacmacmac/gemma-3n-E4B-it-pte) |

source models: [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)

## requirements

```bash
pip install torch>=2.5.0 transformers>=4.48.0 executorch>=0.5.0
```

## known issues and required patches

two bugs in transformers prevent direct export of gemma3n models

### issue 1 - missing self.training guard causes constant is mutated error

file: transformers/models/gemma3n/modeling_gemma3n.py line 1119

```diff
- if self.config.altup_coef_clip is not None:
+ if self.training and self.config.altup_coef_clip is not None:
      self.correction_coefs.weight.data.clamp_(...)
```

### issue 2 - dynamic erfinv not supported by executorch

file: transformers/models/gemma3n/modeling_gemma3n.py - gemma3ntextmlp class

the _gaussian_topk method calls torch.distributions.normal.Normal.icdf() during forward pass which uses erfinv - an operator not in executorch core aten opset

fix: pre-compute std_multiplier at init time instead of during forward

```python
# in __init__ add after self.activation_sparsity = ...
if self.activation_sparsity > 0.0:
    normal_dist = torch.distributions.normal.Normal(0, 1)
    std_multiplier = normal_dist.icdf(torch.tensor(self.activation_sparsity, dtype=torch.float32))
    self.register_buffer("_std_multiplier", std_multiplier, persistent=False)

# replace _gaussian_topk method
def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
    std_multiplier = self._std_multiplier.to(inputs.dtype)
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    cutoff_x = inputs_mean + inputs_std * std_multiplier
    return nn.functional.relu(inputs - cutoff_x)
```

## usage

```bash
# apply patches first then
cd e2b && python export.py
cd e4b && python export.py
```

## validation

```bash
python validate.py /path/to/model.pte
```

## environment

- pytorch 2.9.1
- transformers 5.0.0rc1
- executorch 1.0.1
- python 3.11
- ubuntu 24.04
