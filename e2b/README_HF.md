---
license: gemma
library_name: executorch
pipeline_tag: text-generation
tags:
  - executorch
  - gemma
  - mobile
  - on-device
base_model: google/gemma-3n-E2B-it
---

# gemma-3n-E2B-it-pte

executorch .pte export of google/gemma-3n-E2B-it for on-device mobile inference

## available models

| variant | dtype | size | file |
|---------|-------|------|------|
| bf16 | bfloat16 | 8.4 gb | Gemma3n-E2B-IT-text-only.pte |
| int8 | int8 weights | 7.0 gb | Gemma3n-E2B-text-only-int8.pte |

## model details

| property | value |
|----------|-------|
| source model | google/gemma-3n-E2B-it |
| text parameters | 4.46b |
| transformer layers | 30 |
| format | executorch .pte |

## text-only export

this export contains only the text decoder components extracted from the full multimodal gemma-3n model

**included:**
- language_model (transformer decoder)
- lm_head (output projection)

**not included:**
- vision_tower (image encoder)
- audio_tower (audio encoder)

use this export for text-only inference tasks. if you need multimodal capabilities use the original huggingface model

## quantization

- **bf16**: full bfloat16 precision weights
- **int8**: int8 weight-only quantization via torchao - recommended for mobile deployment

note: int4 quantization requires gpu for inference and is not suitable for cpu-only mobile deployment

## export configuration

- fixed sequence length: 32 tokens
- torch.export with strict=False
- executorch to_edge conversion

## usage

```python
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program("Gemma3n-E2B-text-only-int8.pte")
method = program.load_method("forward")

# input_ids shape: [1, 32] dtype: torch.long
output = method.execute([input_ids])
# output shape: [1, 32, 262400] dtype: torch.bfloat16
```

## required patches

the transformers library requires two patches before export. see [maceip/gemma3n-executorch](https://github.com/maceip/gemma3n-executorch) for details

## benchmarks

coming soon

## links

- source code: https://github.com/maceip/gemma3n-executorch
- original model: https://huggingface.co/google/gemma-3n-E2B-it
