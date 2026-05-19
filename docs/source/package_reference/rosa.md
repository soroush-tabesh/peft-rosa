# RoSA

[RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation](https://arxiv.org/abs/2401.04679)
combines low-rank (LoRA) and sparse adapters for accurate fine-tuning with fewer trainable parameters than full fine-tuning.

## Usage

```python
from peft import RosaConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
config = RosaConfig(
    r=16,
    d=0.01,
    lora_alpha=16,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

## API

[[autodoc]] peft.tuners.rosa.config.RosaConfig
[[autodoc]] peft.tuners.rosa.model.RosaModel
