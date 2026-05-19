# RoSA fine-tuning example

Minimal causal LM fine-tuning with [RoSA](https://arxiv.org/abs/2401.04679) via PEFT.

Requires `spops` when using sparse density `d > 0`. For LoRA-only smoke tests, set `d=0` and `schedule=lora_only`.

```bash
python rosa_finetuning.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./rosa-llama3 \
  --batch_size 1 \
  --num_epochs 1
```
