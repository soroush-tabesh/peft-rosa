# Upstream PEFT PR checklist (RoSA)

Draft PR: `soroush-tabesh/peft-rosa:rosa-tuner` → `huggingface/peft:main`

## Summary

Adds **RoSA** (Robust Adaptation): combined low-rank (LoRA) and sparse adapters for parameter-efficient fine-tuning. Paper: https://arxiv.org/abs/2401.04679

## Checklist

- [x] `PeftType.ROSA` in `peft_types.py`
- [x] Tuner package `src/peft/tuners/rosa/`
- [x] `register_peft_method(name="rosa", ...)`
- [x] Exports from `peft/__init__.py` and `peft/tuners/__init__.py`
- [x] `get_peft_model_state_dict` branch for ROSA
- [x] Docs: `docs/source/package_reference/rosa.md` + toctree
- [x] Tests: `test_custom_models`, `test_config`, `test_decoder_models`, `test_initialization`
- [x] `method_comparison/MetaMathQA/experiments/rosa/` configs
- [x] Example: `examples/rosa_finetuning/`
- [ ] GPU regression / full CI on fork before marking ready for review

## Notes

- `spops` is optional at import time; required when sparse density `d > 0`.
- `RosaScheduler` is a `TrainerCallback` for mask generation schedules (`wl64`, `lora_only`, etc.).
- Reuses `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` for default target modules.
