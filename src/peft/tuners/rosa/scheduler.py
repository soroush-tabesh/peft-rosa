# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any

import torch

from .hooks import ManualGradCollectorHook, SaveInputHook
from .layer import RosaLayer
from .model import RosaModel

logger = logging.getLogger(__name__)


class RosaScheduler:
    """RoSA training schedule: LoRA warmup, gradient collection, and sparse mask activation.

    Duck-typed for `transformers.TrainerCallback`. We intentionally do not subclass
    `TrainerCallback` to avoid a circular import: importing ``transformers`` here triggers
    ``transformers.trainer_utils`` which does ``from peft import PeftMixedModel, PeftModel``
    while ``peft`` is still initializing. HF's ``CallbackHandler`` only invokes the
    ``on_*`` hooks by name, so duck typing is sufficient.
    """

    def __init__(self, model: RosaModel) -> None:
        self._model = model

        config = model.peft_config
        if len(config) != 1 or "default" not in config:
            raise ValueError("RosaScheduler currently supports a single adapter named 'default'.")

        rosa_config = config["default"]

        self._mask_load_path = getattr(rosa_config, "mask_load_path", None)
        self._mask_save_path = getattr(rosa_config, "mask_save_path", None)
        self._spa_num_grads = getattr(rosa_config, "spa_num_grads", 1)
        self._grad_acc_mode = getattr(rosa_config, "grad_acc_mode", "mean_squared")
        self._terminate_after_mask_generation = getattr(rosa_config, "terminate_after_mask_generation", False)

        self._d = getattr(rosa_config, "d", 0.0)
        self._r = getattr(rosa_config, "r", 0)

        if self._mask_load_path is not None and self._mask_save_path is not None:
            raise ValueError("Only one of mask_save_path and mask_load_path may be set.")

        if self._d > 0:
            if self._terminate_after_mask_generation:
                if self._mask_save_path is None:
                    raise ValueError("mask_save_path is required when terminate_after_mask_generation is True.")
                if self._mask_load_path is not None:
                    raise ValueError("mask_load_path must be None during mask generation.")

            if self._mask_load_path is not None:
                self._set_spa_masks(torch.load(self._mask_load_path, weights_only=True))

        schedule_name = getattr(rosa_config, "schedule", None)
        self._schedule = self._create_schedule(schedule_name)

        self._step = 0
        self._handles: list[Any] = []

    def _create_schedule(self, schedule_name: str | None) -> list[dict]:
        if schedule_name is None:
            raise ValueError("RoSA schedule must be specified (e.g. 'wl64', 'default', 'lora_only').")

        if schedule_name in ("default", "df"):
            return self._create_schedule("wl0")

        if schedule_name == "spa_only":
            if self._d <= 0:
                raise ValueError("spa_only schedule requires density d > 0.")
            return self._generate_spa_schedule(self._mask_load_path is None)

        if schedule_name == "lora_only":
            if self._d != 0:
                raise ValueError("lora_only schedule requires density d = 0.")
            return self._generate_lora_schedule()

        if schedule_name.startswith("wl"):
            if schedule_name != "wl0" and self._d <= 0:
                raise ValueError("wl schedule requires density d > 0.")
            lora_warmup_steps = int(schedule_name.split("wl")[-1])
            return self._generate_wl_schedule(lora_warmup_steps, self._mask_load_path is None)

        raise ValueError(
            f"RoSA schedule {schedule_name!r} is not implemented "
            "(supported: default, lora_only, spa_only, wl<N>)."
        )

    def _generate_spa_schedule(self, grad_collection_needed: bool) -> list[dict]:
        schedule = []
        if grad_collection_needed:
            schedule.append({"agenda": ["grad_collection"], "end": self._spa_num_grads})
        schedule.append({"agenda": ["spa"], "end": None})
        return schedule

    def _generate_lora_schedule(self) -> list[dict]:
        return [{"agenda": ["lora"], "end": None}]

    def _generate_wl_schedule(self, warmup: int, grad_collection_needed: bool) -> list[dict]:
        schedule = []
        if warmup > 0:
            schedule.append({"agenda": ["lora"], "end": warmup})
        if grad_collection_needed:
            schedule.append({"agenda": ["lora", "grad_collection"], "end": warmup + self._spa_num_grads})
        schedule.append({"agenda": ["lora", "spa"], "end": None})
        return schedule

    def _get_agenda(self, step: int) -> list[str]:
        for item in self._schedule:
            if item["end"] is None or step < item["end"]:
                return item["agenda"]
        raise RuntimeError(f"No RoSA agenda for training step {step}.")

    def _get_current_agenda(self) -> list[str]:
        return self._get_agenda(self._step)

    def _get_next_agenda(self) -> list[str]:
        return self._get_agenda(self._step + 1)

    def _set_spa_masks(self, masks: dict[str, torch.Tensor]) -> None:
        self._model.set_spa_masks(masks)

    def on_step_begin(self, args, state, control, **kwargs):
        self._on_step_begin()

    def on_step_end(self, args, state, control, **kwargs):
        self._on_step_end()

    @torch.no_grad()
    def _on_step_begin(self) -> None:
        agenda = self._get_current_agenda()
        logger.debug("RoSA agenda at step %d: %s", self._step, agenda)

        for _, param in self._model.named_parameters():
            param.requires_grad = False

        if self._mask_load_path is not None and not self._model.spa_activated:
            logger.info("Loading RoSA sparse masks from %s", self._mask_load_path)
            masks = torch.load(self._mask_load_path, weights_only=True)
            self._set_spa_masks(masks)

        for name, module in self._model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            weight = module.find_weight()
            if "grad_collection" in agenda and not self._model.spa_activated:
                handle1 = module.register_forward_hook(SaveInputHook(name, module))
                handle2 = module.register_full_backward_hook(
                    ManualGradCollectorHook(name, module, self._grad_acc_mode)
                )
                self._handles.append(handle1)
                self._handles.append(handle2)
            elif weight.is_floating_point():
                weight.requires_grad = False

            module.set_lora_requires_grad("lora" in agenda)

            if self._model.spa_activated:
                module.set_spa_requires_grad("spa" in agenda)

    @torch.no_grad()
    def _on_step_end(self) -> None:
        agenda = self._get_current_agenda()
        next_agenda = self._get_next_agenda()

        if (
            not self._model.spa_activated
            and "grad_collection" in agenda
            and "grad_collection" not in next_agenda
        ):
            logger.info("Finished RoSA gradient collection; generating sparse masks.")
            self._generate_masks_and_activate_spa(self._model)

        for handle in self._handles:
            handle.remove()

        self._handles = []
        self._step += 1

    @torch.no_grad()
    def _grad_to_mask_fn(self, grad: torch.Tensor) -> torch.Tensor:
        idx = torch.topk(torch.abs(grad.flatten()).float(), int(self._d * grad.numel()), sorted=False).indices
        mask = torch.zeros_like(grad.flatten())
        mask.scatter_(0, idx, 1.0)
        return mask.reshape_as(grad).bool()

    @torch.no_grad()
    def _generate_masks_and_activate_spa(self, model: RosaModel) -> None:
        if self._d <= 0:
            raise ValueError("Mask generation requires sparse density d > 0.")

        masks = {}
        for name, module in model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            if not hasattr(module, "collected_grad"):
                raise RuntimeError(
                    f"Module {name} is missing collected gradients for mask generation."
                )

            logger.info(
                "Generating sparse mask for %s from %d gradient(s).",
                name,
                module.collected_grad_cnt,
            )
            masks[name] = self._grad_to_mask_fn(module.collected_grad)
            delattr(module, "collected_grad")
            delattr(module, "collected_grad_cnt")
            if hasattr(module, "saved_input"):
                delattr(module, "saved_input")

        if self._mask_save_path is not None:
            logger.info("Saving RoSA sparse masks to %s", self._mask_save_path)
            torch.save(masks, self._mask_save_path)

        if self._terminate_after_mask_generation:
            logger.info("RoSA mask generation complete; terminating training.")
            raise SystemExit(0)

        self._set_spa_masks(masks)
