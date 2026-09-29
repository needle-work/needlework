"""The training loop shared by every component.

Per step: bf16 autocast forward, fp32 backward, gradient clipping, AdamW, cosine
schedule with warmup, EMA update. Per epoch: the task's metrics on the EMA model
(``val_loss`` every epoch, others at their own cadence), one metrics record, and a
checkpoint every ``checkpoint.every`` epochs and after the last.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from typing import Protocol

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from needlework.models.ema import Ema
from needlework.training import determinism
from needlework.training.batches import BatchSource, epoch_order, to_device


class Task(Protocol):
    model: nn.Module
    train_data: BatchSource
    val_data: BatchSource

    def loss(self, model: nn.Module, batch: dict) -> torch.Tensor: ...

    def epoch_metrics(self, ema_model: nn.Module, epoch: int) -> dict[str, float]:
        """Metrics after ``epoch`` completed epochs: always ``val_loss`` over the
        whole validation set, plus any metric that is due."""
        ...


class Trainer:
    def __init__(
        self,
        task: Task,
        cfg: DictConfig,
        *,
        device: torch.device,
        data_generator: torch.Generator,
    ) -> None:
        self.task = task
        self.cfg = cfg
        self.device = device
        self.data_generator = data_generator
        self.model = task.model.to(device)
        self.ema = Ema(copy.deepcopy(self.model), **cfg.ema)
        optim = cfg.optim
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optim.lr,
            betas=tuple(optim.betas),
            eps=optim.eps,
            weight_decay=optim.weight_decay,
            fused=True,
        )
        batch_size = cfg.component.train.batch_size
        steps_per_epoch = math.ceil(len(task.train_data) / batch_size)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=optim.warmup_steps,
            num_training_steps=steps_per_epoch * cfg.component.train.epochs,
        )
        self.epoch = 0  # completed epochs
        self.global_step = 0

    def state(self) -> dict:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "ema": self.ema.model.state_dict(),
            "ema_step": self.ema.step_count,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": determinism.capture(self.data_generator),
        }

    def load_state(self, payload: dict) -> None:
        self.epoch = payload["epoch"]
        self.global_step = payload["global_step"]
        self.model.load_state_dict(payload["model"])
        self.ema.model.load_state_dict(payload["ema"])
        self.ema.step_count = payload["ema_step"]
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        determinism.restore(payload["rng"], self.data_generator)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.task.train_data.set_epoch(self.epoch)
        clip = self.cfg.optim.clip_grad
        losses, norms = [], []
        order = epoch_order(
            len(self.task.train_data),
            self.cfg.component.train.batch_size,
            self.data_generator,
        )
        for indices in tqdm(order, desc=f"epoch {self.epoch}", leave=False):
            batch = to_device(self.task.train_data.batch(indices), self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.task.loss(self.model, batch)
            loss.backward()
            norms.append(nn.utils.clip_grad_norm_(self.model.parameters(), clip))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.ema.update(self.model)
            losses.append(loss.detach())
            self.global_step += 1
        return {
            "train_loss": float(torch.stack(losses).mean()),
            "grad_norm": float(torch.stack(norms).mean()),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def run(
        self,
        *,
        log: Callable[[dict], None],
        save: Callable[[dict, dict], None],
    ) -> None:
        cfg = self.cfg
        epochs = cfg.component.train.epochs
        while self.epoch < epochs:
            started = time.perf_counter()
            record = {"epoch": self.epoch, **self._train_epoch()}
            self.epoch += 1
            record["global_step"] = self.global_step
            record["time/train_s"] = time.perf_counter() - started
            started = time.perf_counter()
            record.update(self.task.epoch_metrics(self.ema.model, self.epoch))
            record["time/metrics_s"] = time.perf_counter() - started
            log(record)
            if (
                self.epoch % cfg.checkpoint.every == 0
                or self.epoch == epochs
                or "eval/success_rate" in record
            ):
                save(self.state(), record)
