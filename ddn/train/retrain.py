"""Full retraining engine for the best NAS-searched architecture."""

import logging
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..student.student_net import StudentNet
from ..utils.metrics import compute_depth_metrics
from .losses import TotalRetrainLoss
from .schedulers import warmup_cosine_scheduler

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    n_epochs: int = 50
    lr_decoder: float = 1e-4
    lr_encoder_bias: float = 1e-5
    batch_size: int = 8
    lambda_distill: float = 0.1
    warmup_epochs: int = 5
    weight_decay: float = 0.01
    lr_schedule: str = "epoch"
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0
    accum_steps: int = 1
    save_every: int = 10
    log_every_batches: int = 50
    val_every_batches: int = 0
    min_depth_eval: float = 1e-3
    max_depth_eval: float = 10.0
    eigen_crop: bool = False
    garg_crop: bool = False
    post_process_eval: bool = False
    device: str = "cuda"
    amp: bool = True
    sync_bn: bool = False
    full_finetune: bool = False
    finetune_blocks: int = 0
    encoder_lr: Optional[float] = None
    output_dir: str = "checkpoints/retrain"
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False
    # Disabled by default to preserve the original fixed-epoch protocol.  When
    # set, stop only after this many *completed epochs* fail to improve d1.
    early_stop_patience: Optional[int] = None
    # Ignore the patience counter before this completed epoch.  The default
    # keeps the pre-existing early-stop behaviour unchanged.
    early_stop_min_epochs: int = 0
    # Minimum cumulative absolute d1 gain from the last qualifying reference
    # best that resets patience. best_d1.pth still tracks every strict
    # improvement for model selection.
    early_stop_min_delta: float = 0.0


class Retrainer:
    """
    Full retraining of a StudentNet with searched decoder.

    Phase transitions:
    - Decoder + scale_proj are always trainable.
    - Encoder bias params (~0.1%) are unfrozen via enable_retrain_mode().
    - Teachers (F_TA, F_TB) can be provided as live models or pre-cached dicts.

    Args:
        student:      StudentNet with searched decoder (on CPU initially).
        train_loader: DataLoader yielding {"image", "depth", "sparse_depth"}.
        val_loader:   DataLoader for validation.
        config:       RetrainConfig hyperparameters.
        boundary_teacher: optional live BoundaryTeacher for distillation.
        scale_teacher:    optional live ScaleTeacher for distillation.
    """

    def __init__(
        self,
        student: StudentNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: RetrainConfig,
        boundary_teacher: Optional[nn.Module] = None,
        scale_teacher: Optional[nn.Module] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.local_rank = getattr(config, "local_rank", 0)
        self.is_distributed = getattr(config, "distributed", False)
        self.is_main = self.local_rank == 0

        if config.sync_bn and self.is_distributed:
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            if self.is_main:
                logger.info("Converted BatchNorm layers to SyncBatchNorm.")
        elif config.sync_bn and self.is_main:
            logger.warning("--sync_bn was requested but DDP is not enabled; keeping BatchNorm unchanged.")

        self.student = student.to(self.device)
        # Encoder adaptation: full fine-tune unlocks the whole DINOv2 backbone;
        # otherwise BitFit+LN+LayerScale (~0.1%). This must run HERE (not only in
        # the launch script) because Retrainer owns the final pre-DDP state — a
        # script-side enable_full_finetune() would otherwise be silently relocked.
        n_ft = int(getattr(config, "finetune_blocks", 0) or 0)
        if n_ft > 0:
            self.student.enable_partial_finetune(n_ft)
            mode_str = f"PARTIAL fine-tune (last {n_ft} blocks)"
        elif getattr(config, "full_finetune", False):
            self.student.enable_full_finetune()
            mode_str = "FULL fine-tune"
        else:
            self.student.enable_retrain_mode()
            mode_str = "BitFit+LN+LayerScale"
        if self.is_main:
            pc = self.student.param_count()
            ratio = 100.0 * pc["encoder_trainable"] / max(pc["encoder_total"], 1)
            logger.info(
                "Encoder adaptation: %s | encoder_trainable=%d/%d (%.2f%%), decoder=%d",
                mode_str, pc["encoder_trainable"], pc["encoder_total"], ratio, pc["decoder_total"],
            )
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.student,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                # in the forward, so keep this False — True only adds a per-iter
                # autograd-graph traversal (~10% slower) for no benefit here.
                find_unused_parameters=False,
            )
        else:
            self.model = self.student

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.boundary_teacher = boundary_teacher
        self.scale_teacher = scale_teacher

        self.criterion = TotalRetrainLoss(lambda_distill=config.lambda_distill)

        self.optimizer = self._build_optimizer()
        self.lr_schedule = getattr(config, "lr_schedule", "epoch")
        self.scheduler = warmup_cosine_scheduler(
            self.optimizer,
            n_epochs=config.n_epochs,
            warmup_epochs=config.warmup_epochs,
            steps_per_epoch=len(train_loader) if self.lr_schedule == "step" else 1,
            eta_min_ratio=getattr(config, "min_lr_ratio", 0.0),
        )
        self.scaler = torch.amp.GradScaler("cuda") if config.amp and self.device.type == "cuda" else None
        self.output_dir = Path(config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._history: List[Dict] = []
        self.global_step = 0
        self.best_d1 = float("-inf")
        self.early_stop_reference_d1 = float("-inf")
        self.no_improve_epochs = 0
        self.start_epoch = 1

    def _build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = [
            {
                "params": self.student.decoder_parameters(),
                "lr": self.config.lr_decoder,
            },
            {
                # All encoder params with requires_grad=True. With BitFit+LN
                # this is biases + LayerNorm weights + LayerScale gammas; with
                # plain BitFit it falls back to biases only.
                "params": self.student.encoder_finetune_parameters(),
                # Full fine-tune wants a real (but small) encoder LR; bias-only
                # keeps the tiny lr_encoder_bias. encoder_lr overrides when set.
                "lr": (self.config.encoder_lr
                       if getattr(self.config, "encoder_lr", None) is not None
                       else self.config.lr_encoder_bias),
            },
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self) -> List[Dict]:
        stopped_early = False
        for epoch in range(self.start_epoch, self.config.n_epochs + 1):
            train_sampler = getattr(self.train_loader, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(epoch)
            if self.lr_schedule == "epoch":
                self.scheduler.step()

            log_line = f"Epoch {epoch}/{self.config.n_epochs} | train_loss={train_metrics['loss']:.4f}"

            val_metrics: Dict = {}
            if self.val_loader is not None:
                val_metrics = self._validate()
                log_line += " | " + self._format_depth_metrics(val_metrics)
                self._maybe_save_best_d1(val_metrics, epoch=epoch, step=self.global_step)

            if self.is_main:
                logger.info(log_line)
            record = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._history.append(record)

            if self.is_main and (epoch % self.config.save_every == 0 or epoch == self.config.n_epochs):
                self.save_checkpoint(f"epoch_{epoch:03d}.pth")

            # A patience unit is one fully completed epoch, never an
            # in-epoch validation.  Update this before last.pth so an
            # interruption after the epoch boundary can resume consistently.
            patience = getattr(self.config, "early_stop_patience", None)
            min_epochs = max(0, int(getattr(self.config, "early_stop_min_epochs", 0) or 0))
            min_delta = max(0.0, float(getattr(self.config, "early_stop_min_delta", 0.0) or 0.0))
            should_stop_early = False
            if patience is not None and int(patience) > 0 and self.val_loader is not None:
                # Include any in-epoch validation in this completed-epoch
                # decision: an epoch containing a new selection best does not
                # consume patience even if its final validation is lower.
                if epoch <= min_epochs:
                    # Burn-in epochs neither consume nor pre-fill patience.
                    self.no_improve_epochs = 0
                    self.early_stop_reference_d1 = self.best_d1
                else:
                    reference = self.early_stop_reference_d1
                    if reference == float("-inf"):
                        improved_d1 = self.best_d1 > reference
                    elif min_delta == 0.0:
                        improved_d1 = self.best_d1 > reference
                    else:
                        # Small gains accumulate because the reference moves
                        # only after the total gain reaches min_delta.
                        improved_d1 = self.best_d1 >= reference + min_delta
                    if improved_d1:
                        self.no_improve_epochs = 0
                        self.early_stop_reference_d1 = self.best_d1
                    else:
                        self.no_improve_epochs += 1
                should_stop_early = (
                    epoch > min_epochs and self.no_improve_epochs >= int(patience)
                )
            # Always refresh last.pth at every epoch boundary for resume.
            self.save_checkpoint("last.pth")

            if should_stop_early:
                stopped_early = True
                if self.is_main:
                    logger.info(
                        "Early stopping at epoch=%d: no d1 improvement for %d completed epochs "+
                        "(best_d1=%.4f, min_epochs=%d, min_delta=%.6f).",
                        epoch, self.no_improve_epochs, self.best_d1, min_epochs, min_delta,
                    )
                break

        if self.is_main:
            # This marker is written only after the final epoch checkpoint has
            # completed. Launchers use it instead of assuming epoch_050.pth.
            self._write_completion_marker(
                epoch=max((r.get("epoch", 0) for r in self._history), default=0),
                stopped_early=stopped_early,
            )

        return self._history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.student.train()
        total_loss = 0.0
        window_loss = 0.0
        window_batches = 0
        n_batches = 0
        accum_steps = max(1, int(getattr(self.config, "accum_steps", 1) or 1))

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader, 1):
            self.global_step += 1
            should_step = batch_idx % accum_steps == 0 or batch_idx == len(self.train_loader)
            images = batch["image"].to(self.device)
            gt_depth = batch["depth"].to(self.device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device).bool()
            sparse = batch.get("sparse_depth")
            if sparse is not None:
                sparse = sparse.to(self.device)

            F_TA, F_TB = self._get_teacher_feats(images, sparse)

            sync_ctx = (
                self.model.no_sync()
                if self.is_distributed and hasattr(self.model, "no_sync") and not should_step
                else nullcontext()
            )
            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                    student_out = self.model(images)
                    loss_dict = self.criterion(student_out, gt_depth, F_TA, F_TB, mask=valid_mask)
                    loss = loss_dict["total"]
                    loss_for_backward = loss / accum_steps

                if self.scaler is not None:
                    self.scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if should_step and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            elif should_step:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if should_step and self.lr_schedule == "step":
                self.scheduler.step()

            total_loss += loss.item()
            window_loss += loss.item()
            window_batches += 1
            n_batches += 1

            log_every = getattr(self.config, "log_every_batches", 0)
            if log_every and self.global_step % log_every == 0:
                window = torch.tensor(
                    [window_loss, window_batches],
                    device=self.device,
                    dtype=torch.float64,
                )
                if self.is_distributed:
                    dist.all_reduce(window, op=dist.ReduceOp.SUM)
                if self.is_main:
                    lr_decoder = self.optimizer.param_groups[0]["lr"]
                    lr_bias = self.optimizer.param_groups[1]["lr"]
                    logger.info(
                        f"Epoch {epoch}/{self.config.n_epochs} | "
                        f"batch={batch_idx}/{len(self.train_loader)} | "
                        f"step={self.global_step} | "
                        f"loss={window[0].item() / max(window[1].item(), 1.0):.4f} | "
                        f"lr_dec={lr_decoder:.2e} | lr_bias={lr_bias:.2e}"
                    )
                window_loss = 0.0
                window_batches = 0

            val_every = getattr(self.config, "val_every_batches", 0)
            if (
                val_every
                and self.val_loader is not None
                and should_step
                and self.global_step % val_every == 0
            ):
                val_metrics = self._validate()
                self.student.train()
                record = {
                    "epoch": epoch,
                    "step": self.global_step,
                    "loss": loss.item(),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                self._history.append(record)
                self._maybe_save_best_d1(val_metrics, epoch=epoch, step=self.global_step)
                self.save_checkpoint("last.pth")
                if self.is_main:
                    logger.info(
                        f"Epoch {epoch}/{self.config.n_epochs} | "
                        f"step={self.global_step} | "
                        f"train_loss={loss.item():.4f} | "
                        f"{self._format_depth_metrics(val_metrics)}"
                    )

        metrics = torch.tensor([total_loss, n_batches], device=self.device, dtype=torch.float64)
        if self.is_distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        return {"loss": (metrics[0] / metrics[1].clamp_min(1)).item()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.student.eval()
        all_metrics: List[Dict] = []
        valid_pixels = torch.zeros(2, device=self.device, dtype=torch.float64)

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            gt_depth = batch["depth"].to(self.device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device).bool()
                eval_mask = valid_mask
            else:
                eval_mask = (
                    torch.isfinite(gt_depth)
                    & (gt_depth > self.config.min_depth_eval)
                    & (gt_depth < self.config.max_depth_eval)
                )
            eval_mask = self._apply_eval_crop(eval_mask)
            valid_pixels += torch.tensor(
                [eval_mask.sum().item(), eval_mask.numel()],
                device=self.device,
                dtype=torch.float64,
            )

            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                out = self.model(images)
                depth_pred = out.get("depth")
                if depth_pred is None:
                    continue
                if getattr(self.config, "post_process_eval", False):
                    flip_out = self.model(torch.flip(images, dims=[3]))
                    depth_flipped = flip_out.get("depth")
                    if depth_flipped is not None:
                        depth_pred = self._post_process_depth(depth_pred, depth_flipped)
            depth_pred = depth_pred.float()

            import torch.nn.functional as F
            if depth_pred.shape[2:] != gt_depth.shape[2:]:
                depth_pred = F.interpolate(
                    depth_pred, size=gt_depth.shape[2:],
                    mode="bilinear", align_corners=False,
                )

            for i in range(depth_pred.shape[0]):
                sample_mask = eval_mask[i:i + 1]
                m = compute_depth_metrics(
                    depth_pred[i:i + 1],
                    gt_depth[i:i + 1],
                    mask=sample_mask,
                    min_depth=self.config.min_depth_eval,
                    max_depth=self.config.max_depth_eval,
                )
                all_metrics.append(m)

        if not all_metrics:
            return {}
        if self.is_distributed:
            dist.all_reduce(valid_pixels, op=dist.ReduceOp.SUM)
        if self.is_main and all(
            all(value != value for value in metric.values())
            for metric in all_metrics
        ):
            logger.warning(
                "Validation metrics are all NaN. Check depth_scale/max_depth: "
                f"valid_pixels={int(valid_pixels[0].item())}/"
                f"{int(valid_pixels[1].item())}."
            )

        keys = list(all_metrics[0].keys())
        values = torch.tensor(
            [sum(m[k] for m in all_metrics) for k in keys] + [len(all_metrics)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        denom = values[-1].clamp_min(1)
        return {k: (values[i] / denom).item() for i, k in enumerate(keys)}

    def _apply_eval_crop(self, mask: Tensor) -> Tensor:
        if not (getattr(self.config, "eigen_crop", False) or getattr(self.config, "garg_crop", False)):
            return mask

        crop = torch.zeros_like(mask, dtype=torch.bool)
        h, w = mask.shape[-2:]
        if getattr(self.config, "garg_crop", False):
            y0, y1 = int(0.40810811 * h), int(0.99189189 * h)
            x0, x1 = int(0.03594771 * w), int(0.96405229 * w)
        else:
            y0, y1 = int(0.3324324 * h), int(0.91351351 * h)
            x0, x1 = int(0.0359477 * w), int(0.96405229 * w)
        crop[..., y0:y1, x0:x1] = True
        return mask & crop

    @staticmethod
    def _post_process_depth(depth: Tensor, depth_flipped: Tensor) -> Tensor:
        depth_hat = torch.flip(depth_flipped, dims=[3])
        if depth_hat.shape != depth.shape:
            depth_hat = torch.nn.functional.interpolate(
                depth_hat, size=depth.shape[-2:], mode="bilinear", align_corners=False
            )
        b, c, h, w = depth.shape
        fused = 0.5 * (depth + depth_hat)
        xs = torch.linspace(0.0, 1.0, w, device=depth.device, dtype=depth.dtype)
        xs = xs.view(1, 1, 1, w).expand(b, c, h, w)
        mask = 1.0 - torch.clamp(20.0 * (xs - 0.05), 0.0, 1.0)
        mask_hat = torch.flip(mask, dims=[3])
        return mask_hat * depth + mask * depth_hat + (1.0 - mask - mask_hat) * fused

    @torch.no_grad()
    def _get_teacher_feats(
        self,
        images: Tensor,
        sparse_depth: Optional[Tensor],
    ):
        """Extract live teacher features if teachers are available."""
        F_TA, F_TB = None, None

        if self.boundary_teacher is not None:
            F_TA = self.boundary_teacher(images)

        if self.scale_teacher is not None and sparse_depth is not None:
            F_TB = self.scale_teacher(images, sparse_depth)

        return F_TA, F_TB

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str) -> None:
        if not self.is_main:
            return
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self._history,
                "global_step": self.global_step,
                "best_d1": self.best_d1,
                "early_stop_reference_d1": self.early_stop_reference_d1,
                "no_improve_epochs": self.no_improve_epochs,
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def _maybe_save_best_d1(self, metrics: Dict[str, float], epoch: int, step: int) -> bool:
        d1 = metrics.get("d1")
        if d1 is None or d1 != d1 or d1 <= self.best_d1:
            return False
        self.best_d1 = d1
        if self.is_main:
            self.save_checkpoint("best_d1.pth")
            logger.info(
                f"New best d1={d1:.4f} at epoch={epoch}, step={step}"
            )
        return True

    def _write_completion_marker(self, epoch: int, stopped_early: bool) -> None:
        """Atomically record that all final checkpoints are safely on disk."""
        if not self.is_main:
            return
        marker = self.output_dir / "training_complete.json"
        tmp = marker.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "completed_epoch": int(epoch),
                    "stopped_early": bool(stopped_early),
                    "best_d1": self.best_d1,
                    "early_stop_reference_d1": self.early_stop_reference_d1,
                    "no_improve_epochs": self.no_improve_epochs,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, marker)
        logger.info("Training completion marker: %s", marker)

    @staticmethod
    def _format_depth_metrics(metrics: Dict[str, float]) -> str:
        keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]
        return " | ".join(
            f"val_{k}={metrics.get(k, float('nan')):.4f}"
            for k in keys
        )

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.student.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self._history = state.get("history", [])
        self.global_step = state.get("global_step", 0)
        self.best_d1 = state.get("best_d1", float("-inf"))
        # Old patience checkpoints did not store a qualifying reference. Use
        # their selection best as a conservative resume baseline.
        self.early_stop_reference_d1 = state.get(
            "early_stop_reference_d1", self.best_d1
        )
        self.no_improve_epochs = int(state.get("no_improve_epochs", 0))
        # Resume after the last FULLY completed epoch. End-of-epoch history
        # records carry "epoch" without "step"; mid-epoch validation records
        # carry "step" too, so they don't count as completed epochs.
        completed = max(
            (r["epoch"] for r in self._history if "epoch" in r and "step" not in r),
            default=0,
        )
        self.start_epoch = completed + 1
        logger.info(
            f"Loaded checkpoint: {path} | resume at epoch {self.start_epoch}, "
            f"global_step={self.global_step}, best_d1={self.best_d1:.4f}"
        )
