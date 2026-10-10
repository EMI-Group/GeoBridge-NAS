"""Predeclared burn-in/reference-delta early stopping for paired experiments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from pytorch_lightning import Callback, LightningModule, Trainer


DDN_ROOT = Path(__file__).resolve().parents[4]
if str(DDN_ROOT) not in sys.path:
    sys.path.insert(0, str(DDN_ROOT))

from experiments.encoding_generalization_20260812.reference_delta_stop import (  # noqa: E402
    ReferenceDeltaStop,
)


class ArmBlindEarlyStopping(Callback):
    """Stop on a validation metric without arm labels or target thresholds."""

    def __init__(
        self,
        *,
        monitor: str,
        mode: str,
        burn_in_checks: int,
        patience: int,
        min_delta: float,
        status_path: Path,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.status_path = Path(status_path)
        self.stopper = ReferenceDeltaStop(
            mode=mode,
            burn_in_checks=burn_in_checks,
            patience=patience,
            min_delta=min_delta,
        )

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}:monitor={self.monitor}:"
            f"mode={self.stopper.mode}"
        )

    def state_dict(self) -> dict[str, object]:
        return {"monitor": self.monitor, "stopper": self.stopper.state_dict()}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if state_dict.get("monitor") != self.monitor:
            raise ValueError("early-stop monitor changed while resuming")
        self.stopper.load_state_dict(state_dict["stopper"])

    def _write_status(self, payload: dict[str, object]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_name(self.status_path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.status_path)

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        should_stop = False
        if trainer.global_rank == 0:
            metric = trainer.callback_metrics.get(self.monitor)
            if metric is None:
                raise RuntimeError(
                    f"early-stop metric {self.monitor!r} was not logged by validation"
                )
            decision = self.stopper.update(float(metric.detach().cpu().item()))
            should_stop = decision.should_stop
            payload = {
                "schema_version": 1,
                "monitor": self.monitor,
                "global_step": int(trainer.global_step),
                **self.stopper.state_dict(),
                "latest_metric": decision.metric,
                "in_burn_in": decision.in_burn_in,
                "strict_best": decision.strict_best,
                "qualifying_improvement": decision.qualifying_improvement,
                "should_stop": should_stop,
            }
            self._write_status(payload)
            print(
                "arm_blind_early_stop "
                f"check={decision.check} metric={decision.metric:.6f} "
                f"best={decision.best:.6f} reference={decision.reference:.6f} "
                f"bad_checks={decision.bad_checks}/{self.stopper.patience} "
                f"burn_in={decision.in_burn_in} stop={should_stop}",
                flush=True,
            )
        stop_tensor = torch.tensor(
            int(should_stop), device=pl_module.device, dtype=torch.int32
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            trainer.should_stop = True
