#!/usr/bin/env python3
"""Pretrain DepthSplat's StudentMonoShim to mimic DA/DINO mono features.

DepthSplat consumes DepthAnything/DINO intermediate tokens, not only the final
monocular depth map. This script makes the replacement StudentMonoShim speak the
same feature "interface" before plugging it into DepthSplat training.

It trains:
  RGB -> frozen DA/DINO teacher get_intermediate_layers(...)
  RGB -> StudentMonoShim get_intermediate_layers(...)

with a cosine + channel-normalized L1 feature loss over the four tapped layers.
The saved checkpoint contains the full shim state_dict, including the 1x1
projection heads, and can be used as:

  model.encoder.student_mono=true
  model.encoder.student_mono_ckpt=/path/to/student_mono_mimic/best.pth
  model.encoder.student_mono_arch='s32:...|s16:...|s8:...|s4:...'
  checkpointing.pretrained_monodepth=null
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "depthsplat"
        / "src"
        / "model"
        / "encoder"
        / "unimatch"
    ),
)

from ddn.data.re10k import Re10KFrameDataset  # noqa: E402
from ddn.vendors.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from student_mono import StudentMonoShim  # noqa: E402


logger = logging.getLogger(__name__)

DA_CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}

TEACHER_LAYERS = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
}

TEACHER_DIM = {"vits": 384, "vitb": 768, "vitl": 1024}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", default="checkpoints/student_mono_mimic_da")
    p.add_argument("--teacher_arch", choices=list(DA_CONFIGS), default="vits")
    p.add_argument("--teacher_ckpt", required=True, help="DepthAnything-V2 checkpoint")
    p.add_argument("--student_arch", default=None)
    p.add_argument("--student_ckpt", default=None, help="Existing StudentNet or shim checkpoint")
    p.add_argument("--student_encoder_weight_path", default=None)
    p.add_argument("--student_decoder_channels", type=int, default=256)
    p.add_argument("--freeze_student", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--unfreeze_student_encoder_bias", action="store_true")
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr_proj", type=float, default=2e-4)
    p.add_argument("--lr_student", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lambda_l1", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_teacher(arch: str, ckpt: str, device: torch.device) -> nn.Module:
    model = DepthAnythingV2(**DA_CONFIGS[arch])
    state = torch.load(ckpt, map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    cleaned = {}
    for name, value in state.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded teacher %s: missing=%d unexpected=%d", ckpt, len(missing), len(unexpected))
    teacher = model.pretrained.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_student(args, device: torch.device) -> StudentMonoShim:
    shim = StudentMonoShim(
        arch=args.student_arch,
        checkpoint=args.student_ckpt,
        encoder_weight_path=args.student_encoder_weight_path,
        decoder_channels=args.student_decoder_channels,
        out_dim=TEACHER_DIM[args.teacher_arch],
        pretrained_encoder=True,
        freeze_student=args.freeze_student,
    ).to(device)
    if args.unfreeze_student_encoder_bias and not args.freeze_student:
        shim.student.enable_retrain_mode()
    return shim


def infinite(loader: DataLoader) -> Iterable[dict]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("Re10K dataloader produced no batches")


def depth_splat_mono_input(images: torch.Tensor) -> torch.Tensor:
    """Match DepthSplat's mono branch resize: floor H/W to multiples of 14."""
    h, w = images.shape[-2:]
    rh, rw = h // 14 * 14, w // 14 * 14
    if (rh, rw) == (h, w):
        return images
    return F.interpolate(images, (rh, rw), mode="bilinear", align_corners=True)


def channel_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(eps)


def mimic_loss(student_feats, teacher_feats, lambda_l1: float):
    total = student_feats[0].sum() * 0.0
    logs = {}
    for i, (s, t) in enumerate(zip(student_feats, teacher_feats)):
        t = t.detach()
        cos = 1.0 - (F.normalize(s.float(), dim=-1) * F.normalize(t.float(), dim=-1)).sum(dim=-1)
        cos = cos.mean()
        l1 = F.smooth_l1_loss(channel_standardize(s), channel_standardize(t))
        loss = cos + lambda_l1 * l1
        total = total + loss
        logs[f"cos{i}"] = cos.detach()
        logs[f"l1_{i}"] = l1.detach()
    return total / len(student_feats), logs


def save_checkpoint(path: Path, shim: StudentMonoShim, args, step: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "shim_state_dict": shim.state_dict(),
            "step": step,
            "loss": loss,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    teacher = build_teacher(args.teacher_arch, args.teacher_ckpt, device)
    student = build_student(args, device)
    student.train()

    dataset = Re10KFrameDataset(
        roots=args.data_root,
        split=args.split,
        img_size=(args.img_h, args.img_w),
        max_chunks=args.max_chunks,
        max_frames=args.max_frames,
        shuffle=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    batches = infinite(loader)

    proj_params = list(student.proj.parameters())
    student_params = [p for n, p in student.named_parameters() if not n.startswith("proj.") and p.requires_grad]
    param_groups = [{"params": proj_params, "lr": args.lr_proj}]
    if student_params:
        param_groups.append({"params": student_params, "lr": args.lr_student})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        [g["lr"] for g in param_groups],
        total_steps=args.steps,
        pct_start=0.03,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    logger.info(
        "Training mimic: teacher=%s batch=%d steps=%d lr_proj=%.2e lr_student=%.2e student_params=%d",
        args.teacher_arch,
        args.batch_size,
        args.steps,
        args.lr_proj,
        args.lr_student,
        sum(p.numel() for p in student_params),
    )

    out_dir = Path(args.output_dir)
    best = float("inf")
    running = 0.0
    for step in range(1, args.steps + 1):
        batch = next(batches)
        images = batch["image"].to(device, non_blocking=True)
        x = depth_splat_mono_input(images)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_feats = teacher.get_intermediate_layers(
                x,
                TEACHER_LAYERS[args.teacher_arch],
                return_class_token=False,
            )

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            student_feats = student.get_intermediate_layers(
                x,
                TEACHER_LAYERS[args.teacher_arch],
                return_class_token=False,
            )
            loss, parts = mimic_loss(student_feats, teacher_feats, args.lambda_l1)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

        value = float(loss.detach().item())
        running += value
        if value < best:
            best = value
            save_checkpoint(out_dir / "best.pth", student, args, step, best)

        if args.log_every > 0 and step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            detail = " ".join(f"{k}={float(v):.3f}" for k, v in sorted(parts.items())[:4])
            logger.info("step=%d/%d loss=%.4f best=%.4f %s", step, args.steps, avg, best, detail)

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pth", student, args, step, value)

    save_checkpoint(out_dir / "last.pth", student, args, args.steps, value)
    logger.info("Done. best=%.4f out=%s", best, out_dir)


if __name__ == "__main__":
    main()
