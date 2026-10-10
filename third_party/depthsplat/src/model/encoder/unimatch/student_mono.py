"""Option-A mono branch swap: feed DDN StudentNet *decoder* features into
DepthSplat's mono-feature slot.

DepthSplat's monocular branch is a DINOv2 (Depth-Anything-V2-initialized) ViT
used purely as a feature extractor: the only call site is

    self.pretrained.get_intermediate_layers(concat, [2,5,8,11],
                                             return_class_token=False)

which returns a list of token tensors ``[BV, N, C]`` (N = grid_h*grid_w,
C = vit_feature_channel, 384 for vits). DepthSplat then reshapes those to
``[BV, C, grid_h, grid_w]`` and resizes to 1/8.

`StudentMonoShim` is a drop-in replacement for ``self.pretrained``: it exposes
the same ``get_intermediate_layers`` signature, but instead of returning raw
DINOv2 layer tokens it runs the DDN StudentNet (DINOv2-ViT-S encoder + the
NAS-searched decoder) and returns the *decoder* intermediate maps, projected to
the channel width DepthSplat expects. The cost-volume / MV-transformer / CNN
paths are untouched, so metric scale still comes from multi-view matching.

Usage (one line in MultiViewUniMatch.__init__, after super build):

    from .student_mono import StudentMonoShim
    self.pretrained = StudentMonoShim(arch=..., checkpoint=..., out_dim=vit_feature_channel)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# third_party/depthsplat/src/model/encoder/unimatch/student_mono.py -> DDN root
_DDN_ROOT = Path(__file__).resolve().parents[6]
if str(_DDN_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDN_ROOT))

from ddn.search_space.tree import DecoderTree  # noqa: E402
from ddn.student.student_net import StudentNet  # noqa: E402


class StudentMonoShim(nn.Module):
    """Drop-in replacement for DepthSplat's ``self.pretrained`` DINOv2.

    Returns DDN decoder features in DINOv2's ``get_intermediate_layers`` token
    layout, so DepthSplat's downstream reshape/interpolate/DPTHead code is
    unchanged.
    """

    # Decoder taps ordered shallow->deep / high-res->low-res, matching the
    # DINOv2 layer ordering [2,5,8,11] that DepthSplat's DPTHead expects.
    _DECODER_KEYS: List[str] = ["scale_1_4", "scale_1_8", "scale_1_16", "scale_1_32"]

    def __init__(
        self,
        arch: Optional[str] = None,
        checkpoint: Optional[str] = None,
        encoder_weight_path: Optional[str] = None,
        decoder_channels: int = 256,
        out_dim: int = 384,
        pretrained_encoder: bool = True,
        freeze_student: bool = False,
        strict_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        decoder_tree = (
            DecoderTree.from_summary(arch, channels=decoder_channels) if arch else None
        )
        self.student = StudentNet(
            decoder_tree=decoder_tree,
            decoder_channels=decoder_channels,
            weight_path=encoder_weight_path,
            pretrained_encoder=pretrained_encoder,
        )

        # 256 -> 384 projections, one per decoder tap.
        self.proj = nn.ModuleList(
            [nn.Conv2d(decoder_channels, out_dim, kernel_size=1) for _ in self._DECODER_KEYS]
        )
        self.out_dim = out_dim
        self.checkpoint_load_report: Optional[Dict[str, Any]] = None

        if checkpoint:
            self._load_checkpoint(checkpoint, strict_checkpoint=strict_checkpoint)

        if freeze_student:
            for p in self.student.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _strip_prefixes(name: str, prefixes: tuple[str, ...]) -> str:
        while True:
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            else:
                return name

    @classmethod
    def _normalize_state(
        cls,
        tensors: Dict[str, Tensor],
        prefixes: tuple[str, ...],
        checkpoint: Path,
    ) -> Dict[str, Tensor]:
        cleaned: Dict[str, Tensor] = {}
        sources: Dict[str, str] = {}
        for source_name, value in tensors.items():
            name = cls._strip_prefixes(source_name, prefixes)
            if name in cleaned:
                raise RuntimeError(
                    f"checkpoint key collision after prefix normalization: "
                    f"{sources[name]!r} and {source_name!r} -> {name!r} in {checkpoint}"
                )
            cleaned[name] = value
            sources[name] = source_name
        return cleaned

    @staticmethod
    def _tensor_state(state: object, checkpoint: Path) -> Dict[str, Tensor]:
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint is not a state dictionary: {checkpoint}")
        tensors = {name: value for name, value in state.items() if torch.is_tensor(value)}
        if not tensors:
            raise RuntimeError(f"checkpoint contains no tensors: {checkpoint}")
        return tensors

    def _load_checkpoint(self, checkpoint: str, *, strict_checkpoint: bool) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
        state: object = torch.load(checkpoint_path, map_location="cpu")
        for key in ("shim_state_dict", "model_state_dict", "state_dict", "model"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        tensors = self._tensor_state(state, checkpoint_path)

        outer_prefixes = ("module.", "depth_predictor.pretrained.", "pretrained.")
        outer_cleaned = self._normalize_state(tensors, outer_prefixes, checkpoint_path)
        shim_scope = any(name.startswith("proj.") for name in outer_cleaned)

        if shim_scope:
            target: nn.Module = self
            cleaned = outer_cleaned
            expected = set(self.state_dict())
            student_prefix = "student."
        else:
            target = self.student
            student_prefixes = ("module.", "student.", "depth_predictor.ddn.", "ddn.")
            cleaned = self._normalize_state(
                tensors, student_prefixes, checkpoint_path
            )
            expected = set(self.student.state_dict())
            student_prefix = ""

        missing, unexpected = target.load_state_dict(cleaned, strict=False)
        missing_set = set(missing)
        unexpected_set = set(unexpected)
        loaded = expected.intersection(cleaned)
        encoder_prefix = student_prefix + "encoder."
        decoder_prefix = student_prefix + "decoder."
        expected_encoder = {name for name in expected if name.startswith(encoder_prefix)}
        expected_decoder = {name for name in expected if name.startswith(decoder_prefix)}
        loaded_encoder = loaded.intersection(expected_encoder)
        loaded_decoder = loaded.intersection(expected_decoder)

        verified = 0
        if strict_checkpoint:
            problems = []
            if missing_set:
                problems.append(f"missing={sorted(missing_set)[:5]}")
            if unexpected_set:
                problems.append(f"unexpected={sorted(unexpected_set)[:5]}")
            if loaded_encoder != expected_encoder:
                absent = sorted(expected_encoder - loaded_encoder)
                problems.append(f"encoder_coverage={len(loaded_encoder)}/{len(expected_encoder)} {absent[:3]}")
            if loaded_decoder != expected_decoder:
                absent = sorted(expected_decoder - loaded_decoder)
                problems.append(f"decoder_coverage={len(loaded_decoder)}/{len(expected_decoder)} {absent[:3]}")
            if problems:
                raise RuntimeError(
                    "strict StudentMonoShim checkpoint audit failed: " + "; ".join(problems)
                )

            loaded_state = target.state_dict()
            for name in sorted(loaded):
                if not torch.equal(loaded_state[name].detach().cpu(), cleaned[name].detach().cpu()):
                    raise RuntimeError(f"checkpoint tensor was not loaded exactly: {name}")
                verified += 1

        projection_keys = sorted(name for name in self.state_dict() if name.startswith("proj."))
        self.checkpoint_load_report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256(checkpoint_path),
            "scope": "shim" if shim_scope else "student",
            "strict": strict_checkpoint,
            "checkpoint_tensors": len(cleaned),
            "expected_tensors": len(expected),
            "loaded_tensors": len(loaded),
            "verified_exact_tensors": verified,
            "missing_tensors": len(missing_set),
            "unexpected_tensors": len(unexpected_set),
            "encoder_expected": len(expected_encoder),
            "encoder_loaded": len(loaded_encoder),
            "decoder_expected": len(expected_decoder),
            "decoder_loaded": len(loaded_decoder),
            "projection_parameters_initialized_by_shim": projection_keys,
        }
        if missing_set:
            print(f"[StudentMonoShim] Missing keys: {len(missing_set)}")
        if unexpected_set:
            print(f"[StudentMonoShim] Unexpected keys: {len(unexpected_set)}")
        print(f"[StudentMonoShim] Checkpoint audit: {self.checkpoint_load_report}")

    # Mirrors dinov2.get_intermediate_layers(x, idx, return_class_token=False).
    # `n` (the layer indices) is ignored: we always return our 4 decoder taps.
    def get_intermediate_layers(
        self,
        x: Tensor,
        n,
        return_class_token: bool = False,
    ) -> List[Tensor]:
        # x: [BV, 3, H14, W14], already a multiple of 14 (DepthSplat resizes it).
        out = self.student(x)
        dec = out["dec_intermediates"]

        grid_h = x.shape[-2] // 14
        grid_w = x.shape[-1] // 14

        tokens: List[Tensor] = []
        for key, proj in zip(self._DECODER_KEYS, self.proj):
            feat = proj(dec[key])  # [BV, out_dim, h_k, w_k]
            feat = F.interpolate(
                feat, size=(grid_h, grid_w), mode="bilinear", align_corners=True
            )
            # [BV, out_dim, gh, gw] -> [BV, gh*gw, out_dim] (DINOv2 token layout)
            tok = feat.flatten(2).transpose(1, 2).contiguous()
            tokens.append(tok)
        return tokens


def attach_student_mono(unimatch: nn.Module, **kwargs) -> None:
    """Swap a built MultiViewUniMatch's DINOv2 mono branch for the DDN shim.

    Reads the expected channel width from the module so vits/vitb/vitl all work.
    """
    out_dim = getattr(unimatch, "vit_feature_channel", 384)
    device = next(unimatch.parameters()).device
    shim = StudentMonoShim(out_dim=out_dim, **kwargs).to(device)
    unimatch.pretrained = shim
