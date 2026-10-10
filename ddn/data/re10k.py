"""RealEstate10K frame dataset for teacher-student depth distillation."""

from __future__ import annotations

import json
import random
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset


class Re10KFrameDataset(IterableDataset):
    """Stream individual RGB frames from ReSplat-style RealEstate10K chunks.

    The expected layout is the one used by ReSplat:

        <root>/<split>/*.torch

    Each chunk contains a list of scene dictionaries with an ``images`` field,
    where each image is a tensor of encoded image bytes.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        split: str = "train",
        img_size: Tuple[int, int] = (252, 448),
        max_chunks: Optional[int] = None,
        max_frames: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1234,
        pseudo_depth_root: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(root) for root in roots]
        self.split = split
        self.img_size = img_size
        self.max_chunks = max_chunks
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        # When set, precomputed DA pseudo-depth (disparity) is loaded from
        # <pseudo_depth_root>/<split>/<chunk_stem>.torch and returned as
        # "target". Labels are keyed by (scene_idx, frame_idx) in the ORIGINAL
        # unshuffled chunk order — see scripts/prepare_pseudo_depth.py.
        self.pseudo_depth_root = (
            Path(pseudo_depth_root) / split if pseudo_depth_root is not None else None
        )
        self._pseudo_cache_path: Optional[Path] = None
        self._pseudo_cache: Optional[dict] = None

        self.chunks = self._collect_chunks()
        if not self.chunks:
            searched = ", ".join(str(root / split) for root in self.roots)
            raise FileNotFoundError(f"No Re10K .torch chunks found under: {searched}")

        self.transform = T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

    def _collect_chunks(self) -> list[Path]:
        chunks: list[Path] = []
        for root in self.roots:
            split_root = root / self.split
            if not split_root.exists():
                continue

            index_path = split_root / "index.json"
            if index_path.exists():
                with index_path.open("r") as f:
                    index = json.load(f)
                root_chunks = sorted({split_root / rel for rel in index.values()})
            else:
                root_chunks = sorted(split_root.glob("*.torch"))
            chunks.extend(path for path in root_chunks if path.exists())

        chunks = sorted(set(chunks))
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
        return chunks

    def __iter__(self) -> Iterable[dict]:
        chunks = list(self.chunks)
        worker = torch.utils.data.get_worker_info()
        rank, world_size = self._distributed_info()

        if self.shuffle:
            rng = random.Random(self.seed + rank)
            rng.shuffle(chunks)

        shard_id = rank
        num_shards = world_size
        if worker is not None:
            shard_id = rank * worker.num_workers + worker.id
            num_shards = world_size * worker.num_workers

        all_chunks = chunks
        chunks = [
            chunk
            for index, chunk in enumerate(chunks)
            if index % num_shards == shard_id
        ]
        if not chunks and all_chunks:
            chunks = [all_chunks[shard_id % len(all_chunks)]]

        emitted = 0
        for chunk_path in chunks:
            chunk = self._load_chunk(chunk_path)
            pseudo = self._load_pseudo(chunk_path)

            # Enumerate BEFORE shuffling so scene_idx / frame_idx stay aligned
            # with the original chunk order used by the pseudo-depth precompute.
            indexed_scenes = list(enumerate(chunk))
            if self.shuffle:
                rng = random.Random(self.seed + self._stable_hash(chunk_path.name))
                rng.shuffle(indexed_scenes)

            for scene_idx, scene in indexed_scenes:
                scene_key = scene.get("key", chunk_path.stem)
                indexed_imgs = list(enumerate(scene["images"]))
                if self.shuffle:
                    rng.shuffle(indexed_imgs)

                for frame_idx, encoded in indexed_imgs:
                    target = self._lookup_pseudo(pseudo, scene_idx, frame_idx, chunk_path)
                    if pseudo is not None and target is None:
                        continue  # label missing for this frame; skip rather than crash
                    image = self._decode_image(encoded)
                    item = {
                        "image": self.transform(image),
                        "scene": scene_key,
                        "frame_idx": frame_idx,
                        "chunk": str(chunk_path),
                    }
                    if target is not None:
                        item["target"] = target
                    yield item
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        return

    def __len__(self) -> int:
        if self.max_frames is None:
            raise TypeError("Re10KFrameDataset length is unknown unless max_frames is set")
        return self.max_frames

    def _load_pseudo(self, chunk_path: Path) -> Optional[dict]:
        """Load (and cache) the pseudo-depth label file matching a chunk.

        Returns the dict {scene_idx: fp16 tensor [N, Hs, Ws]} or None when no
        pseudo-depth root is configured. Caches the most recent chunk so the
        per-frame lookups inside a chunk don't re-read the file.
        """
        if self.pseudo_depth_root is None:
            return None
        path = self.pseudo_depth_root / f"{chunk_path.stem}.torch"
        if path != self._pseudo_cache_path:
            self._pseudo_cache_path = path
            self._pseudo_cache = self._load_chunk(path) if path.exists() else None
        return self._pseudo_cache

    @staticmethod
    def _lookup_pseudo(
        pseudo: Optional[dict],
        scene_idx: int,
        frame_idx: int,
        chunk_path: Path,
    ) -> Optional[Tensor]:
        if pseudo is None:
            return None
        scene_disp = pseudo.get(scene_idx)
        if scene_disp is None or frame_idx >= scene_disp.shape[0]:
            return None
        # (Hs, Ws) fp16 -> (1, Hs, Ws) float for stacking + loss.
        return scene_disp[frame_idx].unsqueeze(0).float()

    @staticmethod
    def _decode_image(encoded: Tensor | bytes | bytearray) -> Image.Image:
        if isinstance(encoded, Tensor):
            data = encoded.cpu().numpy().tobytes()
        else:
            data = bytes(encoded)
        return Image.open(BytesIO(data)).convert("RGB")

    @staticmethod
    def _load_chunk(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _stable_hash(text: str) -> int:
        return sum((index + 1) * byte for index, byte in enumerate(text.encode("utf8")))

    @staticmethod
    def _distributed_info() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1


class Re10KSceneGroupDataset(IterableDataset):
    """Yield GROUPS of frames from the same scene, with multi-view-consistent
    pseudo-depth, for the scale-consistent (per-scene) distillation loss.

    Each item is one group: ``image`` (G,3,h,w) + ``target`` (G,1,Hs,Ws), all
    from one scene, so a per-scene joint normalisation in the loss can see the
    cross-frame relative scale. Pair with :class:`GroupSSILoss`.

    Reads the MVC pseudo-depth written by scripts/prepare_multiview_depth.py:
        <mvc_root>/<split>/<chunk_stem>.torch
        = {scene_idx: {"frames": Long[K], "depth": fp16[K,Hs,Ws], "accept": bool}}
    Only the stored keyframes (chosen for parallax) are used. ``require_accept``
    keeps only scenes whose alignment passed the confidence gate.
    """

    IMAGENET_MEAN = Re10KFrameDataset.IMAGENET_MEAN
    IMAGENET_STD = Re10KFrameDataset.IMAGENET_STD

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        mvc_root: str | Path,
        split: str = "train",
        img_size: Tuple[int, int] = (252, 448),
        group_size: int = 4,
        max_chunks: Optional[int] = None,
        require_accept: bool = False,
        shuffle: bool = True,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(root) for root in roots]
        self.mvc_root = Path(mvc_root) / split
        self.split = split
        self.img_size = img_size
        self.group_size = group_size
        self.max_chunks = max_chunks
        self.require_accept = require_accept
        self.shuffle = shuffle
        self.seed = seed
        self.chunks = Re10KFrameDataset._collect_chunks(self)
        if not self.chunks:
            raise FileNotFoundError(f"No Re10K chunks under {self.roots}/{split}")
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def __iter__(self) -> Iterable[dict]:
        chunks = list(self.chunks)
        worker = torch.utils.data.get_worker_info()
        rank, world_size = Re10KFrameDataset._distributed_info()
        if self.shuffle:
            random.Random(self.seed + rank).shuffle(chunks)
        shard_id, num_shards = rank, world_size
        if worker is not None:
            shard_id = rank * worker.num_workers + worker.id
            num_shards = world_size * worker.num_workers
        chunks = [c for i, c in enumerate(chunks) if i % num_shards == shard_id]

        for chunk_path in chunks:
            mvc_path = self.mvc_root / f"{chunk_path.stem}.torch"
            if not mvc_path.exists():
                continue
            mvc = Re10KFrameDataset._load_chunk(mvc_path)
            chunk = Re10KFrameDataset._load_chunk(chunk_path)
            scene_ids = list(mvc.keys())
            if self.shuffle:
                random.Random(self.seed + self._h(chunk_path.name)).shuffle(scene_ids)
            for sidx in scene_ids:
                rec = mvc[sidx]
                if self.require_accept and not bool(rec.get("accept", True)):
                    continue
                frames = rec["frames"]
                depth = rec["depth"]
                k = int(frames.shape[0])
                if k == 0:
                    continue
                rng = random.Random(self.seed + self._h(f"{chunk_path.name}{sidx}"))
                order = list(range(k))
                if self.shuffle:
                    rng.shuffle(order)
                sel = (order * ((self.group_size + k - 1) // k))[: self.group_size]  # pad w/ repeats if k<G
                images, targets = [], []
                scene = chunk[sidx]
                for j in sel:
                    orig = int(frames[j].item())
                    pil = Re10KFrameDataset._decode_image(scene["images"][orig])
                    images.append(self.transform(pil))
                    targets.append(depth[j].unsqueeze(0).float())
                yield {
                    "image": torch.stack(images),       # (G,3,h,w)
                    "target": torch.stack(targets),     # (G,1,Hs,Ws)
                    "scene": scene.get("key", chunk_path.stem),
                }

    @staticmethod
    def _h(text: str) -> int:
        return Re10KFrameDataset._stable_hash(text)
