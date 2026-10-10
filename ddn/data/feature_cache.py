"""LMDB-based feature cache for pre-computed teacher/encoder features."""

import io
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False


class FeatureCache:
    """
    Read/write pre-computed feature tensors to LMDB or disk.

    Usage (write):
        cache = FeatureCache("path/to/cache", write=True, map_size_gb=300)
        cache.put(idx=0, data={"shallow": tensor, "deep": tensor})
        cache.close()

    Usage (read):
        cache = FeatureCache("path/to/cache", write=False)
        feats = cache.get(idx=0)
    """

    def __init__(
        self,
        cache_path: str,
        write: bool = False,
        map_size_gb: float = 320,
        backend: str = "lmdb",
    ):
        self.cache_path = Path(cache_path)
        self.write = write
        self.backend = backend

        if backend == "lmdb":
            if not HAS_LMDB:
                raise ImportError("lmdb not installed. Run: pip install lmdb")
            self.cache_path.mkdir(parents=True, exist_ok=True)
            map_size = int(map_size_gb * 1024 ** 3)
            self.env = lmdb.open(
                str(self.cache_path),
                map_size=map_size,
                readonly=not write,
                lock=write,
                readahead=False,
                meminit=False,
            )
        elif backend == "disk":
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.env = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _sync_mapsize(self) -> None:
        """Adopt the on-disk map size (call after MapResizedError)."""
        self.env.set_mapsize(0)

    def put(self, idx: int, data: Dict[str, Any]) -> None:
        assert self.write, "Cache opened in read-only mode"
        key = self._make_key(idx)
        value = self._serialize(data)

        if self.backend == "lmdb":
            while True:
                try:
                    with self.env.begin(write=True) as txn:
                        txn.put(key, value)
                    break
                except lmdb.MapFullError:
                    new_size = self.env.info()["map_size"] * 2
                    self.env.set_mapsize(new_size)
                except lmdb.MapResizedError:
                    self._sync_mapsize()
        else:
            path = self.cache_path / f"{idx}.pt"
            torch.save(data, path)

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        key = self._make_key(idx)

        if self.backend == "lmdb":
            while True:
                try:
                    with self.env.begin(write=False) as txn:
                        value = txn.get(key)
                    break
                except lmdb.MapResizedError:
                    self._sync_mapsize()
            if value is None:
                return None
            return self._deserialize(value)
        else:
            path = self.cache_path / f"{idx}.pt"
            if not path.exists():
                return None
            return torch.load(path, map_location="cpu")

    def __contains__(self, idx: int) -> bool:
        key = self._make_key(idx)
        if self.backend == "lmdb":
            while True:
                try:
                    with self.env.begin(write=False) as txn:
                        return txn.get(key) is not None
                except lmdb.MapResizedError:
                    self._sync_mapsize()
        else:
            return (self.cache_path / f"{idx}.pt").exists()

    def close(self) -> None:
        if self.backend == "lmdb" and self.env is not None:
            self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _make_key(idx: int) -> bytes:
        return f"{idx:010d}".encode()

    @staticmethod
    def _serialize(data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        torch.save(data, buf)
        return buf.getvalue()

    @staticmethod
    def _deserialize(value: bytes) -> Dict[str, Any]:
        buf = io.BytesIO(value)
        return torch.load(buf, map_location="cpu")


class DictFeatureCache:
    """In-memory feature cache for testing (no disk I/O)."""

    def __init__(self):
        self._store: Dict[int, Any] = {}

    def put(self, idx: int, data: Any) -> None:
        self._store[idx] = data

    def get(self, idx: int) -> Optional[Any]:
        return self._store.get(idx)

    def __contains__(self, idx: int) -> bool:
        return idx in self._store

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
