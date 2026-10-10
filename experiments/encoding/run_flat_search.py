#!/usr/bin/env python3
"""Launch the ordinary fixed-length vector baseline without changing project code."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from flat_full_vector import FlatFullVectorArch, FlatFullVectorOperator, random_valid_flat
import ddn.search.nsga2 as nsga2_module

nsga2_module.DecoderTree = FlatFullVectorArch
nsga2_module.TreeOperator = FlatFullVectorOperator
nsga2_module.random_valid_tree = random_valid_flat
runpy.run_path(str(ROOT / "scripts" / "search.py"), run_name="__main__")
