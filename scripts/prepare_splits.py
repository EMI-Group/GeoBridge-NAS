"""
Generate train/val/test split files for TartanAir dataset.

Usage:
    python scripts/prepare_splits.py --root /path/to/tartanair

Outputs (written to <root>/):
    train_split.txt
    val_split.txt
    test_split.txt

Each line: <relative_img_path> <relative_depth_path>

Split strategy (environment-level, reproducible):
  - test  : abandonedfactory, neighborhood, office2
  - val   : carwelding, japanesealley
  - train : everything else
"""

import argparse
from pathlib import Path

# Fixed environment-level split — consistent with common TartanAir benchmarks
TEST_ENVS = {"abandonedfactory", "neighborhood", "office2"}
VAL_ENVS  = {"carwelding", "japanesealley"}


def collect(root: Path, env_names: set | None = None) -> list[tuple[str, str]]:
    """Walk root and return (img_rel, depth_rel) pairs, filtered by env_names if given."""
    samples = []
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        if env_names is not None and env_dir.name.lower() not in env_names:
            continue
        for diff_dir in sorted(env_dir.iterdir()):
            if not diff_dir.is_dir():
                continue
            for traj_dir in sorted(diff_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                img_dir = traj_dir / "image_left"
                dep_dir = traj_dir / "depth_left"
                if not img_dir.exists() or not dep_dir.exists():
                    continue
                for img_file in sorted(img_dir.glob("*.png")):
                    dep_file = dep_dir / f"{img_file.stem}_depth.npy"
                    if dep_file.exists():
                        samples.append((
                            str(img_file.relative_to(root)),
                            str(dep_file.relative_to(root)),
                        ))
    return samples


def write_split(path: Path, samples: list[tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for img, dep in samples:
            f.write(f"{img} {dep}\n")
    print(f"  {path.name}: {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="TartanAir root directory")
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"

    all_envs = {d.name.lower() for d in root.iterdir() if d.is_dir()}
    train_envs = all_envs - TEST_ENVS - VAL_ENVS

    print(f"Environments found : {sorted(all_envs)}")
    print(f"  train ({len(train_envs)}): {sorted(train_envs)}")
    print(f"  val   ({len(VAL_ENVS)}) : {sorted(VAL_ENVS & all_envs)}")
    print(f"  test  ({len(TEST_ENVS)}): {sorted(TEST_ENVS & all_envs)}")
    print()

    write_split(root / "train_split.txt", collect(root, train_envs))
    write_split(root / "val_split.txt",   collect(root, VAL_ENVS))
    write_split(root / "test_split.txt",  collect(root, TEST_ENVS))

    print("\nDone. Split files written to", root)


if __name__ == "__main__":
    main()
