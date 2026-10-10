# COS3D transfer

The public weights contain three methods (`a1`, `a2`, `a3`) for four scenes
(`figurines`, `teatime`, `ramen`, `waldo_kitchen`). Each scene directory contains
the complete trained semantic Gaussian checkpoint `chkpnt30000.pth`. The point
cloud, RGB model and grounding tensors used in the paper are regenerated locally
from this checkpoint with the commands below; no semantic retraining is required.

## Environment

Install DDN's core dependencies, then install the scene dependencies in a separate
environment from DepthSplat's modified rasterizer:

```bash
python -m pip install -r cos3d/requirements.txt
python -m pip install cos3d/submodules/segment-anything
python -m pip install cos3d/submodules/simple-knn --no-build-isolation
```

Scene data must retain the COLMAP cameras, images and COS3D/LangSplat preprocessing
layout. Text grounding uses the OpenCLIP ViT-B-16 (`laion2b_s34b_b88k`) model checkpoint. SAM preprocessing
uses the ViT-H checkpoint. These third-party pretrained models are obtained
separately; the released scene weights do not require running semantic training again.

To generate the image language features from a COLMAP scene, run from `cos3d/`:

```bash
python scripts/preprocess_langsplat_parallel.py \
  --dataset_path /path/to/lerf/figurines \
  --sam_ckpt_path /path/to/sam_vit_h_4b8939.pth \
  --clip_ckpt_path /path/to/open_clip_model.safetensors \
  --output_name language_features
```

Repeat for each scene. `language_features` is the directory read by
`--foundation_model sam_clip` below.

## Ground, render and evaluate

Run from `cos3d/`. `WEIGHTS` is the public weight root; `RUN_ROOT` is a writable
output directory; `LERF_ROOT` contains the four prepared scene directories, and
`LABEL_ROOT` contains the official annotations. First re-export the scene fields
from the released checkpoints, then ground and render against the re-exported
runs. The RGB model referenced below is produced by `scripts/refine_gaussians.py`
(see Retraining and RGB refinement) under `"$RUN_ROOT/$METHOD/$NAME/rgb"`.

```bash
cd cos3d
export WEIGHTS=/path/to/weights/public
export RUN_ROOT=/path/to/scene_runs
export LERF_ROOT=/path/to/lerf
export LABEL_ROOT=/path/to/lerf_ovs/label
export CLIP_CKPT=/path/to/open_clip_model.safetensors
METHOD=a3

for NAME in figurines teatime ramen waldo_kitchen; do
  SCENE="$RUN_ROOT/$METHOD/$NAME"
  python train_instance.py -s "$LERF_ROOT/$NAME" -m "$SCENE" \
    --start_checkpoint "$WEIGHTS/cos3d/$METHOD/$NAME/chkpnt30000.pth" \
    --iterations 30000 --save_iterations 30000
  python scripts/ground_official_scene.py \
    --exp-dir "$SCENE" --scene "$NAME" --clip-ckpt "$CLIP_CKPT"
  python render_3D_prompt.py \
    -s "$LERF_ROOT/$NAME" -m "$SCENE" \
    --original_model_path "$SCENE/rgb" --iteration 30000 \
    --foundation_model sam_clip --method_name Grouding_3D_mask --normalize_flag
done

python scripts/evaluate_official_wave.py \
  --method-root "$WEIGHTS/cos3d/$METHOD" --label-root "$LABEL_ROOT"
```

If annotations are still polygon JSON files, prepare the masks with
`scripts/prepare_official_lerf_gt.py` (`--help` describes the input directory).
Evaluation uses the official 56/59/71/22 object-frame pairs and an unweighted mean
across scenes. The paper's mAcc is Acc@0.25; Acc@0.5 is reported separately.

## Retraining and RGB refinement

`train_instance.py` trains the scene semantic field; use `--transfer_arch` to choose
the auxiliary DDN branch where required by the recorded protocol. The searched
initializer and the optional semantic-training branch are distinct components.

The original Gaussian Splatting trainer is retained in `third_party/gaussian-splatting`.
Build its CUDA rasterizer in the RGB environment, then initialize RGB refinement
from an exported PLY:

```bash
python scripts/refine_gaussians.py \
  --gs-root third_party/gaussian-splatting \
  --initial-ply /path/to/initial_gaussians.ply \
  -s /path/to/scene -m /path/to/rgb_run \
  --iterations 30000 --save_iterations 30000 --disable_viewer
```

`scripts/prepare_depthsplat_lerf_aligned.py` prepares camera-aligned feed-forward
inputs. `merge_depthsplat_gaussians_aligned.py` merges exported per-view Gaussians.
