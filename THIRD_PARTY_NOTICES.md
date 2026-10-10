# Third-party code

The following retained components remain subject to their upstream licenses:

- `ddn/vendors/depth_anything_v2`: Depth Anything V2 / DINOv2 encoder implementation.
- `ddn/vendors/idisc`: iDisc head implementation; license retained in that directory.
- `third_party/LocalMamba` and `third_party/VMamba`: official operator implementations.
- `third_party/TransNASBench`: benchmark API and models.
- `third_party/depthsplat`: DepthSplat with the GeoBridge monocular-branch adapter.
- `cos3d`: COS3D / Gaussian Splatting derived scene training and rendering code.
- CUDA extensions and Segment Anything under `cos3d/submodules` and
  `cos3d/third_party/gaussian-splatting/submodules` retain their own license files.

The Gaussian Splatting/COS3D license includes non-commercial research and
evaluation terms. Component licenses are not replaced by a repository-wide license.
