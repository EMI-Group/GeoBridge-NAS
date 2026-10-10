# Media sources and credits

## Current comparison baselines

The active indoor comparison uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (NeurIPS 2024), official Metric Hypersim Small checkpoint. Six predefined RGB frames are extracted from the existing videos; the matching GeoBridgeNet visualization is reused without GeoBridgeNet inference. Baseline inference is FP32 with input size 518 and the official 20 m output range. Visualization uses the same fixed TURBO 0–10 m scale as the existing GeoBridgeNet clips. Predictions above 10 m saturate only for display. See `comparisons/depth-dav2/manifest.json` for provenance. The older Depth Pro experiment below is archived and no longer shown on the page.

`scenes3d/living-room-mvsplat.js` and `scenes3d/garden-mvsplat.js` are actual [MVSplat](https://github.com/donydchen/mvsplat) (ECCV 2024) predictions from its official full Re10K checkpoint. The model receives the two released 256×256 JPEG input views for each scene. Camera-to-world poses are reconstructed from the saved viewer position/target/up, with centered square-image intrinsics derived from its FOV. Original full-resolution inputs and calibration were not included in the release; this is a qualitative comparison based on the released assets. All 131,072 predicted Gaussians per scene are retained. Both methods use identical compact DC-only viewer conventions, the same rendering settings, and linked cameras starting midway between the two saved input cameras. No scene fine-tuning, geometry filtering or visual enhancement was applied. See `comparisons/nvs/manifest.json` for source commit, checkpoint/input hashes and camera matrices. ETH3D attribution and **CC BY-NC-SA 4.0** terms below apply to these derived scene previews.

The added OpenGaussian, Dr. Splat and InstanceGaussian semantic table rows are source-reported values from the [COS3D paper's LeRF comparison](https://papers.neurips.cc/paper_files/paper/2025/file/10a6bdcabbd5a3d36b760daa295f63c1-Paper-Conference.pdf). They are not new local measurements or predictions of the displayed semantic scenes.

## Archived Depth Pro comparisons

`comparisons/depth/` contains six paired frames (0.5, 2.5 and 4.5 seconds from each indoor depth demo). RGB and GeoBridgeNet visualization are extracted from the left and right halves of the same existing video frame. GeoBridgeNet is reused, not re-inferred. Apple [Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro) (ICLR 2025), official source commit `9e65e4dbe9568d23c546fcec53302b10445e109e`, runs on the decoded RGB in FP32 on an RTX 4090, with its own focal-length estimate. Its raw metric depths are retained as NPY files and displayed with the existing fixed 0–10 m TURBO scale. Source footage credits below also apply to these derivatives. Video compression affects the extracted input and GeoBridgeNet visualization. No ground truth or new benchmark scores are available, and these examples do not establish a model ranking. See the included `manifest.json` for hashes and settings, and the upstream repository for Apple's model license.

## Continuous depth demos

- `demos/depth-living_room.{mp4,jpg}`: input footage by **Curtis Adams**, [Tracking Shot of a Living Room](https://www.pexels.com/video/tracking-shot-of-a-living-room-3773487/), used under the [Pexels License](https://www.pexels.com/license/). Retrieved 2026-09-13. The license permits website use and modification; no creator endorsement is implied. It is not a Creative Commons license.
- `demos/depth-meeting_room.{mp4,jpg}`: original footage by **Rui Wang**, Hangzhou Normal University, from `勤12-429.mp4`.

Both demos use a continuous opening segment of approximately 5.6 seconds, center-cropped to 4:3 and resized to 640×480. Every displayed RGB frame has a corresponding GeoBridgeNet-A3 depth prediction from the public NYU checkpoint. Input and prediction are placed side by side; audio is omitted. Predictions use FP32 and horizontal-flip averaging. Colors use a fixed 0–10 metre scale across every frame and both scenes, without temporal smoothing. These are qualitative demos on new footage, not benchmark measurements or ground-truth depth.

The opening depth films use a moving divider to reveal the corresponding prediction over the continuous RGB footage:

- `demos/hero-depth-indoor.{mp4,jpg}` reuses Rui Wang's meeting-room footage and the same NYU GeoBridgeNet-A3 predictions described above.
- `demos/hero-depth-outdoor.{mp4,jpg}` uses consecutive color-camera frames 0000000000–0000000055 from **KITTI** sequence `2011_09_26_drive_0009_sync`, `image_02`, by **Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun**. Source: [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php), retrieved from its documented [public AWS mirror](https://registry.opendata.aws/kitti/). Source imagery and the derived outdoor film are licensed under **[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)**, as stated on the [official KITTI website](https://www.cvlibs.net/datasets/kitti/). Retain attribution and the same media license; use noncommercially. Reference: Geiger et al., *Vision meets Robotics: The KITTI Dataset*, IJRR 2013.

Each opening depth film is encoded at 30 fps and uses an 11.2-second reveal cycle. Its 5.6-second source segment loops twice at the original playback speed. The indoor film starts with depth on the left and reveals it toward the right; the outdoor film starts with depth on the right and reveals it toward the left. Both begin with an equal RGB/depth split, and the source images are not mirrored. The two depth previews display at 80% of their column width. The indoor source is 30 fps; KITTI's consecutive 10 Hz captured frames are each held for three output frames, without frame interpolation. Outdoor predictions use the public KITTI A3 checkpoint, the native 352 × 1216 bottom-center crop, FP32 and horizontal-flip averaging. Display crops RGB and its aligned prediction to 8:5, with a narrow feather into the white background. Fixed TURBO scales are 0–10 m indoors and 0–80 m outdoors; depth values are not temporally smoothed. This is qualitative demo inference, with no new benchmark evaluation.

## Continuous novel-view renders

`demos/nvs-0*` (Living room) and `demos/nvs-1*` (Botanical garden) use calibrated source images from [ETH3D](https://www.eth3d.net/datasets), by **Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger**.

**These source images and their derived demos are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**, as stated on the [official ETH3D homepage](https://www.eth3d.net/). Retain attribution, use noncommercially, and share adaptations under that license. This media license is separate from the repository's code licenses.

| Example | Two input images | Held-out reference |
|---|---|---|
| Living room | DSC_1535.JPG, DSC_1537.JPG | DSC_1536.JPG |
| Botanical garden | DSC_0872.JPG, DSC_0874.JPG | DSC_0873.JPG |

The GeoBridgeNet-A3 / DepthSplat pipeline uses the released RealEstate10K checkpoint, without scene fine-tuning. Inputs are center-cropped with adjusted intrinsics. Each 6-second, 30 fps clip contains 180 model-rendered frames along an interpolated camera trajectory, easing from one input camera to the other and back. Native inference/rendering resolution is 256×256. The videos are continuous **novel-view renders**, not captured source videos or crossfades between photographs. The two context thumbnails are source photographs. The optional comparison places a prediction at the held-out camera beside its captured reference.

Examples were selected for visual clarity from candidate demos, not to estimate dataset-average performance. No generative enhancement, frame interpolation, or synthetic detail was added after model rendering.

Reference: Schöps et al., *A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos*, CVPR 2017. [Official BibTeX](https://www.eth3d.net/data/schoeps2017cvpr.bib).

## Interactive 3D scenes

`scenes3d/living-room.js` and `scenes3d/garden.js` contain the actual 131,072 Gaussian predictions for each of the two ETH3D examples above. They use the same input views and public Re10K A3 checkpoint as the continuous novel-view renders. They inherit the ETH3D **CC BY-NC-SA 4.0** media terms and attribution above. These are static 3D scenes with an interactive camera, not time-varying 4D reconstructions.

`scenes3d/teatime.js` and `scenes3d/waldo_kitchen.js` are web previews of the authors' released GeoBridgeNet-A3-initialized COS3D semantic Gaussian models. Source scenes are from [LERF](https://www.lerf.io/), by **Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa and Matthew Tancik** (ICCV 2023). Segmentation uses the original saved `Grouding_3D_mask` predictions from the authors' experiments. The four controls select **text queries**, not live language-model inference. Selecting a query uses its corresponding reference camera (Teatime frames 00025, 00107 and 00129; Waldo Kitchen frame 00066). Each selected object's complete set of Gaussians is retained; the remaining background is uniformly sampled to 900,000 Gaussians per scene with a fixed seed. Scene models and experiment outputs are separate from the viewer's MIT-licensed code.

| Scene | Web Gaussians | Text query | Selected Gaussians retained |
|---|---:|---|---:|
| Teatime | 1,196,523 | sheep | 97,718 |
| Teatime | 1,196,523 | stuffed bear | 196,974 |
| Teatime | 1,196,523 | yellow pouf | 1,831 |
| Waldo Kitchen | 1,000,802 | pot | 100,802 |

For the static browser preview, RGB uses the DC spherical-harmonic coefficient; RGB, opacity and quaternion components are quantized to eight bits. Positions and Gaussian scales remain float32. The viewer uses the source camera poses and preserves the native image's field of view. The NVS display's screen-space Gaussian footprint scales with the source-to-display pixel ratio. Background sampling and this compact appearance representation make the browser preview different from the full model's benchmark renders. Highlight mode recolors the predicted foreground and desaturates the background; object-only mode hides the remaining Gaussians without changing the selected geometry.

The bundled renderer combines [GaussianSplats3D 0.4.7](https://github.com/mkkellogg/GaussianSplats3D) and [Three.js 0.160.1](https://github.com/mrdoob/three.js), both MIT-licensed. Their notices are retained in `vendor/LICENSE-gaussian-splats-3d.txt` and `vendor/LICENSE-three.txt`. All viewer scripts and scene previews are served locally by the project page, with no external CDN or model server.

## GeoBridgeNet-A3 image-space results and manuscript figures

`grounding/*-reference.jpg` and `grounding/*-mask.png` are reference views and GeoBridgeNet-A3 predictions from the authors' COS3D experiments on the LERF Teatime and Waldo Kitchen scenes. The webpage colors stored binary masks and overlays them on the corresponding RGB image. The four text queries remain available in a collapsible panel below the 3D viewers. `demos/grounding.jpg` shows the corresponding interactive results for the README.

## COS3D interactive semantic baseline

`scenes3d/teatime-cos3d.js` and `scenes3d/waldo_kitchen-cos3d.js` are local inference exports of the official [COS3D (NeurIPS 2025)](https://github.com/Runsong123/COS3D) 30,000-step checkpoints. Scene images and calibration come from the LERF-OVS release linked by [LangSplat](https://github.com/minghanqin/LangSplat), based on [LERF](https://www.lerf.io/). Underlying data and model rights remain with their respective authors; this project does not relicense them.

Language anchors omitted from the checkpoint release were reconstructed from all 177 Teatime and 187 Waldo Kitchen views using SAM ViT-H and OpenCLIP ViT-B-16. Local kernel grounding uses original COS3D thresholds and the same queries as the existing GeoBridgeNet scene assets. The page uses linked cameras, one renderer and identical display modes. It retains every queried foreground Gaussian and samples up to 900,000 background Gaussians per scene, matching the existing GeoBridgeNet previews. Checkpoint/asset hashes and implementation details are recorded in `comparisons/semantics/`. These are qualitative local results; the published numerical table has not been re-evaluated. Existing GeoBridgeNet predictions are reused unchanged. COS3D's upstream license notice is retained in `vendor/LICENSE-cos3d.txt`.

`readme/geobridge-overview.png` and `readme/geobridge-decoder.png` are rendered from the current manuscript figures `F1.pdf` and `F2.pdf`. `figure-0.png`, `figure-1.png`, and `figure-5.png`–`figure-8.png` are retained exports from earlier manuscript versions.

## Opening result films

`demos/hero-nvs.{mp4,jpg}` shows the same GeoBridgeNet-A3 Botanical Garden Gaussian reconstruction used by the interactive viewer. It inherits the ETH3D attribution and CC BY-NC-SA 4.0 terms above. `demos/hero-semantics.{mp4,jpg}` shows the GeoBridgeNet-A3-initialized COS3D Teatime scene and its original “sheep” prediction, with LERF attribution as above.

Both films are 10-second, 30 fps, 960 × 600 renders of static Gaussian scenes along continuous camera paths. They use the same compact scene representation as the interactive viewers. White backgrounds, soft spatial cropping and an edge vignette frame the scenes for the opening layout. The segmentation film additionally animates foreground highlighting, background opacity and camera framing; query membership, Gaussian positions, scales and rotations are unchanged. These display renders do not add detail to the 256 × 256 NVS inputs or model dynamic scene motion. No new training or benchmark evaluation was performed. Only GeoBridgeNet scene renders are included; the reference site's videos are not redistributed.

## README animations

The four looping GIFs in `readme/` are compact versions of the opening result films. They preserve the source videos' playback speed and continuous motion, including the 11.2-second depth reveal cycle and its opposite indoor/outdoor directions. GIFs use 400 × 250 pixels, 10 fps and a 128-color palette; the README links each preview to its original MP4.

| GIF | Source film | Source attribution / media terms |
|---|---|---|
| `readme/novel-view.gif` | `demos/hero-nvs.mp4` | Schöps et al. / ETH3D; CC BY-NC-SA 4.0 |
| `readme/semantic-grounding.gif` | `demos/hero-semantics.mp4` | Kerr et al. / LERF; GeoBridgeNet-A3-initialized COS3D scene, as described above |
| `readme/depth-indoor.gif` | `demos/hero-depth-indoor.mp4` | Original footage by Rui Wang |
| `readme/depth-outdoor.gif` | `demos/hero-depth-outdoor.mp4` | Geiger et al. / KITTI; CC BY-NC-SA 3.0 |

Source credits and media license terms above also apply to these derived previews. GIF palette conversion is only for README delivery; it does not change the underlying model predictions.

## Quantitative tables

The NYU, KITTI, RealEstate10K and LERF tables are selected main-experiment rows from the authors' GeoBridge manuscript (`TableNyu`, `TableKitti`, `TableReTenK`, `TableCosThreeD` and `TableCosThreeDVariants`). Values were transcribed directly; no new benchmark evaluation was run for the page redesign. RealEstate10K baseline metrics remain source-reported reference results. Shading identifies GeoBridgeNet rows, not the best score in every column.

## Layout

The project page follows the white content blocks, centered publication header, two rows of mouse/keyboard guidance, paired interactive windows and task-result sequence of [MonST3R](https://monst3r-project.github.io/). The interactive sections follow its Google Sans declaration with a Times New Roman / serif fallback, 17.5 px regular body text, a 16 px abstract, 25 px paragraph line height and 30 px section titles, matching the reference page’s rendered typography. The small mouse/key icons are original vector UI drawings. The README presentation follows [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror). HTML, CSS, JavaScript and the tree icon were written for GeoBridge.

## Publication header

The publication header follows [Shape of Motion](https://shape-of-motion.github.io/): centered Google Sans title at 40 px / 45 px and weight 600, 20 px author and affiliation rows, and Noto Sans resource buttons with a dark rounded background. Author names and institutional information are GeoBridge's own; no venue, equal-contribution designation or unavailable paper link is copied.

The two Latin webfonts are bundled locally under SIL OFL 1.1: [Google Sans](https://github.com/googlefonts/googlesans) and [Noto Sans](https://github.com/google/fonts/tree/main/ofl/notosans). Their license files are in `fonts/`. The same regular-face request as the reference is used, with browser-synthesized title weight 600. CSS font aliases scope the loaded fonts to the publication header.

The GitHub, building, PDF and download resource icons are from [Font Awesome Free 5.15.4](https://github.com/FortAwesome/Font-Awesome/tree/5.15.4), by Fonticons, Inc., licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are embedded as SVG paths in the header; `vendor/LICENSE-font-awesome.txt` retains the upstream license notice.

The method overview fits the section content width (1000 px within the 1100 px desktop card) and scales with the layout on smaller screens. The overview uses the current GeoBridge manuscript figure.
