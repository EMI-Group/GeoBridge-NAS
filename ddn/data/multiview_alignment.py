"""Multi-view pseudo-depth alignment in the RealEstate10K camera gauge."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ddn.data.re10k import Re10KFrameDataset
from src.geometry.projection import get_world_rays
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_cameras(cameras: torch.Tensor):
    """re10k cameras[N,18] -> normalized intrinsics K[N,3,3] and c2w extrinsics[N,4,4].

    Matches src/dataset/dataset_re10k.py:convert_poses. Intrinsics stay NORMALIZED
    (fx,fy,cx,cy as fractions of W,H), so use coordinates in [0,1] with them.
    """
    cameras = cameras.float()
    n = cameras.shape[0]
    fx, fy, cx, cy = cameras[:, 0], cameras[:, 1], cameras[:, 2], cameras[:, 3]
    K = torch.eye(3).repeat(n, 1, 1)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = fx, fy, cx, cy
    w2c = torch.eye(4).repeat(n, 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(n, 3, 4)
    c2w = w2c.inverse()
    return K, c2w


@torch.no_grad()
def da_depth(teacher, pil_imgs, size, device, da_batch=8):
    """Return per-frame metric-ish depth [N,H,W] = 1/disparity (DA relu output).

    Runs DA in small batches so a long clip at 518 doesn't OOM (DA-ViT-L
    activations are heavy, and the GPU may be shared)."""
    tf = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    frames = torch.stack([tf(im) for im in pil_imgs])
    out = []
    for s in range(0, frames.shape[0], da_batch):
        disp = teacher.predict_depth(frames[s:s + da_batch].to(device)).squeeze(1)
        out.append((1.0 / disp.clamp_min(1e-3)).cpu())
        del disp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def detect_sift(gray, nfeatures=4000):
    """Detect SIFT once for a frame -> (keypoint xy ndarray [K,2], descriptors).

    Detecting per FRAME (and caching) instead of per pair is the main speed win:
    each frame is otherwise re-detected ~2*len(pair_strides) times."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return np.empty((0, 2), np.float32), None
    xy = np.array([k.pt for k in kp], dtype=np.float32)
    return xy, des


def match_pair(xyA, desA, xyB, desB, ratio, max_matches):
    """Match two frames from CACHED SIFT (keypoints/descriptors). Returns matched
    pixel coords in the detection-image space. Uses FLANN (KD-tree) which is far
    faster than brute-force L2 for the 128-d SIFT descriptors."""
    if desA is None or desB is None:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = _FLANN.knnMatch(desA, desB, k=2)
    ia, ib = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            ia.append(m.queryIdx); ib.append(m.trainIdx)
    if not ia:
        return np.empty((0, 2)), np.empty((0, 2))
    ptsA, ptsB = xyA[ia], xyB[ib]
    if len(ptsA) >= 8:
        _, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is not None:
            keep = mask.ravel().astype(bool)
            ptsA, ptsB = ptsA[keep], ptsB[keep]
    if len(ptsA) > max_matches:
        sel = np.random.choice(len(ptsA), max_matches, replace=False)
        ptsA, ptsB = ptsA[sel], ptsB[sel]
    return ptsA, ptsB


def bilinear_sample_depth(depth_hw, pts_xy):
    """Sample depth [H,W] at float pixel coords pts_xy (M,2) -> (M,) via grid_sample."""
    H, W = depth_hw.shape
    d = depth_hw[None, None]                                  # [1,1,H,W]
    gx = pts_xy[:, 0] / (W - 1) * 2 - 1
    gy = pts_xy[:, 1] / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None, None]          # [1,1,M,2]
    out = torch.nn.functional.grid_sample(d, grid, align_corners=True, mode="bilinear")
    return out.reshape(-1)


def build_terms(pts_xy, depth_hw, K, c2w, size):
    """For pixels pts_xy in one frame: world ray origin o (M,3), z-normalized
    world ray direction dir (M,3), and sampled DA depth D (M,). World point of a
    pixel at metric depth t is  o + t * dir."""
    d = bilinear_sample_depth(depth_hw, pts_xy)               # (M,) DA depth
    coords_norm = pts_xy.clone()
    coords_norm[:, 0] /= size
    coords_norm[:, 1] /= size                                 # -> [0,1] to match normalized K
    o, dirs = get_world_rays(coords_norm, c2w[None].expand(pts_xy.shape[0], 4, 4),
                             K[None].expand(pts_xy.shape[0], 3, 3))
    return o, dirs, d


def triangulate_midpoint(o1, d1, o2, d2):
    """Midpoint two-view triangulation for batched rays.

    Solves min_t ||o1 + t1 d1 - (o2 + t2 d2)||^2 per correspondence.
    Returns ray depths t1,t2 (M,) and triangulation angle in degrees (M,)."""
    # 2x2 normal equations per correspondence: [[d1.d1, -d1.d2],[-d1.d2, d2.d2]] t = [d1.b, -d2.b]
    b = o2 - o1
    a11 = (d1 * d1).sum(-1)
    a22 = (d2 * d2).sum(-1)
    a12 = -(d1 * d2).sum(-1)
    r1 = (d1 * b).sum(-1)
    r2 = -(d2 * b).sum(-1)
    det = (a11 * a22 - a12 * a12).clamp_min(1e-12)
    t1 = (a22 * r1 - a12 * r2) / det
    t2 = (a11 * r2 - a12 * r1) / det
    cos = ((d1 * d2).sum(-1) / (d1.norm(dim=-1) * d2.norm(dim=-1)).clamp_min(1e-9)).clamp(-1, 1)
    angle = cos.arccos().rad2deg()
    return t1, t2, angle


def solve_scene(scene, args, teacher, device, verbose=True):
    """Run the full multi-view scale alignment on ONE scene.

    Returns a metrics dict; on degeneracy (no texture / no parallax) returns
    {"status": "degenerate", ...} instead of raising, so dataset preparation can skip the scene.
    On success also returns the per-frame fit + the data needed to build clouds.
    """
    key = scene.get("key", "scene")
    n_total = len(scene["images"])
    # Spread frames evenly across the whole clip: re10k is video, so consecutive
    # frames have ~zero baseline; frames far apart in time give real parallax.
    if args.spread:
        fidx = torch.linspace(0, n_total - 1, min(args.max_frames, n_total)).round().long().unique().tolist()
    else:
        fidx = list(range(min(args.max_frames, n_total)))
    n = len(fidx)

    pil_imgs = [Re10KFrameDataset._decode_image(scene["images"][i]) for i in fidx]
    cams = scene["cameras"]
    cams = cams if torch.is_tensor(cams) else torch.as_tensor(cams)
    K, c2w = parse_cameras(cams[fidx])

    depth = da_depth(teacher, pil_imgs, args.da_size, device, args.da_batch)   # [n,S,S]
    S = args.da_size
    rgbs = [np.array(im.resize((S, S))) for im in pil_imgs]

    # SIFT on smaller images (match_size) for speed; detect ONCE per frame and
    # cache, in parallel threads (cv2 releases the GIL). Coords scale back to S.
    ms = args.match_size
    grays = [cv2.cvtColor(np.array(im.resize((ms, ms))), cv2.COLOR_RGB2GRAY) for im in pil_imgs]
    with ThreadPoolExecutor(max_workers=args.match_workers) as ex:
        feats = list(ex.map(lambda g: detect_sift(g, args.sift_nfeatures), grays))
    coord_scale = float(S) / float(ms)

    rows_i, rows_j = [], []
    O_i, DIR_i, D_i, O_j, DIR_j, D_j = [], [], [], [], [], []
    T_i, T_j, ANG = [], [], []
    for i in range(n):
        for s in args.pair_strides:
            j = i + s
            if j >= n:
                continue
            pA, pB = match_pair(feats[i][0], feats[i][1], feats[j][0], feats[j][1],
                                args.ratio, args.matches_per_pair)
            if len(pA) == 0:
                continue
            pA = torch.from_numpy(pA).float() * coord_scale
            pB = torch.from_numpy(pB).float() * coord_scale
            oi, diri, Di = build_terms(pA, depth[i], K[i], c2w[i], S)
            oj, dirj, Dj = build_terms(pB, depth[j], K[j], c2w[j], S)
            t1, t2, ang = triangulate_midpoint(oi, diri, oj, dirj)
            rows_i.append(torch.full((len(pA),), i)); rows_j.append(torch.full((len(pB),), j))
            O_i.append(oi); DIR_i.append(diri); D_i.append(Di)
            O_j.append(oj); DIR_j.append(dirj); D_j.append(Dj)
            T_i.append(t1); T_j.append(t2); ANG.append(ang)
    if not rows_i:
        return {"status": "degenerate", "key": key, "reason": "no correspondences", "n": n}
    rows_i = torch.cat(rows_i); rows_j = torch.cat(rows_j)
    O_i = torch.cat(O_i); DIR_i = torch.cat(DIR_i); D_i = torch.cat(D_i)
    O_j = torch.cat(O_j); DIR_j = torch.cat(DIR_j); D_j = torch.cat(D_j)
    T_i = torch.cat(T_i); T_j = torch.cat(T_j); ANG = torch.cat(ANG)
    n_corr = rows_i.shape[0]

    good = (ANG >= args.min_tri_angle) & (T_i > 0) & (T_j > 0)
    median_angle = float(ANG.median())
    if int(good.sum()) < n:
        return {"status": "degenerate", "key": key, "reason": "too few parallax pts",
                "n": n, "n_corr": n_corr, "n_good": int(good.sum()), "median_angle": median_angle}

    def corr_error(a, b):
        Xi = O_i + (a[rows_i] * D_i + b[rows_i])[:, None] * DIR_i
        Xj = O_j + (a[rows_j] * D_j + b[rows_j])[:, None] * DIR_j
        return (Xi - Xj).norm(dim=1)

    a1 = torch.ones(n); b0 = torch.zeros(n)
    r_before = corr_error(a1, b0)[good]

    a = torch.ones(n); bsh = torch.zeros(n); coverage = torch.zeros(n, dtype=torch.long)
    for f in range(n):
        mi = good & (rows_i == f); mj = good & (rows_j == f)
        Ds = torch.cat([D_i[mi], D_j[mj]]); Ts = torch.cat([T_i[mi], T_j[mj]])
        coverage[f] = Ds.numel()
        if Ds.numel() < 5:
            continue
        if args.affine:
            af, bf = 1.0, 0.0
            for _ in range(10):
                res = (af * Ds + bf - Ts).abs().clamp_min(1e-9)
                w = torch.where(res <= res.median(), torch.ones_like(res), res.median() / res)
                Sw = w.sum(); Dw = (w * Ds).sum(); Tw = (w * Ts).sum()
                Ddw = (w * Ds * Ds).sum(); DTw = (w * Ds * Ts).sum()
                det = (Sw * Ddw - Dw * Dw).clamp_min(1e-12)
                af = (Sw * DTw - Dw * Tw) / det; bf = (Ddw * Tw - Dw * DTw) / det
            a[f] = float(af); bsh[f] = float(bf)
        else:
            a[f] = (Ts / Ds.clamp_min(1e-6)).median()
    covered = coverage >= 5
    if covered.any():
        a[~covered] = a[covered].median(); bsh[~covered] = 0.0
    a = a.clamp_min(1e-9)

    r_after = corr_error(a, bsh)[good]
    eb, ea = float(r_before.median()), float(r_after.median())
    improvement = 100 * (1 - ea / max(eb, 1e-9))
    scale_ratio = float(a.max() / a.min())

    # Confidence gate: accept alignment only if it measurably helps AND scales are
    # plausible. Otherwise fall back to identity (keep DA) so we never make a scene
    # worse than DA — this is what makes a full-dataset rollout net-positive.
    accept = (improvement >= 100 * args.min_gain) and (scale_ratio <= args.max_scale_ratio)
    if accept:
        a_final, b_final = a, bsh
    else:
        a_final, b_final = torch.ones(n), torch.zeros(n)

    metrics = {
        "status": "ok", "key": key, "n": n, "n_corr": n_corr, "n_good": int(good.sum()),
        "median_angle": median_angle, "scale_ratio": scale_ratio,
        "err_before": eb, "err_after": ea, "improvement": improvement,
        "accept": bool(accept), "coverage_ok": int(covered.sum()),
        # artifacts for cloud building (single-scene mode); a/bsh are the FINAL
        # (gated) values so clouds reflect what a rollout would actually store.
        "a": a_final, "bsh": b_final, "depth": depth, "K": K, "c2w": c2w,
        "rgbs": rgbs, "S": S, "fidx": fidx,
    }
    if verbose:
        print(f"[match] {n_corr} corr; {int(good.sum())} pass angle>={args.min_tri_angle}deg "
              f"(median tri-angle {median_angle:.2f}deg)")
        print(f"[coverage] frames with >=5 pts: {int(covered.sum())}/{n} (min {int(coverage.min())}/frame)")
        print(f"[before a=1] median 3D corr error {eb:.4f}")
        print(f"[after fit]  median 3D corr error {ea:.4f}")
        print(f"[scales] ratio(max/min)={scale_ratio:.2f}")
        print(f"[improvement] {eb:.4f} -> {ea:.4f} ({improvement:.1f}% lower)")
        print(f"[gate] accept={accept}  (min_gain={100*args.min_gain:.0f}% ratio_cap={args.max_scale_ratio}) "
              f"-> {'use aligned depth' if accept else 'KEEP DA (alignment rejected)'}")
    return metrics

