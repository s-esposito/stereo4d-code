import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from typing import Dict, List, Tuple

# ======== B-spline  ========
PRECOMPUTED_KNOTS = {
    4:  torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    7:  torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    10: torch.tensor([0.0, 0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
}

def _precompute_knot_differences(n_ctrl_pts, degree, knots):
    denom1 = torch.zeros(n_ctrl_pts, degree + 1, device=knots.device)
    denom2 = torch.zeros(n_ctrl_pts, degree + 1, device=knots.device)
    for k in range(degree + 1):
        for i in range(n_ctrl_pts):
            denom1[i, k] = knots[i + k] - knots[i] if i + k < len(knots) else 0.0
            denom2[i, k] = knots[i + k + 1] - knots[i + 1] if i + k + 1 < len(knots) else 1.0
    return denom1, denom2

PRECOMPUTED_DENOMS = {n: _precompute_knot_differences(n, 3, PRECOMPUTED_KNOTS[n]) for n in [4, 7, 10]}

def evaluate_bspline_conf(ctrl_pts3d, ctrl_conf, t_values):
    """ctrl_pts3d:[N_ctrl,H,W,3], ctrl_conf:[N_ctrl,H,W], t_values:[T] -> (T,H,W,3),(T,H,W)"""
    n_ctrl_pts, H, W, _ = ctrl_pts3d.shape
    assert n_ctrl_pts in (4, 7, 10), f"unsupported n_ctrl_pts={n_ctrl_pts}"
    degree = 3
    knot_vector = PRECOMPUTED_KNOTS[n_ctrl_pts].to(ctrl_pts3d.device)
    denom1, denom2 = [d.to(ctrl_pts3d.device) for d in PRECOMPUTED_DENOMS[n_ctrl_pts]]
    ctrl_pts3d = ctrl_pts3d.permute(0, 3, 1, 2)         # [N,3,H,W]
    ctrl_conf  = ctrl_conf.unsqueeze(-1).permute(0, 3, 1, 2)  # [N,1,H,W]
    basis = _compute_bspline_basis(n_ctrl_pts, degree, t_values, knot_vector, denom1, denom2)  # [T,N]
    basis = basis.view(-1, n_ctrl_pts, 1, 1, 1)               # [T,N,1,1,1]
    pts3d_t = torch.sum(basis * ctrl_pts3d.unsqueeze(0), dim=1).permute(0, 2, 3, 1)  # [T,H,W,3]
    conf_t  = torch.sum(basis * ctrl_conf.unsqueeze(0),  dim=1).squeeze(1)           # [T,H,W]
    return pts3d_t, conf_t

def _compute_bspline_basis(n_ctrl_pts, degree, t_values, knots, denom1, denom2):
    N = t_values.size(0)
    basis = torch.zeros(N, n_ctrl_pts, degree + 1, device=t_values.device)
    t = t_values
    basis_k0 = torch.zeros(N, n_ctrl_pts, device=t.device)
    for i in range(n_ctrl_pts):
        if i == n_ctrl_pts - 1:
            basis_k0[:, i] = ((knots[i] <= t) & (t <= knots[i + 1])).float()
        else:
            basis_k0[:, i] = ((knots[i] <= t) & (t < knots[i + 1])).float()
    basis[:, :, 0] = basis_k0
    for k in range(1, degree + 1):
        basis_k = torch.zeros(N, n_ctrl_pts, device=t.device)
        for i in range(n_ctrl_pts):
            term1 = ((t - knots[i]) / denom1[i, k]) * basis[:, i, k-1] if denom1[i, k] > 0 else 0.0
            term2 = ((knots[i + k + 1] - t) / denom2[i, k]) * basis[:, i + 1, k-1] if (denom2[i, k] > 0 and i + 1 < n_ctrl_pts) else 0.0
            basis_k[:, i] = term1 + term2
        basis[:, :, k] = basis_k
    return basis[:, :, degree]

# -------------- precompute tensors for viewer -------------

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def as_float(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0])

def build_precomputes(
    output: Dict,
    t_step: float,
    ds: int,
) -> Tuple[np.ndarray, List[Dict], List[np.ndarray], np.ndarray, np.ndarray]:
    preds = output["preds"]
    views = output["views"]
    n = len(preds)
    assert n == len(views)

    root = output.get("_root_dir", os.getcwd())

    # timeline
    t_vals = np.arange(0.0, 1.0 + 1e-6, t_step, dtype=np.float32)
    if t_vals[-1] >= 1.0:
        t_vals[-1] = 0.99
    T = len(t_vals)
    t_tensor = torch.from_numpy(t_vals)

    frames: List[Dict] = []
    fg_conf_pool_per_t: List[List[np.ndarray]] = [[] for _ in range(T)]
    bg_conf_pool: List[np.ndarray] = []

    stride = slice(None, None, ds)  # ::ds

    for i in range(n):
        pred = preds[i]
        view = views[i]

        # image ([-1,1]) -> uint8 RGB
        img = to_numpy(view["img"].squeeze().permute(1, 2, 0))
        img_uint8 = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        img_uint8 = img_uint8[::ds, ::ds]  # downsample for saving/vis
        H, W = img_uint8.shape[:2]
        HW = H * W
        img_flat = (img_uint8.astype(np.float32) / 255.0).reshape(HW, 3)

        # load FG mask from output dict
        fg_mask = pred['fg_mask']
        # fg_mask = _load_fg_mask_for_index(root, i, pred)

        # match resolution to downsampled view
        # if mask at full-res and we downsampled, stride it; else resize with nearest
        if fg_mask.shape == (H * ds, W * ds) and ds > 1:
            fg_mask = fg_mask[::ds, ::ds]
        elif fg_mask.shape != (H, W):
            fg_mask = cv2.resize(
                (fg_mask.astype(np.uint8) * 255),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        bg_mask = ~fg_mask
        bg_mask_flat = bg_mask.reshape(-1)
        fg_mask_flat = fg_mask.reshape(-1)

        # control points/conf (K,H,W,[3]) at downsampled stride
        ctrl_pts3d = pred["ctrl_pts3d"][:, stride, stride, :]    # [K,H,W,3]
        ctrl_conf  = pred["ctrl_conf"][:, stride, stride]         # [K,H,W]

        # evaluate curve over T timesteps
        pts3d_t, conf_t = evaluate_bspline_conf(ctrl_pts3d, ctrl_conf, t_tensor)  # [T,H,W,3], [T,H,W]
        pts3d_t = to_numpy(pts3d_t).reshape(T, HW, 3)
        conf_t  = to_numpy(conf_t).reshape(T, HW)

        # FG per t (keep per-t list for later filtering)
        pts_fg_per_t  = [pts3d_t[t][fg_mask_flat] for t in range(T)]
        conf_fg_per_t = [conf_t[t][fg_mask_flat]  for t in range(T)]
        for t in range(T):
            if pts_fg_per_t[t].size > 0:
                fg_conf_pool_per_t[t].append(conf_fg_per_t[t])

        # BG static
        bg_pts = pts3d_t.mean(axis=0)[bg_mask_flat]
        bg_conf_mean = conf_t.mean(axis=0)[bg_mask_flat]
        bg_conf_pool.append(bg_conf_mean)

        frames.append(dict(
            img_rgb_uint8=img_uint8,
            img_rgb_float=img_flat,
            H=H, W=W, HW=HW,
            bg_mask_flat=bg_mask_flat,
            fg_mask_flat=fg_mask_flat,
            pts_fg_per_t=pts_fg_per_t,
            conf_fg_per_t=conf_fg_per_t,
            bg_pts=bg_pts,
            bg_conf_mean=bg_conf_mean,
        ))

    # pools for percentiles
    fg_conf_all_t: List[np.ndarray] = []
    for t in range(T):
        if len(fg_conf_pool_per_t[t]) == 0:
            fg_conf_all_t.append(np.empty((0,), dtype=np.float32))
        else:
            fg_conf_all_t.append(np.concatenate(fg_conf_pool_per_t[t], axis=0).astype(np.float32))

    if len(bg_conf_pool):
        bg_conf_all_flat = np.concatenate(bg_conf_pool, axis=0).astype(np.float32)
    else:
        bg_conf_all_flat = np.empty((0,), dtype=np.float32)

    # frame times (fallback to views' time_step if missing)
    def _get_time(i):
        ti = preds[i].get("time", None)
        if ti is None:
            ti = views[i].get("time_step", float(i / max(1, n - 1)))
        return as_float(ti)
    times = np.array([_get_time(i) for i in range(n)], dtype=np.float64)

    return t_vals, frames, fg_conf_all_t, bg_conf_all_flat, times

def choose_nearest_frame_indices(frame_times: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(frame_times - tv))) for tv in t_vals], dtype=np.int64)