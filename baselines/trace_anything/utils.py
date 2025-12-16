import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from typing import Dict, List, Tuple
from o3d_renderer import run_open3d_viewer
from o3d_renderer import run_open3d_offline_renderer


def view_with_open3d_viewer(frames: list[dict]):
    
    nr_frames = len(frames)
    # print("nr_frames:", nr_frames)
    
    rgbs = None
    depths = None
    point_clouds = None
    K = None
    poses_c2w = None
    tracks3d = None
    instances_masks = None
    
    scale = 50.0  # Scale factor for better visibility
    
    first_frame = frames[0]
    H, W = first_frame['H'], first_frame['W']
    
    rgbs = []
    instances_masks = []
    for frame in frames:
        
        # width, height
        width, height = frame['W'], frame['H']
        
        # get RGB
        rgb = frame['img_rgb_uint8']  # (H, W, 3) uint8
        # print("RGB Image Shape:", rgb.shape)
        rgbs.append(rgb)
        
        # get motion mask as instance mask
        fg_mask_flat = frame['fg_mask_flat']  # (H*W,) bool tensor
        # convert to numpy
        fg_mask_flat = fg_mask_flat.cpu().numpy()
        fg_mask = fg_mask_flat.reshape((height, width)).astype(np.uint8)
        # print("Foreground Mask Shape:", fg_mask.shape)
        instances_masks.append(fg_mask)
        
    # concatenate inputs
    rgbs = np.stack(rgbs, axis=0)  # (T, H, W, 3) uint8
    instances_masks = np.stack(instances_masks, axis=0)  # (T, H, W) uint8
    
    # get background point cloud 
    bg_pts = first_frame['bg_pts']  # (N_bg, 3) float32 (constant over time)
    img_rgb = first_frame['img_rgb_float']
    bg_mask = first_frame['bg_mask_flat']
    bg_colors = img_rgb[bg_mask][:bg_pts.shape[0]]
    
    point_clouds = []
    for fid, frame in enumerate(frames):
        
        # get point cloud
        pts_fg = frame['pts_fg_per_t'][fid]  # list of (N_t, 3) float32
        fg_mask = frame['fg_mask_flat']
        img_rgb = frame['img_rgb_float']
        fg_colors = img_rgb[fg_mask][:pts_fg.shape[0]]
        
        # concatenate fg and bg points and colors
        all_pts = np.concatenate([pts_fg, bg_pts], axis=0) * scale
        all_colors = np.concatenate([fg_colors, bg_colors], axis=0)
        all_segm = np.concatenate([np.ones(pts_fg.shape[0], dtype=np.uint8), 
                                   np.zeros(bg_pts.shape[0], dtype=np.uint8)], axis=0)
        
        # print(f"{fid} pointcloud:", all_pts.shape, all_colors.shape)
        pcd = {'xyz': all_pts, 'rgb': all_colors, 'inst_id': all_segm}
        point_clouds.append(pcd)
        
    # Build trajectories across timesteps
    N = first_frame['pts_fg_per_t'][0].shape[0]
    tracks3d = np.full((N, nr_frames, 3), np.nan, dtype=np.float32)
    for t_idx in range(nr_frames):
        pts_t = first_frame['pts_fg_per_t'][t_idx]  # (N_t, 3)
        tracks3d[:, t_idx, :] = pts_t * scale
    
    run_open3d_viewer(
        nr_frames,
        rgbs=rgbs,
        depths=depths,
        point_clouds=point_clouds,
        K=K,
        poses_c2w=poses_c2w,
        tracks3d=tracks3d,
        instances_masks=instances_masks,
    )

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
    ds: int,  # downsample factor
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


def load_output(filepath):
    # preds[i]['ctrl_pts3d'] — 3D control points, shape [K, H, W, 3]
    # preds[i]['ctrl_conf'] — confidence maps, shape [K, H, W]
    # preds[i]['fg_mask'] — binary mask [H, W], computed via Otsu thresholding on control-point variance.
    # preds[i]['time'] — predicted scalar time ∈ [0, 1).
    # views[i]['img'] — normalized input image tensor ∈ [-1, 1]

    output = torch.load(filepath, map_location=torch.device('cpu'), weights_only=True)
    # print(f"Loaded output from {filepath}")
    # preds: list = output["preds"]
    views: list = output["views"]
    
    nr_frames = len(views)
    
    # t_steps evenly spaced in [0, 1), one for each frame
    t_step = 1 / (nr_frames - 1)
    ds = 1  # no downsampling
    t_vals, frames, fg_conf_all_t, bg_conf_all_flat, times = build_precomputes(output, t_step, ds)
    
    return frames