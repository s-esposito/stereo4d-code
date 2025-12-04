import pickle
import numpy as np
from utils import load_video_frames
import matplotlib.pyplot as plt
import open3d as o3d
import os
import math
import time
import utils
import o3d_utils
from tqdm import tqdm

MIN_DEPTH = 0.1
MAX_DEPTH = 20.0


def load_rgbd_cam_from_pkl(root_dir: str, split:str, scene:str, timestamp:str, hfov: float):
    """load rgb, depth, and camera"""
    
    # prepare output dict
    input_dict = {
        'left': {
            'camera': [],
            'depth': [],
            'video': []
        }
    }
    
    # load test data from npz
    flow_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-flows_stereo.pkl"
    meta_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}.npz"
    video_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-right_rectified.mp4"
    disp_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-disps.npz"
    sam_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-sam3.npz"
        
    # Load video frames     
    rgbs, _ = load_video_frames(video_path)
    input_dict['left']['video'] = rgbs
    nfr = len(rgbs)
    
    #
    height, width = rgbs[0].shape[:2]
    print("video frames", len(rgbs), height, width)
    
    # Load sam data
    sam_data = None
    if os.path.exists(sam_path):
        sam_data = np.load(sam_path, allow_pickle=True)
        print("Loaded sam data from:", sam_path)
        
    # Plot sam masks for first frame
    instances_masks = None
    if sam_data is not None:
        instances_masks = []
        for fid in range(len(rgbs)):
            objs_dict = sam_data[str(fid)].item()
            instance_mask = np.zeros((height, width), dtype=np.int32)
            for oid in objs_dict:
                mask = objs_dict[oid]  # (N, H, W)
                instance_mask[mask] = int(oid) + 1  # start from 1
            instances_masks.append(instance_mask)
            # plt.imshow(instance_mask, cmap='tab20')
            # plt.colorbar()
            # plt.savefig(f"sam_instance_mask_{fid:03d}.png")
            # plt.close()
        instances_masks = np.stack(instances_masks, axis=0)  # (N, H, W)
    
    flow_as_disp = False
    if flow_as_disp:
        
        assert os.path.exists(flow_path), f"flow file not found: {flow_path}"
        
        with open(flow_path, 'rb') as f:
            flows = pickle.load(f)
        
        disps = []
        for fid in range(nfr):
            # load flow data, interpreted as disparity
            flow_fwd = flows[fid]['fwd'].astype(np.float32)
            flow_bwd = flows[fid]['bwd'].astype(np.float32)
            disp = np.clip(-flows[fid]['fwd'][..., 0], 0, None)  # (H, W)
        
            # Remove occluded points
            flow_bwd_warp = utils.inverse_warp(flow_bwd, flow_fwd)
            occ_mask_left = np.linalg.norm(flow_bwd_warp + flow_fwd, axis=-1) > 1
            disp[occ_mask_left] = np.inf
            disp[np.abs(flow_fwd[..., 1]) > 1] = np.inf
            
            disps.append(disp)
        disps = np.stack(disps, axis=0)  # (N, H, W)
    
    else:
        
        assert os.path.exists(disp_path), f"disparity file not found: {disp_path}"
    
        # load disparity data
        disps = np.load(disp_path)["disps"]  # (N, H, W)
        if disps.dtype == np.uint16:
            disps = utils.unquantize_from_uint16(disps, X_min=0.0, X_max=float(width))
            # remove highest value
            disps[disps >= width] = np.inf
        assert disps.dtype == np.float32, f"unexpected disparity dtype: {disps.dtype}"
        
        disps[disps <= MIN_DEPTH] = np.inf
        disps[disps == disps.max()] = np.inf
    
    assert (disps == 0).sum() == 0, "disparities have zero values"
    
    # Load meta
    dp = utils.load_dataset_npz(meta_path)
    # print("dp.keys():", dp.keys())
    
    # Load extrinsics
    extrs_rectified = dp['extrs_rectified']
    
    tracks3d = dp["track3d"]  # (N, T, 3)

    input_dict['nfr'] = nfr
    
    depths = []
    uncertainties = []
        
    # Iterate frames
    for fid in tqdm(range(nfr)):
        
        # load camera intrinsics
        intr_normalized = {
            'fx': (1 / 2.0) / math.tan(math.radians(hfov / 2.0)),
            'fy': (
                (1 / 2.0) / math.tan(math.radians(hfov / 2.0))
            ),
            'cx': 0.5,
            'cy': 0.5,
            'k1': 0,
            'k2': 0,
        }
        # print("intr_normalized", intr_normalized)
        
        input_dict['left']['camera'].append(
            utils.CameraAZ(
                from_json={
                    'extr': extrs_rectified[fid][:3, :],
                    'intr_normalized': intr_normalized,
                }
            )
        )
        
        # get disparity map
        disp = np.clip(disps[fid], 0, None)
        
        # # plot disparity map
        # plt.imshow(disp, cmap='turbo')
        # plt.colorbar()
        # plt.savefig(f"disp_{fid:03d}.png")
        # plt.close()
        
        hfov_deg = input_dict['left']['camera'][fid].get_hfov_deg()
        
        depth, uncertainty = utils.disparity_to_depth(
            disp, hfov_deg, baseline=0.063
        )  # pytype: disable=attribute-error
        
        
        # depth = utils.radial_to_z_depth(
        #     depth,
        #     fx=intr_normalized['fx'] * width,
        #     fy=intr_normalized['fy'] * height,
        #     cx=intr_normalized['cx'] * width,
        #     cy=intr_normalized['cy'] * height
        # )
        
        depths.append(depth)
        uncertainties.append(uncertainty)
        
    depths = np.stack(depths, axis=0)
    depths[depths > MAX_DEPTH] = 0
    depths[depths <= MIN_DEPTH] = 0

    # remove floating points
    mask = utils.gradient_check_mask_relative(depths, 0.03)
    depths[mask] = 0
    
    input_dict['left']['depth'] = depths
    
    # unproject to point cloud
    intr_normalized = input_dict['left']['camera'][0].intr_normalized
    K = np.array([
        [intr_normalized['fx'], 0, intr_normalized['cx']],
        [0, intr_normalized['fy'], intr_normalized['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    K[0, :] *= width
    K[1, :] *= height
    
    update_tracks3d = False
    if update_tracks3d:
        
        # project tracks3d to 2d
        tracks2d = []
        for t in range(tracks3d.shape[1]):
            points3d = tracks3d[:, t, :]  # (N, 3)
            pose_c2w = np.eye(4, dtype=np.float32)
            pose_c2w[:3, :4] = extrs_rectified[t]
            
            points2d = utils.project_points_3d_to_2d(
                points3d, K, pose_c2w
            )  # (N, 2)
            tracks2d.append(points2d)
            # assert np.isnan(points2d).sum() == 0, "nan in projected 2d points"
            
        tracks2d = np.stack(tracks2d, axis=1)  # (N, T, 2)
        
        vis_tracks2d = False
        if vis_tracks2d:
        
            # plot 2d points on images
            tracks2d_subset = tracks2d[::10]  # subsample for visualization
            
            # first frame points
            pts_colors = np.zeros_like(tracks2d_subset[:, 0, :] , dtype=np.float32)
            empty_color = np.ones((tracks2d_subset.shape[0],), dtype=bool)
            for t in range(tracks2d_subset.shape[1]):
                pts2d = tracks2d_subset[:, t, :]  # (N, 2)
                # filter nan or inf points
                mask = np.isfinite(pts2d).all(axis=-1)
                mask = mask & empty_color
                color = pts2d[mask] / np.array([width, height])
                color = np.clip(color, 0, 1)
                pts_colors[mask] = color 
                empty_color |= mask
                print(f"Frame {t}: filled {mask.sum()} points")
                if ~empty_color.any():
                    break
            pts_colors = np.concatenate([pts_colors, np.zeros((pts_colors.shape[0], 1), dtype=np.float32)], axis=-1)
            
            os.makedirs("tracks2d", exist_ok=True)
            for fid in range(tracks2d_subset.shape[1]):
                # fig = plt.figure(figsize=(5, 5))
                # plt.subplot(1, len(frames), fid + 1)
                plt.imshow(rgbs[fid])
                pts2d = tracks2d_subset[:, fid, :]  # (N, 2)
                # filter nan or inf points
                mask = np.isfinite(pts2d).all(axis=-1)
                pts2d = pts2d[mask]
                color = pts_colors[mask]
                plt.scatter(pts2d[:, 0], pts2d[:, 1], s=2, c=color)
                plt.savefig(f"tracks2d/{fid:03d}.png")
                plt.close()
        
        # unproject 2d tracks to 3d using new depth maps
        tracks3d = []
        for t in range(tracks2d.shape[1]):
            points2d = tracks2d[:, t, :]  # (N, 2)
            depth = depths[t]  # (H, W)
            points_depth = utils.sample_depth_from_2d_points(
                points2d, depth
            )  # (N,)
            points_depth[points_depth <= MIN_DEPTH] = np.nan  # mark invalid depth
            # unproject to 3d
            pose_c2w = np.eye(4, dtype=np.float32)
            pose_c2w[:3, :4] = extrs_rectified[t]
            points3d = utils.unproject_points_2d_to_3d(
                points2d, points_depth, K, pose_c2w
            )  # (N, 3)
            tracks3d.append(points3d)
            # assert np.isnan(tracks3d).sum() == 0, "nan in unprojected 3d points"
            
        tracks3d = np.stack(tracks3d, axis=1)  # (N, T, 3)
        
    vis_tracks = True
    o3d_utils.run_open3d_viewer(
        rgbs,
        depths,
        K,
        poses_c2w=extrs_rectified,
        tracks3d=tracks3d if vis_tracks else None,
        instances_masks=instances_masks
    )
    
    return input_dict



if __name__ == "__main__":
    
    root_dir = "/home/stefano/Codebase/stereo4d-code/data"
    # 
    split = "test"
    scene = "H5xOyNqJkPs"
    timestamp = "38738739"
    
    load_rgbd_cam_from_pkl(root_dir, split, scene, timestamp, hfov=60.0)