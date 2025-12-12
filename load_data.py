import pickle
import numpy as np
from utils import load_video_frames
import matplotlib.pyplot as plt
import os
import math
import utils
from tqdm import tqdm

MIN_DEPTH = 0.1
# MAX_DEPTH = 20.0


def load_data(root_dir: str, split:str, scene:str, timestamp:str):
    """load rgb, depth, cameras and addational data from dataset"""
    
    # prepare output dict
    input_dict = {
        'left': {
            'camera': [],
            'video': []
        },
        'right': {
            'camera': [],
            'video': []
        },
    }
    
    # fixed parameters
    hfov=60.0
    baseline=0.063
    
    # load test data from npz
    flow_path = f"{root_dir}/stereo4d-flows-stereo/{split}/{scene}_{timestamp}-flows_stereo.pkl"
    meta_path = f"{root_dir}/stereo4d-npz/{split}/{scene}_{timestamp}.npz"
    left_video_path = f"{root_dir}/stereo4d-lefteye-perspective/{split}_mp4s/{scene}_{timestamp}-left_rectified.mp4"
    right_video_path = f"{root_dir}/stereo4d-righteye-perspective/{split}_mp4s/{scene}_{timestamp}-right_rectified.mp4"
    disp_path = f"{root_dir}/stereo4d-disps/{split}/{scene}_{timestamp}-disps.npz"
    sam_path = f"{root_dir}/stereo4d-sam3/{split}/{scene}_{timestamp}-sam3.npz"
        
    # Load video frames     
    rgbs_left, _ = load_video_frames(left_video_path)
    rgbs_left = np.stack(rgbs_left, axis=0)  # (N, H, W, 3)
    input_dict['left']['video'] = rgbs_left
    
    rgbs_right, _ = load_video_frames(right_video_path)
    rgbs_right = np.stack(rgbs_right, axis=0)  # (N, H, W, 3)
    input_dict['right']['video'] = rgbs_right
    
    nfr = len(rgbs_right)
    input_dict['nfr'] = nfr
    
    #
    height, width = rgbs_right[0].shape[:2]
    print("video frames", len(rgbs_right), height, width)
    
    # Load sam data
    sam_data = None
    semantic_instances_masks = {}
    if os.path.exists(sam_path):
        sam_data = np.load(sam_path, allow_pickle=True)
        for key in sam_data:
            value = sam_data[key]
            if value is not None:
                semantic_instances_masks[key] = sam_data[key]
    
    # convert semantic instances masks to instances masks
    instances_masks = None
    nr_instances = 0
    for key, instances in semantic_instances_masks.items():
        if instances_masks is None:
            instances_masks = instances.copy()
        else:
            non_zero = instances > 0
            instances_masks[non_zero] += instances[non_zero] + nr_instances
        nr_instances += len(np.unique(instances)) - 1  # exclude background
        
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
    
    # Load extrinsics (TODO Stefano: double check if this is indeed the right eye)
    extrs_left = dp['extrs_rectified']
    # use baseline to compute left camera extrinsics
    extrs_right = extrs_left.copy()
    extrs_right[:, :3, 3] -= extrs_right[:, :3, 0] * baseline  # translate along x axis
    
    # Invert both extrinsics to get camera-to-world
    extrs_right_inv = []
    extrs_left_inv = []
    for fid in range(nfr):
        cam_right = extrs_right[fid] # (3, 4)
        # to (4, 4)
        cam_right_hom = np.eye(4, dtype=np.float32)
        cam_right_hom[:3, :4] = cam_right
        extrs_right_inv.append(np.linalg.inv(cam_right_hom))
        cam_left = extrs_left[fid] # (3, 4)
        # to (4, 4)
        cam_left_hom = np.eye(4, dtype=np.float32)
        cam_left_hom[:3, :4] = cam_left
        extrs_left_inv.append(np.linalg.inv(cam_left_hom))
    extrs_right = np.stack(extrs_right_inv, axis=0)
    extrs_left = np.stack(extrs_left_inv, axis=0)
    
    tracks3d = dp["track3d"]  # (N, T, 3)
    
    depths = []
    uncertainties = []
    
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
        
    # Iterate frames
    for fid in tqdm(range(nfr)):
        
        # store cameras
        input_dict['right']['camera'].append(extrs_right[fid][:3, :])
        input_dict['left']['camera'].append(extrs_left[fid][:3, :])
        
        # get disparity map
        disp = np.clip(disps[fid], 0, None)
        
        # # plot disparity map
        # plt.imshow(disp, cmap='turbo')
        # plt.colorbar()
        # plt.savefig(f"disp_{fid:03d}.png")
        # plt.close()
        
        depth, uncertainty = utils.disparity_to_depth(
            disp, hfov, baseline
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
        
    # stack extrs
    input_dict['left']['camera'] = np.stack(input_dict['left']['camera'], axis=0)
    input_dict['right']['camera'] = np.stack(input_dict['right']['camera'], axis=0)
    
    # max depth is any depth larger than <1 disparity
    MAX_DEPTH = baseline * (intr_normalized['fx'] * width) / 1.0
    
    depths = np.stack(depths, axis=0)
    depths[depths > MAX_DEPTH] = 0
    depths[depths <= MIN_DEPTH] = 0

    # remove floating points
    mask = utils.gradient_check_mask_relative(depths, 0.03)
    depths[mask] = 0
    
    # unproject to point cloud
    intr_normalized = np.array([
        [intr_normalized['fx'], 0, intr_normalized['cx']],
        [0, intr_normalized['fy'], intr_normalized['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height
    
    update_tracks3d = True
    if update_tracks3d:
        
        # project tracks3d to 2d
        tracks2d = []
        for t in range(tracks3d.shape[1]):
            points3d = tracks3d[:, t, :]  # (N, 3)
            # pose_c2w = np.eye(4, dtype=np.float32)
            # pose_c2w[:3, :4] = extrs_right[t]
            pose_c2w = extrs_right[t]
            
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
                # plt.imshow(rgbs[fid])
                pts2d = tracks2d_subset[:, fid, :]  # (N, 2)
                # filter nan or inf points
                mask = np.isfinite(pts2d).all(axis=-1)
                pts2d = pts2d[mask]
                color = pts_colors[mask]
                # plt.scatter(pts2d[:, 0], pts2d[:, 1], s=2, c=color)
                # plt.savefig(f"tracks2d/{fid:03d}.png")
                # plt.close()
        
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
            # pose_c2w = np.eye(4, dtype=np.float32)
            # pose_c2w[:3, :4] = extrs_right[t]
            pose_c2w = extrs_right[t]
            points3d = utils.unproject_points_2d_to_3d(
                points2d, points_depth, K, pose_c2w
            )  # (N, 3)
            tracks3d.append(points3d)
            # assert np.isnan(tracks3d).sum() == 0, "nan in unprojected 3d points"
            
        tracks3d = np.stack(tracks3d, axis=1)  # (N, T, 3)
        
    input_dict['intr_normalized'] = intr_normalized
    input_dict['depths'] = depths
    
    if tracks3d is not None:
        input_dict['tracks3d'] = tracks3d
    
    if instances_masks is not None:
        input_dict['instances_masks'] = instances_masks
    
    return input_dict

def split_data(input_dict, test_every: int = 10):
    
    # split data into train and test sets
    train_dict = {}
    test_dict = {}
    
    test_idxs = list(range(1, len(input_dict['left']['video']), test_every))
    # use all remaining indices for training
    train_idxs = [i for i in range(len(input_dict['left']['video'])) if i not in test_idxs]
    test_idxs = np.array(test_idxs)
    train_idxs = np.array(train_idxs)

    tracks3d = None
    for key in input_dict:
        
        # if the value is a dict, split each subkey
        if isinstance(input_dict[key], dict):
            train_dict[key] = {}
            test_dict[key] = {}
            for subkey in input_dict[key]:
                data = input_dict[key][subkey]
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    train_dict[key][subkey] = data[train_idxs]
                    test_dict[key][subkey] = data[test_idxs]
                else:
                    train_dict[key][subkey] = data
                    test_dict[key][subkey] = data
        else:
            
            # split directly
            print("Splitting key:", key)
            data = input_dict[key]
            print("Data type:", type(data))
            
            if key == "intr_normalized":
                # do not split intrinsics, global for all frames
                train_dict[key] = data
                test_dict[key] = data
                
            elif key == "tracks3d":
                # do not split tracks3d, global for all frames
                tracks3d = data
                
            elif key == "nfr":
                continue  # skip nfr key
            
            else:
            
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    train_dict[key] = data[train_idxs]
                    test_dict[key] = data[test_idxs]
                else:
                    train_dict[key] = data
                    test_dict[key] = data
    
    # add frames indices
    train_dict['frame_idxs'] = train_idxs
    test_dict['frame_idxs'] = test_idxs
    
    print("Train data:")
    for key, value in train_dict.items():
        if isinstance(train_dict[key], dict):
            print(f"{key}:")
            for subkey in train_dict[key]:
                print(f"  {subkey}: {train_dict[key][subkey].shape if isinstance(train_dict[key][subkey], np.ndarray) else type(train_dict[key][subkey])}")
        else:
            print(f"{key}: {train_dict[key].shape if isinstance(train_dict[key], np.ndarray) else type(train_dict[key])}")
        
    print("Test data:")
    for key, value in test_dict.items():
        if isinstance(test_dict[key], dict):
            print(f"{key}:")
            for subkey in test_dict[key]:
                print(f"  {subkey}: {test_dict[key][subkey].shape if isinstance(test_dict[key][subkey], np.ndarray) else type(test_dict[key][subkey])}")
        else:
            print(f"{key}: {test_dict[key].shape if isinstance(test_dict[key], np.ndarray) else type(test_dict[key])}")
    
    if tracks3d is not None:
        print("tracks3d shape:", tracks3d.shape)
    
    return train_dict, test_dict, tracks3d