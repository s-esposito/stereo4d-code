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
    meta_path = f"{root_dir}/{split}/{scene}_{timestamp}/meta.npz"
    video_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-left_rectified.mp4"
    disp_path = f"{root_dir}/{split}/{scene}_{timestamp}/disps.npz"
        
    # Load video frames     
    rgbs, _ = load_video_frames(video_path)
    input_dict['left']['video'] = rgbs
    nfr = len(rgbs)

    #
    height, width = rgbs[0].shape[:2]
    print("video frames", len(rgbs), height, width)
    
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
        disps = np.load(disp_path)["disps_left"]  # (N, H, W)
        if disps.dtype == np.uint16:
            disps = utils.unquantize_from_uint16(disps, X_min=0.0, X_max=float(width))
            # remove highest value
            disps[disps >= width] = np.inf
        assert disps.dtype == np.float32, f"unexpected disparity dtype: {disps.dtype}"
        
        disps[disps <= 0] = np.inf
        disps[disps == disps.max()] = np.inf
    
    assert (disps == 0).sum() == 0, "disparities have zero values"
    
    # Load camera
    dp = utils.load_dataset_npz(meta_path)
    # print("dp.keys():", dp.keys())
    extrs_rectified = dp['extrs_rectified']
    
    track3d = dp["track3d"]  # (N, T, 3)

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
    depths[depths > 20] = 0
    depths[depths < 0] = 0

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
    
    vis_tracks = False
    o3d_utils.run_open3d_viewer(
        rgbs,
        depths,
        K,
        poses_c2w=extrs_rectified,
        tracks3d=track3d if vis_tracks else None,
    )
    
    return input_dict



if __name__ == "__main__":
    
    root_dir = "/home/stefano/Codebase/stereo4d-code/data"
    # 
    split = "test"
    scene = "H5xOyNqJkPs"
    timestamp = "38738739"
    
    load_rgbd_cam_from_pkl(root_dir, split, scene, timestamp, hfov=60.0)