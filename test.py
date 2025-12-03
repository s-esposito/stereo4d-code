import numpy as np
from utils import load_video_frames
import matplotlib.pyplot as plt
import open3d as o3d
# import torch
import math
import utils


def load_rgbd_cam_from_pkl(root_dir: str, split:str, scene:str, timestamp:str, hfov: float):
    """load rgb, depth, and camera"""
    
    # load test data from npz
    npz_path = f"{root_dir}/{split}/{split}_{scene}_{timestamp}.npz"
    video_path = f"{root_dir}/{split}/{split}_{scene}_{timestamp}.mp4"
    disp_path = f"{root_dir}/{split}/{split}_{scene}_{timestamp}_disp.npz"

    # load disparity data
    disps = np.load(disp_path)["disps_left"]
    disps = disps.astype(np.float32)
    disps[disps == disps.max()] = np.inf
    # disps /= 256.0
    # print("disps", disps.shape, disps.dtype, disps.min(), disps.max())
    
    # prepare input dict
    input_dict = {
        'left': {
            'camera': [],
            'depth': [],
            'video': []
        }
    }
    
    # Load video frames     
    rgbs, _ = load_video_frames(video_path)
    input_dict['left']['video'] = rgbs
    
    #
    height, width = rgbs[0].shape[:2]
    print("video frames", len(rgbs), height, width)
    
    # Load camera
    dp = utils.load_dataset_npz(npz_path)
    extrs_rectified = dp['extrs_rectified']
    
    # name = data["name"]
    # print("name:", name)

    # video_id = data["video_id"]
    # print("video_id:", video_id)

    # # timestamps = data["timestamps"]
    # # print("timestamps:", timestamps)

    # camera2world = data["camera2world"]
    # print("camera2world shape:", camera2world.shape)

    # track_lengths = data["track_lengths"]
    # print("track_lengths:", track_lengths)

    # track_indices = data["track_indices"]
    # print("track_indices:", track_indices)

    # rectified2rig = data["rectified2rig"]
    # print("rectified2rig shape:", rectified2rig)

    # fov_bounds = data["fov_bounds"]
    # print("fov_bounds shape:", fov_bounds)

    nfr = len(extrs_rectified)
    input_dict['nfr'] = nfr
    
    depths = []
    uncertainties = []
        
    # Iterate frames
    for fid in range(nfr):
        
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
        print("intr_normalized", intr_normalized)
        
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
        
        assert (disp == 0).sum() == 0, "disparity map has zero values"
        
        # plot disparity map
        plt.imshow(disp, cmap='turbo')
        plt.colorbar()
        plt.savefig(f"disp_{fid:03d}.png")
        plt.close()
        
        hfov_deg = input_dict['left']['camera'][fid].get_hfov_deg()
        
        depth, uncertainty = utils.disparity_to_depth(
            disp, hfov_deg, baseline=0.063
        )  # pytype: disable=attribute-error
        
        depths.append(depth)
        uncertainties.append(uncertainty)
        
        break
        
    depths = np.stack(depths, axis=0)
    depths[depths > 20] = 0
    depths[depths < 0] = 0

    # remove floating points
    # mask = utils.gradient_check_mask_relative(depths, 0.03)
    # depths[mask] = 0
    
    input_dict['left']['depth'] = depths
    
    # plot depth map
    # plt.subplot(1, 2, 1)
    depth_vis = depths.copy()
    depth_vis[depth_vis == 0] = np.nan
    plt.imshow(depth_vis[0], cmap='turbo')
    plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(uncertainties[0], cmap='turbo')
    # plt.colorbar()
    plt.savefig(f"depth_{0:03d}.png")
    plt.close()
    
    # unproject to point cloud
    intr_normalized = input_dict['left']['camera'][0].intr_normalized
    K = np.array([
        [intr_normalized['fx'], 0, intr_normalized['cx']],
        [0, intr_normalized['fy'], intr_normalized['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    K[0, :] *= width
    K[1, :] *= height
    pc = utils.depth2xyz(depths[0], K)
    print(pc.shape, pc.dtype, pc.min(), pc.max())
    
    return input_dict



## visualize point cloud with open3d
#pc_path = "cloud.ply"
#pcd = o3d.io.read_point_cloud(pc_path)
#o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    
    root_dir = "/home/geiger/gwb987/work/data/stereo4d/stereo4d-code"
    # 
    split = "test"
    scene = "-0OgBspHPQE"
    timestamp = "15080429"
    
    load_rgbd_cam_from_pkl(root_dir, split, scene, timestamp, hfov=60.0)