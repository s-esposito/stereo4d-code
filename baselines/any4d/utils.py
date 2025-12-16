import numpy as np
import torch
import os
from o3d_renderer import run_open3d_viewer
from o3d_renderer import run_open3d_offline_renderer


def view_with_open3d_viewer(data: dict):
    
    rgbs = data['images']  # (T, H, W, 3), float32 in [0, 1]
    # convert to uint8 [0, 255]
    rgbs = (rgbs * 255.0).astype(np.uint8)

    depths = data['depths']  # (T, H, W)
    point_maps = data['points']  # (T, H, W, 3)  
    K = data['camera_intrinsics']  # (T, 3, 3)
    poses_c2w = data['camera_poses']  # (T, 4, 4)
    
    # use foreground mask as instance masks (convert to uint8)
    masks = data['masks']
    # if instances_masks is not None:
    #     instances_masks = instances_masks.astype(np.uint8)
    instances_masks = None
    
    nr_frames = len(rgbs)
    
    point_clouds = []
    for t in range(nr_frames):
        pts_t = point_maps[t]  # (H, W, 3)
        rgbs_t = rgbs[t]  # (H, W, 3)
        mask_t = masks[t] # (H, W)
        # flatten
        pts_t = pts_t.reshape(-1, 3)
        rgbs_t = rgbs_t.reshape(-1, 3)
        mask_t = mask_t.reshape(-1)
        # mask
        pts_t = pts_t[mask_t]
        rgbs_t = rgbs_t[mask_t]
        pcd = {'xyz': pts_t, 'rgb': rgbs_t}
        point_clouds.append(pcd)
        
    #
    tracks3d = data['point_tracks']  # (N, T, 3)
    
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
    
def load_output(output_dir):
    """Load all time-concatenated data files."""
    
    # assert output_dir is a directory
    assert os.path.isdir(output_dir), f"Output directory {output_dir} does not exist."
    
    all_data = {}
    
    # Load depths (T, H, W)
    depths_path = os.path.join(output_dir, "depths.npy")
    if os.path.exists(depths_path):
        all_data['depths'] = np.load(depths_path)  # (T, H, W)
        # remove first element of T dimension
        all_data['depths'] = all_data['depths'][1:]
        # print(f"Loaded depths: {all_data['depths'].shape}")
    
    # # Load depth visualizations (T, H, W)
    # depths_vis_path = os.path.join(output_dir, "depths_vis.npy")
    # if os.path.exists(depths_vis_path):
    #     all_data['depths_vis'] = np.load(depths_vis_path)
    #     print(f"Loaded depth visualizations: {all_data['depths_vis'].shape}")
    
    # Load points (T, H, W, 3)
    points_path = os.path.join(output_dir, "points.npy")
    if os.path.exists(points_path):
        all_data['points'] = np.load(points_path)
        # remove first element of T dimension
        all_data['points'] = all_data['points'][1:]
        # print(f"Loaded point clouds: {all_data['points'].shape}")
    
    # Load masks (T, H, W)
    masks_path = os.path.join(output_dir, "masks.npy")
    if os.path.exists(masks_path):
        all_data['masks'] = np.load(masks_path)
        # remove first element of T dimension
        all_data['masks'] = all_data['masks'][1:]
        # print(f"Loaded masks: {all_data['masks'].shape}")
    
    # Load images (T, H, W, 3)
    images_path = os.path.join(output_dir, "images.npy")
    if os.path.exists(images_path):
        all_data['images'] = np.load(images_path)
        # remove first element of T dimension
        all_data['images'] = all_data['images'][1:]
        # print(f"Loaded images: {all_data['images'].shape}")
    
    # Load camera poses (T, 4, 4)
    poses_path = os.path.join(output_dir, "camera_poses.npy")
    if os.path.exists(poses_path):
        all_data['camera_poses'] = np.load(poses_path)
        # remove first element of T dimension
        all_data['camera_poses'] = all_data['camera_poses'][1:]
        # print(f"Loaded camera poses: {all_data['camera_poses'].shape}")
    
    # Load camera intrinsics (T, 3, 3)
    intrinsics_path = os.path.join(output_dir, "camera_intrinsics.npy")
    if os.path.exists(intrinsics_path):
        all_data['camera_intrinsics'] = np.load(intrinsics_path)
        # remove first element of T dimension
        all_data['camera_intrinsics'] = all_data['camera_intrinsics'][1:]
        # print(f"Loaded camera intrinsics: {all_data['camera_intrinsics'].shape}")
    
    # # Load scene flow masks (T, H, W)
    # sf_masks_path = os.path.join(output_dir, "scene_flow_masks.npy")
    # if os.path.exists(sf_masks_path):
    #     all_data['scene_flow_masks'] = np.load(sf_masks_path)
    #     print(f"Loaded scene flow masks: {all_data['scene_flow_masks'].shape}")
    
    # # Load scene flows (T-1, H, W, 3)
    # scene_flows_path = os.path.join(output_dir, "scene_flows.npy")
    # if os.path.exists(scene_flows_path):
    #     all_data['scene_flows'] = np.load(scene_flows_path)
    #     print(f"Loaded scene flows: {all_data['scene_flows'].shape}")
    
    # # Load scene flow magnitude visualizations (T-1, H, W)
    # sf_mag_vis_path = os.path.join(output_dir, "scene_flows_magnitude_vis.npy")
    # if os.path.exists(sf_mag_vis_path):
    #     all_data['scene_flows_magnitude_vis'] = np.load(sf_mag_vis_path)
    #     print(f"Loaded scene flow magnitude vis: {all_data['scene_flows_magnitude_vis'].shape}")
    
    # Load 3D tracks (N, T, 3)
    tracks3d_path = os.path.join(output_dir, "point_tracks.npy")
    if os.path.exists(tracks3d_path):
        tracks3d = np.load(tracks3d_path)  # (T-1, M, N, 3)
        # reshape to (T-1, N, 3)
        tracks3d = tracks3d.reshape(tracks3d.shape[0], -1, 3)
        # permute to (N, T-1, 3)
        all_data['point_tracks'] = np.transpose(tracks3d, (1, 0, 2))
        #
        all_data['point_tracks'] = all_data['point_tracks']
        # # add first frame with nans at t=0
        # n_points = all_data['point_tracks'].shape[0]
        # last_frame = np.full((n_points, 1, 3), np.nan, dtype=all_data['point_tracks'].dtype)
        # all_data['point_tracks'] = np.concatenate((all_data['point_tracks'], last_frame), axis=1)
        #
        print(f"Loaded 3D tracks: {all_data['point_tracks'].shape}")
    
    return all_data