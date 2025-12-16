import pickle
import numpy as np
from utils import load_video_frames
import matplotlib.pyplot as plt
import open3d as o3d
import os
import math
import time
import utils
# import o3d_utils
from tqdm import tqdm
from load_data import load_data
import argparse
from o3d_renderer import run_open3d_viewer
from o3d_renderer import run_open3d_offline_renderer


def _run_open3d_viewer(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intr_normalized: np.ndarray,
    width: int, 
    height: int,
    poses_c2w: tuple[np.ndarray, np.ndarray] | np.ndarray,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None
):
    """
    Run the interactive Open3D viewer for visualizing point clouds and tracks.
    
    Args:
        rgbs: RGB images (T, H, W, 3)
        depths: Depth maps (T, H, W)
        intr_normalized: Normalized intrinsic matrix
        width: Image width
        height: Image height
        poses_c2w: Camera poses (world to camera)
        tracks3d: Optional 3D tracks (N, T, 3)
        instances_masks: Optional instance segmentation masks (T, H, W)
    """
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height
    print("Intrinsic Matrix K:\n", K)
    
    point_clouds = None
    for fid in range(len(rgbs)):
        
        if point_clouds is None:
            point_clouds = []
        
        # Check if stereo camera
        if isinstance(poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = poses_c2w[1][fid]
        else:
            pose_c2w = poses_c2w[fid]
        
        rgb = rgbs[fid]
        depth = depths[fid]
        instances = instances_masks[fid] if instances_masks is not None else None
        
        pcd = utils.generate_point_cloud(rgb, depth, K, pose_c2w, instances=instances)
        point_clouds.append(pcd)
    
    nr_frames = len(rgbs)
    
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
    
    
def _run_open3d_offline_renderer(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intr_normalized: np.ndarray,
    width: int, 
    height: int,
    poses_c2w: tuple[np.ndarray, np.ndarray] | np.ndarray,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None
):
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height

    point_clouds = []
    for fid in range(len(rgbs)):
        # Check if stereo camera
        if isinstance(poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = poses_c2w[1][fid]
        else:
            pose_c2w = poses_c2w[fid]
        
        rgb = rgbs[fid]
        depth = depths[fid]
        instances = instances_masks[fid] if instances_masks is not None else None
        
        pcd = utils.generate_point_cloud(rgb, depth, K, pose_c2w, instances=instances)
        point_clouds.append(pcd)
    
    # Reduced resolution for faster rendering
    nr_frames = len(rgbs)
    base_width, base_height = 512, 512
    res_scale = 1
    width = base_width * res_scale
    height = base_height * res_scale
    
    frames = run_open3d_offline_renderer(
        width, height, nr_frames,
        rgbs=rgbs,
        depths=depths,
        point_clouds=point_clouds,
        K=K,
        poses_c2w=poses_c2w,
        tracks3d=tracks3d,
        instances_masks=instances_masks
    )
    
    return frames


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="View data parameters")
    parser.add_argument('--root_dir', type=str, required=False, default="data", help='Root directory of the dataset')
    parser.add_argument('--split', type=str, required=False, default="test", help='Dataset split (e.g., train, val, test)')
    parser.add_argument('--scene', type=str, required=False, default="H5xOyNqJkPs", help='Scene identifier')
    parser.add_argument('--timestamp', type=str, required=False, default="38738739", help='Timestamp identifier')
    parser.add_argument('--view', action='store_true', help='Whether to view the data using Open3D, else uses offline visualization')
    parser.add_argument('--view-in-browser', action='store_true', help='Whether to view the data in browser')
    parser.add_argument('--output-video-path', type=str, required=False, default="videogallery/videos", help='Output path for the generated video')
    args = parser.parse_args()
    
    if args.view and args.view_in_browser:
        o3d.visualization.webrtc_server.enable_webrtc()
    
    input_dict = load_data(args.root_dir, args.split, args.scene, args.timestamp)

    rgbs_left = input_dict['left']['video']
    print("rgbs_left shape:", rgbs_left.shape)
    rgbs_right = input_dict['right']['video']
    print("rgbs_right shape:", rgbs_right.shape)
    depths = input_dict['depths']
    print("depths shape:", depths.shape)
    intr_normalized = input_dict['intr_normalized']
    print("intr_normalized:", intr_normalized)
    extrs_left = input_dict['left']['camera']
    print("extrs_left length:", len(extrs_left))
    extrs_right = input_dict['right']['camera']
    print("extrs_right length:", len(extrs_right))
    tracks3d = input_dict.get('tracks3d', None)
    if tracks3d is not None:
        print("tracks3d shape:", tracks3d.shape)
    instances_masks = input_dict.get('instances_masks', None)
    if instances_masks is not None:
        print("instances_masks shape:", instances_masks.shape)
        
    width, height = rgbs_left.shape[2], rgbs_left.shape[1]
    
    if args.view:
        
        _run_open3d_viewer(
            rgbs=rgbs_right,
            depths=depths,
            intr_normalized=intr_normalized,
            width=width, height=height,
            poses_c2w=(extrs_left, extrs_right),
            tracks3d=tracks3d,
            instances_masks=instances_masks
        )
    else:
        # Offline rendering
        
        frames = _run_open3d_offline_renderer(
            rgbs=rgbs_right,
            depths=depths,
            intr_normalized=intr_normalized,
            width=width,
            height=height,
            poses_c2w=(extrs_left, extrs_right),
            tracks3d=tracks3d,
            instances_masks=instances_masks,
        )

        # store as mp4 video
        video_path = args.output_video_path
        video_name = f"{args.scene}_{args.timestamp}"
        utils.create_video_from_frames(frames, video_path, video_name, fps=30, file_format='mp4')