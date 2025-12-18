import os
import json
import numpy as np
import sys
from tqdm import tqdm
import imageio
import cv2
from o3d_renderer import run_open3d_viewer
from o3d_renderer import run_open3d_offline_renderer
from projectaria_tools.core.calibration import CameraModelType, CameraProjection

# Add parent directory to path to import utils from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import utils


# Load only a few frames for testing
MAX_SEQ_LEN = 50
FORCED_FPS = 10
FIRST_FRAME_ID = 10  # wait for exposure stabilization


def view_with_open3d_viewer(data):
    nr_frames = data["rectified_images"].shape[0]
    depths = data["depth_maps"]  # (nr_frames, H, W)
    rgbs = data["rectified_images"]  # (nr_frames, H, W, 3)
    K = data["intrinsics"]  # (nr_frames, 3, 3)
    poses_c2w = data["extrinsics"]  # (nr_frames, 4, 4)
    
    point_clouds = []
    for fid in range(nr_frames):
        
        # Get K for this frame
        K_frame = K[fid] if K.ndim == 3 else K
        
        # Check if stereo camera
        if isinstance(poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = poses_c2w[1][fid]
            rgb = rgbs[1][fid]  # Use right camera RGBs
        else:
            pose_c2w = poses_c2w[fid]
            rgb = rgbs[fid]
        
        depth = depths[fid]
        instances = None
        
        pcd = utils.generate_point_cloud(rgb, depth, K_frame, pose_c2w, instances=instances)
        point_clouds.append(pcd)
    
    run_open3d_viewer(
        nr_frames,
        rgbs=rgbs,
        depths=depths,
        point_clouds=point_clouds,
        K=K,
        poses_c2w=poses_c2w,
        tracks3d=None,
        instances_masks=None,
    )


def load_data(scene_dir, scene_name):
    
    # load slam data
    scene_slam_dir = os.path.join(scene_dir, "mps/slam")
    
    # # load closed_loop_trajectory.csv
    # closed_loop_trajectory_path = os.path.join(scene_slam_dir, "closed_loop_trajectory.csv")
    # closed_loop_traj_data = []
    # with open(closed_loop_trajectory_path, 'r') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)  # skip header
    #     for row in reader:
    #         row_vals = []
    #         for val in row:
    #             # convert to float if possible
    #             try:
    #                 val = float(val)
    #             except ValueError:
    #                 pass
    #             row_vals.append(val)
    #         closed_loop_traj_data.append(row_vals)
    # print(f"Loaded closed loop trajectory with {len(closed_loop_traj_data)} entries from {closed_loop_trajectory_path}")
            
    # load online_calibration.jsonl
    online_calibration_path = os.path.join(scene_slam_dir, "online_calibration.jsonl")
    online_calibration_data = []
    with open(online_calibration_path, 'r') as f:
        for line in f:
            online_calibration_data.append(json.loads(line))
    print(f"Loaded online calibration for {len(online_calibration_data)} frames from {online_calibration_path}")

    distortion_params = []
    for calibration_data in online_calibration_data:
        # 'ImageSizes', 'utc_timestamp_ns', 'tracking_timestamp_us', 'ImuCalibrations', 'CameraCalibrations'
        rgb_image_size = calibration_data['ImageSizes'][-1]  # last is RGB
        rgb_calibration_data = calibration_data['CameraCalibrations'][-1]  # last is RGB
        print("RGB image size:", rgb_image_size)
        print("RGB calibration data keys:", rgb_calibration_data.keys())
        # print("Calibrated", rgb_calibration_data['Calibrated'])  # True
        # print("Projection", rgb_calibration_data['Projection'])
        params = rgb_calibration_data['Projection']['Params']  # FisheyeRadTanThinPrism
        print("Distortion parameters:", params)
        distortion_params.append(params)
    distortion_params = np.array(distortion_params)  # (T, 15)
    print("Distortion parameters shape:", distortion_params.shape)
    
    # check if distortion params are the same for all frames
    all_same = np.all(np.isclose(distortion_params, distortion_params[0], atol=1e-6), axis=1)
    if np.all(all_same):
        print("All distortion parameters are the same for all frames.")
    else:
        raise ValueError("Varying distortion parameters not supported yet.")
    distortion_params = distortion_params[0]
    
    # camera_projection = CameraProjection(
    #     CameraModelType.FISHEYE624, distortion_params
    # )
    
    
    # load video
    video_path = os.path.join(scene_dir, "video_main_rgb.mp4")
    frames, nr_frames = utils.load_video_frames(video_path, MAX_SEQ_LEN, FORCED_FPS, FIRST_FRAME_ID)
    
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    print(f"Loaded {nr_frames} frames from video at {video_path}")
    
    # # Initialize the rectifier once
    # rectifier = FisheyeRectifier(distortion_params, w, h)
    
    # # undistort frames
    # undistorted_frames = []
    # for fid in range(nr_frames):
    #     frame = frames[fid]
    #     undistorted_frame = rectifier.apply(frame)
    #     undistorted_frames.append(undistorted_frame)
    # undistorted_frames = np.stack(undistorted_frames, axis=0)  # (T, H, W, 3)
    
    # # plot first frame
    # import matplotlib.pyplot as plt
    # plt.imshow(frames[0])
    # plt.title(f"First frame of {scene_name}")
    # plt.show()
    
    # plt.imshow(undistorted_frames[0])
    # plt.title(f"First undistorted frame of {scene_name}")
    # plt.show()
    
    # exit(0)
    
    # correct scene dir
    scene_depth_dir = os.path.join(scene_dir, "depth")
    
    # camera params
    camera_params_path = os.path.join(scene_depth_dir, "pinhole_camera_parameters.json")
    # load camera intrinsics and extrinsics from json
    # format is:
    # [
    #     {
    #         "T_world_camera": {
    #         "QuaternionXYZW": [
    #             -0.42409029603004456,
    #             0.80664318799972534,
    #             -0.30428311228752136,
    #             0.27728331089019775
    #         ],
    #         "Translation": [
    #             -1.7272903919219971,
    #             1.4772109985351562,
    #             -0.098842501640319824
    #         ]
    #         },
    #         "camera": {
    #         "ModelName": "Linear:fu,fv,u0,v0",
    #         "Parameters": [
    #             307.88760375976562,
    #             307.88760375976562,
    #             253.32270812988281,
    #             259.04135131835938
    #         ]
    #         },
    #         "frameTimestampNs": 4133156311000,
    #         "index": 0
    #     },
    #     ...
    # ]
    with open(camera_params_path, 'r') as f:
        camera_params = json.load(f)
    print(f"Loaded camera parameters for {len(camera_params)} frames from {camera_params_path}")
    
    # normal fps is 30, skip frames to force to FORCED_FPS
    nr_og_frames = len(camera_params)
    frame_skip = max(1, 30 // FORCED_FPS)
    frames_idxs = np.array(list(range(FIRST_FRAME_ID, nr_og_frames, frame_skip)))
    frames_idxs = frames_idxs[:MAX_SEQ_LEN]
    # print(f"Using frame indices: {frames_idxs}")
    
    subsampled_camera_params = []
    for idx in frames_idxs:
        subsampled_camera_params.append(camera_params[idx])
    camera_params = subsampled_camera_params
    print(f"Subsampled to {len(camera_params)} frames after applying forced FPS and max seq len")
    
    # process camera intrinsics and extrinsics
    intrinsics_list = []
    extrinsics_list = []  # these should be camera2world transforms
    for param in camera_params:
        intrinsics = param['camera']['Parameters']
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        intrinsics_list.append(K)
        
        # Convert quaternion (XYZW) and translation to 4x4 transformation matrix
        quat = param['T_world_camera']['QuaternionXYZW']
        translation = param['T_world_camera']['Translation']
        
        # Convert quaternion to rotation matrix
        # Quaternion: [x, y, z, w]
        x, y, z, w = quat
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        
        # Construct 4x4 camera-to-world transformation matrix
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[:3, :3] = R
        T_world_camera[:3, 3] = translation
        
        extrinsics_list.append(T_world_camera)
        
    # Stack intrinsics and extrinsics
    intrinsics = np.stack(intrinsics_list, axis=0)  # (T, 3, 3)
    extrinsics = np.stack(extrinsics_list, axis=0)  # (T, 4, 4)
    print(f"Loaded intrinsics shape: {intrinsics.shape}")
    print(f"Loaded extrinsics (camera-to-world) shape: {extrinsics.shape}")
    
    # re-base all extrinsics such that the first frame is at the origin
    T0_inv = np.linalg.inv(extrinsics[0])
    extrinsics = T0_inv @ extrinsics
    
    # load data
    depth_dir = os.path.join(scene_depth_dir, "depth")
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".png")]
    depth_files.sort()
    print(f"Found {len(depth_files)} frames in {depth_dir}")
    
    subsampled_depth_files = []
    for idx in frames_idxs:
        subsampled_depth_files.append(depth_files[idx])
    depth_files = subsampled_depth_files
    print(f"Subsampled to {len(depth_files)} depth files after applying forced FPS and max seq len")
    
    # load all .png files in rectified_dir
    rectified_dir = os.path.join(scene_depth_dir, "rectified_images")
    rectified_files = [f for f in os.listdir(rectified_dir) if f.endswith(".png")]
    rectified_files.sort()
    print(f"Found {len(rectified_files)} frames in {rectified_dir}")
    
    subsampled_rectified_files = []
    for idx in frames_idxs:
        subsampled_rectified_files.append(rectified_files[idx])
    rectified_files = subsampled_rectified_files
    print(f"Subsampled to {len(rectified_files)} rectified image files after applying forced FPS and max seq len")
    
    # load all depth maps
    depth_maps = []
    for depth_path in tqdm(depth_files):
        # load depth from .png file
        depth_path = os.path.join(depth_dir, depth_path)
        # load .png as numpy array using imageio  
        depth_map = imageio.imread(depth_path)
        # convert to float32
        depth_map = depth_map.astype(np.float32) / 1000.0  # convert from mm to meters
        depth_maps.append(depth_map)
    depth_maps = np.stack(depth_maps, axis=0)  # (T, H, W)
    print(f"Loaded depth maps shape: {depth_maps.shape}")
    
    # load all rectified images
    rectified_images = []
    for rgb_path in tqdm(rectified_files):
        rgb_path = os.path.join(rectified_dir, rgb_path)
        rgb_img = imageio.imread(rgb_path)
        rectified_images.append(rgb_img)
    rectified_images = np.stack(rectified_images, axis=0)  # (T, H, W)
    # convert to RGB for compatibility
    rectified_images = np.repeat(rectified_images[..., np.newaxis], 3, axis=-1)
    w, h = rectified_images.shape[2], rectified_images.shape[1]
    print(f"Loaded rectified images shape: {rectified_images.shape}")
    
    # 1. Your input pixel on the 'clean' undistorted image
    points_undistorted = np.array([[359, 246]])
    print("Points undistorted:", points_undistorted)

    # # 2. Extract Intrinsics
    # fu, fv, cu, cv = distortion_params[0:4]

    # # 3. Convert Pixel -> Normalized Plane (Crucial Step)
    # # This centers the point and scales it so the math doesn't explode
    # x_norm = (points_undistorted[:, 0] - cu) / fu
    # y_norm = (points_undistorted[:, 1] - cv) / fv

    # points_norm = np.array([[x_norm, y_norm]])
    # mapped_points = map_undistorted_to_distorted(points_undistorted, intrinsics[0], distortion_params)
    # print("Mapped points (undistorted to distorted):", mapped_points)
    
    # # plot first rectified image
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(rectified_images[0])
    # # plot points undistorted on image
    # plt.scatter(points_undistorted[:, 0], points_undistorted[:, 1], c='r', s=50, label='Undistorted Points')
    # plt.subplot(1, 2, 2)
    # plt.imshow(frames[0])
    # # plot mapped points distorted on image
    # plt.scatter(mapped_points[:, 0], mapped_points[:, 1], c='b', s=50, label='Distorted Points')
    # # plt.title("First Rectified Image")
    # plt.show()
    
    # print("intrinsics:", intrinsics[0])
    
    # exit(0)
    
    data = {
        "depth_maps": depth_maps,
        "rectified_images": rectified_images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics
    }
    
    return data


def map_undistorted_to_distorted(points_2d, K, aria_params):
    pass