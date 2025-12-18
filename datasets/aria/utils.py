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
    cv_imgs = data["rectified_images"]  # (nr_frames, H, W, 3)
    rgb_imgs = data["undistorted_images"]  # (nr_frames, H, W, 3)
    rgbs = (cv_imgs, rgb_imgs)
    K = data["intrinsics"]  # tuple of (cv_K, rgb_K)
    K_cv = K[0]
    K_rgb = K[1]
    poses_c2w = data["extrinsics"]  # tuple of (cv_poses, rgb_poses)
    cv_poses_c2w = poses_c2w[0]
    rgb_poses_c2w = poses_c2w[1]
    
    point_clouds = []
    for fid in range(nr_frames):
        
        pcd = utils.generate_point_cloud(cv_imgs[fid], depths[fid], K_cv, cv_poses_c2w[fid])
        
        xyz = pcd["xyz"]
        
        # project to rgb camera
        rgb_points_2d = utils.project_points_3d_to_2d(xyz, K_rgb, rgb_poses_c2w[fid])
        
        # sample rgb image at projected points to get colors
        rgb_img = rgb_imgs[fid]
        h, w = rgb_img.shape[:2]
        
        # Extract pixel coordinates
        u = rgb_points_2d[:, 0]
        v = rgb_points_2d[:, 1]
        
        # Create mask for valid points (within image bounds and in front of camera)
        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # Clip coordinates to be within image bounds for sampling
        u_clipped = np.clip(np.round(u).astype(int), 0, w - 1)
        v_clipped = np.clip(np.round(v).astype(int), 0, h - 1)
        
        # Sample RGB colors from the image
        sampled_colors = rgb_img[v_clipped, u_clipped]  # (N, 3)
        
        # Update point cloud colors, but only for valid points
        pcd["rgb"][valid_mask] = sampled_colors[valid_mask]

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
    
    # https://github.dev/facebookresearch/projectaria_tools/blob/08a177a445e39bbf577f399ebaaf8f3eba490639/core/calibration/camera_projections/FisheyeRadTanThinPrism.h#L58
    
    # Load VRS file
    vrs_file_path = os.path.join(scene_dir, "video.vrs")
    
    from projectaria_tools.core import data_provider
    from projectaria_tools.core import calibration
    from projectaria_tools.core.image import InterpolationMethod

    # Load VRS file
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file_path)

    # Obtain device calibration
    device_calib = vrs_data_provider.get_device_calibration()
    if device_calib is None:
        raise RuntimeError(
            "device calibration does not exist! Please use a VRS that contains valid device calibration for this tutorial. "
        )

    # You can obtain device version (Aria Gen1 vs Gen2), or device subtype (DVT with small/large frame width + short/long temple arms, etc) information from calibration
    if device_calib is not None:
        device_version = device_calib.get_device_version()
        device_subtype = device_calib.get_device_subtype()

        print("Obtained valid calibration: ")
        print(f"Device Version: {calibration.get_name(device_version)}")
        print(f"Device Subtype: {device_subtype}")
        
    # Get sensor labels within device calibration
    all_labels = device_calib.get_all_labels()
    print(f"All sensors within device calibration: {all_labels}")
    print(f"Cameras: {device_calib.get_camera_labels()}")

    # retrieve camera to device CV transformation
    # cv_sensor_name = "camera-cv"

    # input: retrieve image as a numpy array
    sensor_name = "camera-rgb"
    # sensor_stream_id = vrs_data_provider.get_stream_id_from_label(sensor_name)
    # image_data = vrs_data_provider.get_image_data_by_index(sensor_stream_id, 0)
    # image_array = image_data[0].to_numpy_array()
    # input: retrieve image distortion
    device_calib = vrs_data_provider.get_device_calibration()
    
    # # get device transformation
    # T_device_camera = device_calib.get_transform_device_camera()
    # print("T_device_camera:\n", T_device_camera.to_matrix())
    # exit(0)
    
    camera_calib = device_calib.get_camera_calib(sensor_name)
    
    rgb_camera_focal_lengths = camera_calib.get_focal_lengths()
    rgb_camera_principal_point = camera_calib.get_principal_point()
    rgb_camera_intrinsics = np.array([[rgb_camera_focal_lengths[0], 0, rgb_camera_principal_point[0]],
                                        [0, rgb_camera_focal_lengths[1], rgb_camera_principal_point[1]],
                                        [0, 0, 1]])
    
    # downscale intrinsics by factor
    rgb_downscale_factor = 4
    rgb_camera_intrinsics[0, 0] /= rgb_downscale_factor
    rgb_camera_intrinsics[1, 1] /= rgb_downscale_factor
    rgb_camera_intrinsics[0, 2] /= rgb_downscale_factor
    rgb_camera_intrinsics[1, 2] /= rgb_downscale_factor
    
    if camera_calib is None:
        raise RuntimeError(
            "camera-rgb calibration does not exist! Please use a VRS that contains valid RGB camera calibration for this tutorial. "
        )

    print(f"-------------- camera calibration for {sensor_name} ----------------")
    print(f"Image Size: {camera_calib.get_image_size()}")
    print(f"Camera Model Type: {camera_calib.get_model_name()}")
    print(
        f"Camera Intrinsics Params: {rgb_camera_intrinsics}, \n"
        f"where focal is {camera_calib.get_focal_lengths()}, "
        f"and principal point is {camera_calib.get_principal_point()}\n"
    )

    # #
    # T_device = vrs_data_provider.get_device_calibration().get_transform_device_camera()
    # print("T_device:\n", T_device.to_matrix())
    # exit(0)
    
    # device frame is defined by originSensorLabel
    origin_label = device_calib.get_origin_label()
    T_device_to_origin = device_calib.get_camera_calib(origin_label).get_transform_device_camera()
    print(f"T_device_to_origin ({origin_label}):\n", T_device_to_origin.to_matrix())
    
    # camera pose in the device frame (device to camera transformation) device frame
    T_device_camera = camera_calib.get_transform_device_camera()
    print(f"Camera Extrinsics T_Device_Camera:\n{T_device_camera.to_matrix()}")
    
    # for label in device_calib.get_camera_labels():
    #     # get T_device (camera-slam-left) transformation
    #     T_device = device_calib.get_camera_calib(label).get_transform_device_camera()
    #     print(f"{label} T_device:\n", T_device.to_matrix())
    # exit(0)
    
    # Undistrort an image
    # create output calibration: a linear model of image example_linear_rgb_camera_model_params.
    
    # Invisible pixels are shown as black.
    example_linear_rgb_camera_model_params = [4032, 3024, 1600]
    dst_calib = calibration.get_linear_camera_calibration(example_linear_rgb_camera_model_params[0], example_linear_rgb_camera_model_params[1], example_linear_rgb_camera_model_params[2], sensor_name)

    # distort image
    # image_array = image_data[0].to_numpy_array()
    # rectified_array = calibration.distort_by_calibration(image_array, dst_calib, camera_calib, InterpolationMethod.BILINEAR)

    # # visualize input and results
    # import matplotlib.pyplot as plt
    # plt.figure()
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # fig.suptitle(f"Image undistortion (focal length = {dst_calib.get_focal_lengths()})")
    # axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
    # axes[0].title.set_text(f"sensor image ({sensor_name})")
    # axes[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    # axes[1].imshow(rectified_array, cmap="gray", vmin=0, vmax=255)
    # axes[1].title.set_text(f"undistorted image ({sensor_name})")
    # axes[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    # plt.show()
    
    # exit(0)
    
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

    # distortion_params = []
    # for calibration_data in online_calibration_data:
    #     # 'ImageSizes', 'utc_timestamp_ns', 'tracking_timestamp_us', 'ImuCalibrations', 'CameraCalibrations'
    #     rgb_image_size = calibration_data['ImageSizes'][-1]  # last is RGB
    #     rgb_calibration_data = calibration_data['CameraCalibrations'][-1]  # last is RGB
    #     print("RGB image size:", rgb_image_size)
    #     print("RGB calibration data keys:", rgb_calibration_data.keys())
    #     # print("Calibrated", rgb_calibration_data['Calibrated'])  # True
    #     # print("Projection", rgb_calibration_data['Projection'])
    #     params = rgb_calibration_data['Projection']['Params']  # FisheyeRadTanThinPrism
    #     print("Distortion parameters:", params)
    #     distortion_params.append(params)
    # distortion_params = np.array(distortion_params)  # (T, 15)
    # print("Distortion parameters shape:", distortion_params.shape)
    
    # # check if distortion params are the same for all frames
    # all_same = np.all(np.isclose(distortion_params, distortion_params[0], atol=1e-6), axis=1)
    # if np.all(all_same):
    #     print("All distortion parameters are the same for all frames.")
    # else:
    #     raise ValueError("Varying distortion parameters not supported yet.")
    # distortion_params = distortion_params[0]
    
    # load video
    video_path = os.path.join(scene_dir, "video_main_rgb.mp4")
    frames, nr_frames = utils.load_video_frames(video_path, MAX_SEQ_LEN, FORCED_FPS, FIRST_FRAME_ID)
    
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    print(f"Loaded {nr_frames} frames from video at {video_path}")
    
    # undistort frames
    # check if undistorted frames have been cached
    undistorted_dir = os.path.join(scene_dir, "undistorted")
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    undistorted_frames = []
    for fid in tqdm(range(nr_frames), desc="Undistorting frames"):
        
        # check if undistorted frame already exists
        if os.path.exists(os.path.join(undistorted_dir, f"frame_{fid:05d}.png")):
            undistorted_frame = imageio.imread(os.path.join(undistorted_dir, f"frame_{fid:05d}.png"))
            undistorted_frames.append(undistorted_frame)
            continue
        else:
            frame = frames[fid]
            undistorted_frame = calibration.distort_by_calibration(frame, dst_calib, camera_calib, InterpolationMethod.BILINEAR)
            undistorted_frames.append(undistorted_frame)
            # downsample to by factor
            undistorted_frame = cv2.resize(undistorted_frame, (undistorted_frame.shape[1] // rgb_downscale_factor, undistorted_frame.shape[0] // rgb_downscale_factor), interpolation=cv2.INTER_LINEAR)
            # save undistorted frame
            imageio.imwrite(os.path.join(undistorted_dir, f"frame_{fid:05d}.png"), undistorted_frame)
            
    undistorted_frames = np.stack(undistorted_frames, axis=0)  # (T, H, W, 3)
    
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
    # intrinsics = np.stack(intrinsics_list, axis=0)  # (T, 3, 3)
    intrinsics = intrinsics_list[0]  # assume all the same
    extrinsics = np.stack(extrinsics_list, axis=0)  # (T, 4, 4)
    print(f"Loaded intrinsics shape: {intrinsics.shape}")
    print(f"Loaded extrinsics (camera-to-world) shape: {extrinsics.shape}")
    
    # # print(extrinsics[0])
    # exit(0)
    
    # re-base all extrinsics such that the first frame is at the origin
    rebase = False
    if rebase:
        T0_inv = np.linalg.inv(extrinsics[0])
        extrinsics = T0_inv @ extrinsics
    
    rgb_camera_extrinsics = []
    for c2w in extrinsics:
        # apply T_device_camera to get device to world
        T_device_world = c2w @ np.linalg.inv(T_device_camera.to_matrix())
        # T_device_world = c2w @ T_device_camera.to_matrix()
        # T_device_world = np.linalg.inv(T_device_camera.to_matrix()) @ c2w
        rgb_camera_extrinsics.append(T_device_world)
    rgb_camera_extrinsics = np.stack(rgb_camera_extrinsics, axis=0)  # (T, 4, 4)
    print(f"Computed RGB camera extrinsics shape: {rgb_camera_extrinsics.shape}")
    
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
    
    data = {
        "depth_maps": depth_maps,
        "rectified_images": rectified_images,
        "undistorted_images": undistorted_frames,
        "intrinsics": (intrinsics, rgb_camera_intrinsics),  # cv and rgb camera intrinsics
        "extrinsics": (extrinsics, rgb_camera_extrinsics),  # cv and rgb camera extrinsics
    }
    
    return data