import copy
import numpy as np
import math
import numpy as np
import einops
import tqdm
from typing import List, Optional
import cv2
from matplotlib.collections import LineCollection
import matplotlib
import open3d as o3d
import matplotlib.pyplot as plt

def sample_depth_from_2d_points(points2d, depth_map):
    """
    Samples depth values from a depth map at specified 2D points.

    Args:
        points2d (np.ndarray): The 2D points in image coordinates, shape (N, 2).
        depth_map (np.ndarray): The depth map of shape (H, W).
    Returns:
        np.ndarray: The sampled depth values at the specified 2D points, shape (N,).
    """
    imh, imw = depth_map.shape

    # Extract pixel coordinates
    u = points2d[:, 0]
    v = points2d[:, 1]

    # Clip coordinates to be within image bounds
    u_clipped = np.clip(np.round(u).astype(int), 0, imw - 1)
    v_clipped = np.clip(np.round(v).astype(int), 0, imh - 1)

    # Sample depth values
    sampled_depth = depth_map[v_clipped, u_clipped]

    return sampled_depth

def project_points_3d_to_2d(points3d, K, pose_c2w):
    """
    Projects 3D points into 2D image coordinates using the intrinsic matrix K
    and the camera-to-world pose.

    Args:
        points3d (np.ndarray): The 3D points in world coordinates, shape (N, 3).
        K (np.ndarray): The 3x3 intrinsic camera matrix.
        pose_c2w (np.ndarray): The 4x4 camera-to-world transformation matrix.

    Returns:
        np.ndarray: The projected 2D points in image coordinates, shape (N, 2).
    """
    # Convert points3d to homogeneous coordinates
    N = points3d.shape[0]
    points3d_hom = np.concatenate((points3d, np.ones((N, 1))), axis=1)  # (N, 4)

    # Transform points from world to camera coordinates
    pose_w2c = np.linalg.inv(pose_c2w)
    points_cam_hom = (pose_w2c @ points3d_hom.T).T  # (N, 4)

    # Project points onto image plane
    points_cam = points_cam_hom[:, :3]
    points_2d_hom = (K @ points_cam.T).T  # (N, 3)

    # Normalize to get pixel coordinates
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]  # (N, 2)

    return points_2d

def unproject_points_2d_to_3d(points2d, depth, K, pose_c2w):
    
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Unproject points2d (N, 2) to camera space
    u = points2d[:, 0]
    v = points2d[:, 1]
    
    # Flatten arrays
    u = u.reshape(-1)  # (H*W,)
    v = v.reshape(-1)  # (H*W,)
    z = depth.reshape(-1)  # (H*W,)
    
    # Apply pinhole camera model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (N, 3) array
    points_3d_cam = np.stack([x, y, z], axis=1)
    # to homogeneous coordinates
    N = points_3d_cam.shape[0]
    points_3d_cam_hom = np.concatenate([points_3d_cam, np.ones((N, 1))], axis=1)  # (N, 4)
    
    # Convert points from camera to world coordinates
    N = points_3d_cam.shape[0]
    points_3d_cam_hom = np.concatenate([points_3d_cam, np.ones((N, 1))], axis=1)  # (N, 4)
    points_3d_world_hom = (pose_c2w @ points_3d_cam_hom.T).T  # (N, 4)
    points_3d_world = points_3d_world_hom[:, :3] / points_3d_world_hom[:, 3:4]  # (N, 3)
    
    return points_3d_world

def depth2xyz(depth, K):
    """
    Unprojects a 2D depth map (Z-depth) into a 3D point cloud (XYZ map)
    in camera coordinates using the intrinsic camera matrix K.
    
    Standard pinhole camera model:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z

    Args:
        depth (np.ndarray): The depth map of shape (H, W).
        K (np.ndarray): The 3x3 intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Returns:
        np.ndarray: The 3D points in camera coordinates, shape (H*W, 3).
    """
    height, width = depth.shape
    
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Create pixel coordinate grid
    # u: horizontal pixel coordinate (column index, x direction)
    # v: vertical pixel coordinate (row index, y direction)
    u, v = np.meshgrid(
        np.arange(width),   # u: 0 to width-1
        np.arange(height),  # v: 0 to height-1
        indexing='xy'
    )
    
    # Add 0.5 to get pixel centers (optional, depends on convention)
    # u = u.astype(np.float32) + 0.5
    # v = v.astype(np.float32) + 0.5
    
    # Flatten arrays
    u = u.reshape(-1)  # (H*W,)
    v = v.reshape(-1)  # (H*W,)
    z = depth.reshape(-1)  # (H*W,)
    
    # Apply pinhole camera model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (N, 3) array
    points_3d = np.stack([x, y, z], axis=1)
    
    return points_3d


def load_video_frames(video_path):
    """
    Loads an MP4 video as a sequence of frames using OpenCV.

    Returns: A list of NumPy arrays (frames) and the total frame count.
    """
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file {video_path}")

    # 2. Get the total number of frames reliably
    # The constant for frame count is CAP_PROP_FRAME_COUNT (often 7)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    # 3. Iterate and read frames
    while cap.isOpened():
        # ret (return value) is a boolean, frame is the frame itself (a NumPy array)
        ret, frame = cap.read()

        if ret:
            # Optionally convert BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Break the loop if we've reached the end of the video
            break

    # 4. Release the video capture object
    cap.release()
    return frames, frame_count


class CameraAZ:
    def __init__(
        self,
        from_json=None,
        from_jaxcam=None,
    ):
        """
        Initialize the object with either JSON data or JAX camera data.

        Parameters:
        from_json (dict, optional): A dictionary containing 'extr' and 'intr_normalized' keys.
        from_jaxcam (jaxcam, optional): Initialize from JAX camera.
        """
        if from_json is not None:
            self.extr = from_json["extr"]
            self.intr_normalized = from_json["intr_normalized"]
        elif from_jaxcam is not None:
            self._init_from_jaxcam(from_jaxcam)
        else:
            raise NotImplementedError()

    def __str__(self):
        return f"extr: \n{self.extr}\n intr_normalized: \n{self.intr_normalized}"

    def _init_from_jaxcam(self, jax_camera):
        self.extr = np.asarray(jax_camera.world_to_camera_matrix[:3])
        self.intr_normalized = {
            "fx": (jax_camera.intrinsic_matrix[0][0] / jax_camera.image_size_x).item(),
            "fy": (jax_camera.intrinsic_matrix[1][1] / jax_camera.image_size_y).item(),
            "cx": (jax_camera.intrinsic_matrix[0][2] / jax_camera.image_size_x).item(),
            "cy": (jax_camera.intrinsic_matrix[1][2] / jax_camera.image_size_y).item(),
            "k1": 0,
            "k2": 0,
        }

    def to_json_format(self):
        return {
            "extr": self.extr,
            "intr_normalized": self.intr_normalized,
        }

    def get_c2w(self):
        """
        Get the camera-to-world transformation matrix.
        Returns:
        numpy.ndarray: A 4x4 camera-to-world transformation matrix.
        """
        w2c = np.concatenate((self.extr, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w

    def get_hfov_deg(self):
        """
        Get the horizontal field of view (HFOV) in degrees.
        """
        return math.degrees(2 * np.arctan(0.5 / self.intr_normalized["fx"]))

    def get_intri_matrix(self, imh: int, imw: int):
        """
        Get the intrinsic matrix

        Parameters:
        imh (int): The height of the image.
        imw (int): The width of the image.

        Returns:
        numpy.ndarray: A 3x3 intrinsic matrix.
        """
        return np.array(
            [
                [self.intr_normalized["fx"] * imw, 0, self.intr_normalized["cx"] * imw],
                [0, self.intr_normalized["fy"] * imh, self.intr_normalized["cy"] * imh],
                [0, 0, 1],
            ]
        )

    def pix_2_world_np(
        self,
        xy: np.ndarray,
        depth: np.ndarray,
        valid_depth_min: float,
        valid_depth_max: float,
    ):
        """unproject points from ndc from to world frame.

        depth: h x w xy definition:

            xy.shape [:, 2]
            left to right: [0, w]
            top to bottom: [0, h]
        """

        _, dim = xy.shape
        assert dim == 2
        imh, imw = depth.shape

        valid_mask = (
            (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < imw) & (xy[:, 1] < imh)
        )

        x_cam = (xy[..., 0] / imw - self.intr_normalized["cx"]) / self.intr_normalized[
            "fx"
        ]
        y_cam = (xy[..., 1] / imh - self.intr_normalized["cy"]) / self.intr_normalized[
            "fy"
        ]
        z_cam = np.ones_like(xy[..., 0])
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        x_query = np.clip(np.round(xy[:, 0]).astype(int), 0, imw - 1)
        y_query = np.clip(np.round(xy[:, 1]).astype(int), 0, imh - 1)
        depth_values = depth[y_query, x_query]

        valid_mask = (
            valid_mask
            & (depth_values > valid_depth_min)
            & (depth_values < valid_depth_max)
        )

        xyz_cam = depth_values[:, None] * xyz_cam

        xyz_world = (self.extr[:3, :3].T @ (xyz_cam - self.extr[:3, 3]).T).T
        return xyz_world, valid_mask

    def world_2_pix_np(
        self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.01
    ):
        """project points from world frame to screen space.

        xyz_world: [:, 3] array of points in world frame.
        """
        xyz_world_hom = np.concatenate(
            (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
        )
        xyz_homo = (
            self.get_intri_matrix(imh, imw) @ self.extr @ xyz_world_hom.T
        ).T  # npt, 3
        depth = xyz_homo[:, 2]
        xy = xyz_homo[:, :2] / xyz_homo[:, 2:]
        valid_mask = (
            (xy[:, 0] >= 0.5)
            & (xy[:, 1] >= 0.5)
            & (xy[:, 0] < imw - 0.5)
            & (xy[:, 1] < imh - 0.5)
            & (depth > min_depth)
        )
        return xy, valid_mask, depth


class Track3d:

    def __init__(
        self,
        tracks: Optional[np.ndarray] = None,
        visibles: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        cameras: Optional[List[CameraAZ]] = None,
        video: Optional[np.ndarray] = None,
        query_points: Optional[np.ndarray] = None,
        valid_depth_min=0,
        valid_depth_max=20,
        load_from_json=None,
        track3d: Optional[np.ndarray] = None,
        visible_list: Optional[np.ndarray] = None,
        color_values: Optional[np.ndarray] = None,
    ):
        """tracks: npt x nframe x d2

        visibles: npt x nframe
        depths: nframe x imh x imw
        cameras: nframe x CameraAZ
        video: nframe x imh x imw
        query_points: npt x 3(t, h, w)
        valid_depth_min: float, min value for valid depth
        valid_depth_max: float, max value for valid depth
        """
        if load_from_json is not None:
            self._load_from_json(load_from_json)
        else:
            # sanity check
            if tracks is not None:
                npt, nframe, d2 = tracks.shape
                assert d2 == 2, "tracks dimension should be 2"
                assert (
                    npt,
                    nframe,
                ) == visibles.shape, f"Wrong shape visibles.shape {visibles.shape}, expected {npt, nframe}"  # pytype: disable=attribute-error
                assert depths.shape[0] == nframe, (  # pytype: disable=attribute-error
                    f"Wrong shape depths.shape[0] {depths.shape[0]}, expected nframe"  # pytype: disable=attribute-error
                    f" {nframe}"
                )
                _, imh, imw = depths.shape  # pytype: disable=attribute-error
                assert (
                    len(cameras) == nframe
                ), f"Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}"
                assert (
                    nframe,
                    imh,
                    imw,
                    3,
                ) == video.shape, f"Wrong shape video.shape {video.shape}, expected (nframe, imh, imw, 3) {nframe, imh, imw, 3}"  # pytype: disable=attribute-error
                if query_points is not None:
                    assert query_points.shape == (
                        npt,
                        3,
                    ), f"query_points should be npt x 3, got {query_points.shape}"

                # unproject track
                track3d = []
                visible_list = []
                for t in range(len(video)):
                    xyz_world, valid_mask = cameras[t].pix_2_world_np(
                        tracks[:, t],
                        depths[t],
                        valid_depth_min,
                        valid_depth_max,
                    )
                    visible_list.append(visibles[:, t] & valid_mask)
                    track3d.append(xyz_world)
                track3d = einops.rearrange(
                    np.stack(track3d, axis=0), "t npt d3->npt t d3"
                )
                visible_list = einops.rearrange(
                    np.stack(visible_list, axis=0), "t npt->npt t"
                )
            elif track3d is not None:
                npt, nframe = track3d.shape[:2]
                assert (
                    track3d.shape[2] == 3
                ), f"track3d should be npt x nframe x 3, got {track3d.shape}"
                assert (
                    npt,
                    nframe,
                ) == visible_list.shape, f"Wrong shape visible_list.shape {visible_list.shape}, expected {npt, nframe}"  # pytype: disable=attribute-error
                assert (
                    len(cameras) == nframe
                ), f"Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}"
                assert (
                    nframe == video.shape[0]
                ), f"Wrong shape video.shape[0] {video.shape[0]}, expected nframe {nframe}"  # pytype: disable=attribute-error
                _, imh, imw, _ = video.shape  # pytype: disable=attribute-error
            else:
                raise NotImplementedError
            if color_values is not None:
                pass
            elif query_points is not None:
                # get point color
                color_values = video[
                    query_points[:, 0],
                    query_points[:, 1].astype(int),
                    query_points[:, 2].astype(int),
                ]  # npt, 3
            else:
                color_values = None

            self.cameras = cameras
            self.track3d = track3d
            self.imh = imh
            self.imw = imw
            self.visible_list = visible_list
            self.color_values = color_values
            self.video = video

    def _load_from_json(self, load_from_json):
        if load_from_json["cameras"] is not None:
            self.cameras = [
                CameraAZ(from_json=camera)
                for camera in load_from_json["cameras"]  # FIXME
                # for camera in load_from_json['camera']
            ]
        self.track3d = np.stack(load_from_json["track3d"], axis=0)
        self.imh = load_from_json["imh"]
        self.imw = load_from_json["imw"]
        self.visible_list = load_from_json["visible_list"]
        self.color_values = load_from_json["color_values"]
        self.video = load_from_json["video"]

    def to_json_format(self, save_video=False, save_camera=True):
        return {
            "cameras": (
                [camera.to_json_format() for camera in self.cameras]
                if save_camera
                else None
            ),
            "track3d": self.track3d,
            "imh": self.imh,
            "imw": self.imw,
            "visible_list": self.visible_list,
            "color_values": self.color_values,
            "video": self.video if save_video else None,
        }

    def get_new_track(self, track_mask=None, percentage=None | float):
        """
        Generate a new track by applying a mask or a random selection based on a given percentage.

        Parameters:
        track_mask (numpy.ndarray, optional): A boolean mask array to select specific tracks. If None, a random mask will be generated.
        percentage (float, optional): The percentage of tracks to randomly select if track_mask is None. Should be a value between 0 and 1.

        Returns:
        new_track (object): A new instance of the track object with the selected tracks.
        """
        if track_mask is None:
            track_mask = np.random.uniform(size=self.track3d.shape[0]) < percentage
        new_track = copy.deepcopy(self)
        new_track.track3d = new_track.track3d[track_mask]
        new_track.visible_list = new_track.visible_list[track_mask]
        if new_track.color_values is not None:
            new_track.color_values = new_track.color_values[track_mask]
        return new_track


def get_scene_motion_2d_displacement(
    track3d: Track3d,
    tracks_leave_trace=16,
):
    """Get 2D point trajectories of scene motion.

    Returns max 2D displacement of 3D points over tracks_leave_trace frames as if
    the camera is static. This measurement decouples camera motion.

    Args:
        track3d: An instance of Track3d containing 3D tracks and visibility info.
        tracks_leave_trace: Number of frames over which to compute displacement.

    Returns:
        displacement: A (npt, nframe) array of max 2D displacements over
        tracks_leave_trace frames.
    """
    all_points = track3d.track3d  # Shape: (npt, nframe, 3)
    npt, nframe, _ = all_points.shape
    displacement = np.zeros_like(track3d.visible_list, dtype=np.float32)
    for t in tqdm.tqdm(range(nframe), desc="Computing 2D displacement"):
        s_start = max(0, t - tracks_leave_trace)
        s_end = t + 1  # Include current frame
        s_list = np.arange(s_start, s_end)  # Shape: (L,)
        L = len(s_list)
        if L < 2:
            # Not enough frames to compute displacement
            continue
        # Extract positions and visibilities for relevant frames
        positions = all_points[:, s_list, :]  # Shape: (npt, L, 3)
        visibilities = track3d.visible_list[:, s_list]  # Shape: (npt, L)
        # Flatten positions for projection
        positions_flat = positions.reshape(-1, 3)
        # Project all positions using the camera at frame t
        points_2d_flat, valid_mask_flat, _ = track3d.cameras[t].world_2_pix_np(
            positions_flat,
            track3d.imh,
            track3d.imw,
        )
        # Reshape back to (npt, L, 2)
        points_2d = points_2d_flat.reshape(npt, L, -1)
        valid_mask = valid_mask_flat.reshape(npt, L)

        # Extract positions and masks at time t
        points_2d_t = points_2d[:, -1, :]  # Shape: (npt, 2)
        valid_mask_t = valid_mask[:, -1]
        visibilities_t = visibilities[:, -1]

        # Compute displacements to previous frames
        deltas = points_2d[:, :-1, :] - points_2d_t[:, None, :]  # Shape: (npt, L-1, 2)
        distances = np.linalg.norm(deltas, axis=2)  # Shape: (npt, L-1)
        # print("distances: ", distances.shape)
        # Validity mask
        valid = (
            valid_mask[:, :-1]
            & valid_mask_t[:, None]
            & visibilities[:, :-1]
            & visibilities_t[:, None]
        )
        # Apply validity mask
        distances[~valid] = 0
        # Compute maximum displacement
        max_displacement = np.max(distances, axis=1)  # Shape: (npt,)
        displacement[:, t] = max_displacement
    return displacement


def flow_to_depth(flow: np.ndarray, hfov_deg, baseline) -> np.ndarray:
    """Calculates depth map from the flow field and camera metadata.

    assumes cx2 - cx1 = 0, valid disparity should be positive

    Args:
        flow: The optical flow field (numpy array).
        hfov_deg: Horizontal field of view (degree).
        baseline: The baseline value in meters (float).

    Returns:
        The calculated depth map (numpy array).
    """
    # disp = np.abs(flow[..., 0])
    # Extract the horizontal component (disparity)
    disp = np.clip(flow[..., 0], 0, None)
    imh, imw = disp.shape
    fx = imw / np.tan(np.radians(hfov_deg / 2)) / 2
    depth = (fx * baseline) / disp
    return depth


def gradient_check_mask_relative(depth_map, threshold):
    if depth_map.ndim == 2:
        # Case for single depth map with shape (h, w)
        padded_depth_map = np.pad(depth_map, pad_width=1, mode="edge")

        # Compute x and y gradients
        grad_x = np.abs(padded_depth_map[1:-1, 2:] - padded_depth_map[1:-1, :-2])
        grad_y = np.abs(padded_depth_map[2:, 1:-1] - padded_depth_map[:-2, 1:-1])

        # Check if any gradient exceeds the threshold and the pixel is non-zero
        mask = ((grad_x > threshold * depth_map) | (grad_y > threshold * depth_map)) & (
            depth_map != 0
        )

    elif depth_map.ndim == 3:
        # Case for batch of depth maps with shape (b, h, w)
        padded_depth_map = np.pad(
            depth_map, pad_width=((0, 0), (1, 1), (1, 1)), mode="edge"
        )

        # Compute x and y gradients
        grad_x = np.abs(padded_depth_map[:, 1:-1, 2:] - padded_depth_map[:, 1:-1, :-2])
        grad_y = np.abs(padded_depth_map[:, 2:, 1:-1] - padded_depth_map[:, :-2, 1:-1])

        # Check if any gradient exceeds the threshold and the pixel is non-zero
        mask = ((grad_x > threshold * depth_map) | (grad_y > threshold * depth_map)) & (
            depth_map != 0
        )

    else:
        raise ValueError("depth_map must have shape (h, w) or (b, h, w)")

    return mask


def disparity_to_depth(disp: np.ndarray, hfov_deg, baseline) -> np.ndarray:
    """Calculates depth map from the disparity map and camera metadata.

    assumes cx2 - cx1 = 0, valid disparity should be positive

    Args:
        disp: The disparity map (numpy array).
        hfov_deg: Horizontal field of view (degree).
        baseline: The baseline value in meters (float).

    Returns:
        The calculated depth map (numpy array).
    """
    disp = np.clip(disp, 0, None)
    imh, imw = disp.shape
    fx = imw / np.tan(np.radians(hfov_deg / 2)) / 2
    nom = fx * baseline
    depth = nom / disp

    uncertainty = np.zeros_like(depth)
    uncertainty[disp > 0] = (depth[disp > 0] ** 2) / nom

    return depth, uncertainty


def inverse_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warps an image based on the provided flow field.

    Args:
        img: The image to warp (numpy array).
        flow: The optical flow field (numpy array).

    Returns:
        The warped image (numpy array).
    """
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def plot_3d_tracks_plt(
    video: np.ndarray,
    track3d: Track3d,
    tracks_leave_trace=16,
    point_size: int = 10,
):
    """Visualize 2D point trajectories.

    The trail shows where the previous 3D point in the current camera frame. This
    visualization decouples camera motion
    """
    num_points, num_frames = track3d.track3d.shape[:2]
    figure_dpi = 64

    # Precompute colormap for points
    color_map = matplotlib.colormaps.get_cmap("hsv")
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]

    disp = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        all_points = track3d.track3d  # npt, nframe, xyz
        npt, nframe, _ = all_points.shape
        all_points = einops.rearrange(all_points, "npt nframe xyz->(npt nframe) xyz")

        points_at_frame, valid_mask, _ = track3d.cameras[t].world_2_pix_np(
            all_points,
            track3d.imh,
            track3d.imw,
        )
        points_at_frame = einops.rearrange(
            points_at_frame,
            "(npt nframe) xyz->nframe npt xyz",
            npt=npt,
            nframe=nframe,
        )
        valid_mask = einops.rearrange(
            valid_mask, "(npt nframe)-> npt nframe", npt=npt, nframe=nframe
        )
        valid_mask = valid_mask & track3d.visible_list
        valid_mask = valid_mask.transpose(1, 0)
        line_tracks = points_at_frame[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = valid_mask[max(0, t - tracks_leave_trace) : t + 1]
        fig = plt.figure(
            figsize=(frame.shape[1] / figure_dpi, frame.shape[0] / figure_dpi),
            dpi=figure_dpi,
            frameon=False,
            facecolor="w",
        )
        ax = fig.add_subplot()
        ax.axis("off")
        ax.imshow(frame / 255.0)

        for s in range(line_tracks.shape[0] - 1):
            # Collect lines and colors for the track
            visible_line_mask = (
                line_visibles[s] & line_visibles[s + 1] & line_visibles[-1]
            )
            pt1 = line_tracks[s, visible_line_mask]
            pt2 = line_tracks[s + 1, visible_line_mask]
            lines = np.concatenate([pt1, pt2], axis=1)
            lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
            c = point_colors[visible_line_mask]
            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            c = np.concatenate([c, np.ones_like(c[..., :1]) * alpha], axis=1)
            lc = LineCollection(lines, colors=c, linewidths=1)
            ax.add_collection(lc)
        visibles_mask = valid_mask[t].astype(bool)
        colalpha = point_colors[visibles_mask]
        plt.scatter(
            points_at_frame[t, visibles_mask, 0],
            points_at_frame[t, visibles_mask, 1],
            s=point_size,
            c=colalpha,
        )

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[
            ..., :3
        ]  # pytype: disable=attribute-error
        disp.append(np.copy(img))
        plt.close(fig)
        del fig, ax

    disp = np.stack(disp, axis=0)
    return disp


def load_dataset_npz(path):
    """
    Load released npz format
    """
    with open(path, "rb") as f:
        data_zip = np.load(f)
        data = {}
        for k in data_zip.keys():
            data[k] = data_zip[k]
    # --------------
    # Camera intrinsics
    # --------------
    data["meta_fov"] = {
        "start_yaw_in_degrees": data["fov_bounds"][0],
        "end_yaw_in_degrees": data["fov_bounds"][1],
        "start_tilt_in_degrees": data["fov_bounds"][2],
        "end_tilt_in_degrees": data["fov_bounds"][3],
    }
    data.pop("fov_bounds")
    # --------------
    # Camera poses
    # --------------
    c2w = data["camera2world"]  # (T, 3, 4)
    R = c2w[:, :, :3]
    t = c2w[:, :, 3:]

    # Compute inverse: R^T and new translation
    R_inv = np.transpose(R, (0, 2, 1))  # Transpose R
    t_inv = -np.matmul(R_inv, t)
    data["extrs_rectified"] = np.concatenate([R_inv, t_inv], axis=-1)
    data.pop("camera2world")
    # --------------
    # 3D tracks
    # --------------
    lengths = data["track_lengths"]
    shape = (len(lengths), len(data["timestamps"]), 3)
    tracks = np.full(shape, np.nan)
    tracks[
        np.repeat(np.arange(lengths.shape[0]), lengths), data["track_indices"], :
    ] = data["track_coordinates"]
    data["track3d"] = tracks
    data.pop("track_lengths")
    data.pop("track_indices")
    data.pop("track_coordinates")
    return data


# Define the maximum value for the target data type (uint16)
UINT16_MAX = 65535.0


def quantize_to_uint16(
    float_data: np.ndarray, X_min: float, X_max: float
) -> np.ndarray:
    """
    Converts a float32 NumPy array to a uint16 array using linear scaling.

    Args:
        float_data: The input NumPy array (float32).
        X_min: The minimum value of the original data range.
        X_max: The maximum value of the original data range.

    Returns:
        A NumPy array of type uint16.
    """
    X_range = X_max - X_min
    if X_range <= 0:
        raise ValueError("X_max must be greater than X_min.")

    # Calculate the scaling factor: (UINT16_MAX / X_range)
    scale_factor = UINT16_MAX / X_range

    # Shift: (X - X_min)
    shifted_data = float_data - X_min

    # Scale and Round: round((X - X_min) * scale_factor)
    # np.clip ensures values slightly outside the range are forced to 0 or 65535
    quantized_data = np.clip(np.round(shifted_data * scale_factor), 0, UINT16_MAX)

    # Cast to uint16
    return quantized_data.astype(np.uint16)


def unquantize_from_uint16(
    uint16_data: np.ndarray, X_min: float, X_max: float
) -> np.ndarray:
    """
    Converts a uint16 NumPy array back to an approximate float32 array.

    Args:
        uint16_data: The input NumPy array (uint16).
        X_min: The minimum value of the original data range.
        X_max: The maximum value of the original data range.

    Returns:
        A NumPy array of type float32.
    """
    X_range = X_max - X_min
    if X_range <= 0:
        raise ValueError("X_max must be greater than X_min.")

    # Calculate the inverse scaling factor: (X_range / UINT16_MAX)
    inv_scale_factor = X_range / UINT16_MAX

    # Convert to float32 for calculations and Scale: Y * inv_scale_factor
    scaled_data = uint16_data.astype(np.float32) * inv_scale_factor

    # Shift back: Y_scaled + X_min
    recovered_data = scaled_data + X_min

    return recovered_data


def radial_to_z_depth(radial_depth_map, fx, fy, cx, cy):
    """
    Convert a radial depth map r(u,v) to a z-depth map z(u,v)
    under a simple pinhole model with intrinsics (fx, fy, cx, cy).

    Parameters
    ----------
    radial_depth_map : (H, W) np.ndarray
        Array of radial depths, in float format.
    fx, fy : float
        Focal lengths of the camera.
    cx, cy : float
        Principal point (image center) in pixel coordinates.

    Returns
    -------
    z_depth_map : (H, W) np.ndarray
        The z-depth map corresponding to the input radial depths.
    """

    assert fx is not None, "Focal length fx is not specified"
    assert fy is not None, "Focal length fy is not specified"
    assert cx is not None, "Principal point cx is not specified"
    assert cy is not None, "Principal point cy is not specified"

    H, W = radial_depth_map.shape[:2]

    # Create a grid of pixel coordinates
    # row_indices ~ v, col_indices ~ u
    row_indices, col_indices = np.indices((H, W))

    # Convert from pixel coords to normalized camera-plane coords
    x_norm = (col_indices - cx) / fx
    y_norm = (row_indices - cy) / fy

    # denominator = sqrt(x_norm^2 + y_norm^2 + 1)
    denom = np.sqrt(x_norm**2 + y_norm**2 + 1)

    # z = radial_depth / denom
    z_depth_map = radial_depth_map / denom

    # make sure output dtype is the same as input dtype
    z_depth_map = z_depth_map.astype(radial_depth_map.dtype)

    return z_depth_map
