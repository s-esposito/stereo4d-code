import numpy as np
import os
import cv2
import mediapy as media
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation as R
from o3d_renderer import run_open3d_viewer
from o3d_renderer import run_open3d_offline_renderer


def view_with_open3d_viewer(rgbs, tracks_xyz, visibility, intrinsics, extrinsics_w2c):
    
    nr_frames = len(rgbs)
    depths = None
    point_clouds = None
    instances_masks = None
    
    tracks3d = tracks_xyz.copy()  # (T, N, 3)
    # make nan invisible points
    not_visible_mask = ~visibility  # (T, N)
    tracks3d[not_visible_mask] = np.nan
    
    # permute to (N, T, 3)
    tracks3d = np.transpose(tracks3d, (1, 0, 2))
    
    # print("tracks3d shape:", tracks3d.shape, tracks3d.dtype, np.nanmin(tracks3d), np.nanmax(tracks3d))
    # print("rgbs shape:", rgbs.shape)
    # print("tracks_xyz shape:", tracks_xyz.shape)
    # print("visibility shape:", visibility.shape)
        
    K = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ], dtype=np.float32)  # (3, 3)
    # print("Camera intrinsics K:\n", K)
    
    poses_c2w = None
    if extrinsics_w2c is not None:
        poses_c2w = np.linalg.inv(extrinsics_w2c)  # (T, 4, 4)
        # print("Camera extrinsics c2w shape:", poses_c2w.shape, poses_c2w.dtype)
    
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

def load_data(data_path: Path, scene_name: str, num_tracks: int=300):
    
    with open(data_path, "rb") as in_f:
        in_npz = np.load(in_f)
        images_jpeg_bytes = in_npz["images_jpeg_bytes"]
        queries_xyt = in_npz["queries_xyt"]
        tracks_xyz = in_npz["tracks_XYZ"]
        visibility = in_npz["visibility"]
        intrinsics = in_npz["fx_fy_cx_cy"]
        if "extrinsics_w2c" in in_npz.files:
            extrinsics_w2c = in_npz["extrinsics_w2c"]
        else:
            extrinsics_w2c = None

    video = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        video.append(image_rgb)
    video = np.stack(video, axis=0)  # (T, H, W, 3)

    print(f"In example {scene_name}:")
    print(
        f"  images_jpeg_bytes: {len(images_jpeg_bytes)} frames, each stored as JPEG bytes (and after decoding, the video shape: {video.shape})"
    )
    print(f"  intrinsics: (fx, fy, cx, cy)={intrinsics}", intrinsics.dtype)
    print(f"  tracks_xyz: {tracks_xyz.shape}", tracks_xyz.dtype)
    print(f"  visibility: {visibility.shape}", visibility.dtype)
    print(f"  queries_xyt: {queries_xyt.shape}", queries_xyt.dtype)
    if extrinsics_w2c is not None:
        print(f"  extrinsics_w2c: {extrinsics_w2c.shape}", extrinsics_w2c.dtype)

    # Sort points by their height in 3D for rainbow visualization

    sorted_indices = np.argsort(tracks_xyz[0, ..., 1])  # Sort points over height
    tracks_xyz = tracks_xyz[:, sorted_indices]
    visibility = visibility[:, sorted_indices]
    
    return video, tracks_xyz, visibility, intrinsics, extrinsics_w2c

def project_points_to_video_frame(
    camera_pov_points3d, camera_intrinsics, height, width
):
    """Project 3d points to 2d image plane."""
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

    f_u, f_v, c_u, c_v = camera_intrinsics

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Mask of points that are in front of the camera and within image boundary
    masks = camera_pov_points3d[..., 2] >= 1
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks


def plot_2d_tracks(
    video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False
):
    """Visualize 2D point trajectories."""
    num_frames, num_points = points.shape[:2]

    # Precompute colormap for points
    color_map = matplotlib.colormaps.get_cmap("hsv")
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3] * 255

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    frames = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
        line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
        for s in range(line_tracks.shape[0] - 1):
            img = frame.copy()

            for i in range(num_points):
                if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(
                        round(line_tracks[s, i, 1])
                    )
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
                        round(line_tracks[s + 1, i, 1])
                    )
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
                elif (
                    show_occ
                    and line_infront_cameras[s, i]
                    and line_infront_cameras[s + 1, i]
                ):  # occluded
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(
                        round(line_tracks[s, i, 1])
                    )
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
                        round(line_tracks[s + 1, i, 1])
                    )
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

            # Draw end points on the frame
            for i in range(num_points):
                if visibles[t, i]:  # visible
                    x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                    cv2.circle(frame, (x, y), 2, point_colors[i], -1)
                elif show_occ and infront_cameras[t, i]:  # occluded
                    x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                    cv2.circle(frame, (x, y), 2, point_colors[i], 1)

        frames.append(frame)
    frames = np.stack(frames)
    return frames


def plot_3d_tracks(
    points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False
):
    """Visualize 3D point trajectories."""
    num_frames, num_points = points.shape[0:2]

    color_map = matplotlib.colormaps.get_cmap("hsv")
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    if show_occ:
        x_min, x_max = np.min(points[infront_cameras, 0]), np.max(
            points[infront_cameras, 0]
        )
        y_min, y_max = np.min(points[infront_cameras, 2]), np.max(
            points[infront_cameras, 2]
        )
        z_min, z_max = np.min(points[infront_cameras, 1]), np.max(
            points[infront_cameras, 1]
        )
    else:
        x_min, x_max = np.min(points[visibles, 0]), np.max(points[visibles, 0])
        y_min, y_max = np.min(points[visibles, 2]), np.max(points[visibles, 2])
        z_min, z_max = np.min(points[visibles, 1]), np.max(points[visibles, 1])

    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_min = (x_min + x_max) / 2 - interval / 2
    x_max = x_min + interval
    y_min = (y_min + y_max) / 2 - interval / 2
    y_max = y_min + interval
    z_min = (z_min + z_max) / 2 - interval / 2
    z_max = z_min + interval

    frames = []
    for t in range(num_frames):
        fig = Figure(figsize=(6.4, 4.8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.invert_zaxis()
        ax.view_init()

        for i in range(num_points):
            if visibles[t, i] or (show_occ and infront_cameras[t, i]):
                color = color_map(cmap_norm(i))
                line = points[max(0, t - tracks_leave_trace) : t + 1, i]
                ax.plot(
                    xs=line[:, 0],
                    ys=line[:, 2],
                    zs=line[:, 1],
                    color=color,
                    linewidth=1,
                )
                end_point = points[t, i]
                ax.scatter(
                    xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3
                )

        fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        fig.canvas.draw()
        frames.append(canvas.buffer_rgba())

    return np.array(frames)[..., :3]


def plot_camera_trajectory(
    camera_rotations,
    camera_positions,
    plot3d_elev=30,
    plot3d_azim=10,
    resolution=(256, 256),
):
    num_frames = camera_positions.shape[0]

    # Convert quaternions to rotation matrices
    rotations = R.from_matrix(camera_rotations)
    camera_directions = rotations.apply(
        np.array([0, 0, -1])
    )  # assuming looking forward along -Z

    x_range = [min(camera_positions[..., 0]), max(camera_positions[..., 0])]
    y_range = [min(camera_positions[..., 1]), max(camera_positions[..., 1])]
    z_range = [min(camera_positions[..., 2]), max(camera_positions[..., 2])]

    differences = np.diff(camera_positions, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    trajectory_length = np.sum(distances)
    quiver_len = trajectory_length * 0.001

    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)

    # Set up the plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=plot3d_elev, azim=plot3d_azim)

    # Prepare frames for video
    frames = []
    for t in range(num_frames):
        ax.cla()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Plot trajectory up to current frame
        ax.plot(
            camera_positions[: t + 1, 0],
            camera_positions[: t + 1, 1],
            camera_positions[: t + 1, 2],
            "b-",
            label="Camera Trajectory",
        )

        # Plot camera position and orientation
        ax.quiver(
            camera_positions[t, 0],
            camera_positions[t, 1],
            camera_positions[t, 2],
            camera_directions[t, 0],
            camera_directions[t, 1],
            camera_directions[t, 2],
            color="r",
            length=quiver_len,
            normalize=True,
            label="Camera Orientation",
        )

        # Capture the frame
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        try:
            rgb_array = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(height, width, 3)
        except:
            rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                height, width, 4
            )[:, :, :3]
        frames.append(rgb_array.copy())
    plt.close()
    return np.stack(frames)