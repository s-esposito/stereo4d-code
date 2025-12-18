import numpy as np
import os
import cv2
import mediapy as media
from pathlib import Path
from tapvid3d.utils import (
    project_points_to_video_frame,
    plot_2d_tracks,
    plot_3d_tracks,
    plot_camera_trajectory,
    load_data,
    view_with_open3d_viewer,
)

# load tapvid3d dataset tracks 3D
NUM_TRACKS = 300


def main(data_path: Path, scene_name: str, save_path: Path):

    # Parse and examine contents of the dataset example file
    video, tracks_xyz, visibility, intrinsics, extrinsics_w2c = load_data(data_path, scene_name)

    if tracks_xyz.shape[1] > NUM_TRACKS:
        indices = np.random.choice(tracks_xyz.shape[1], NUM_TRACKS, replace=False)
        tracks_xyz = tracks_xyz[:, indices]
        visibility = visibility[:, indices]

    # View with Open3D viewer
    view_with_open3d_viewer(video, tracks_xyz, visibility, intrinsics, extrinsics_w2c)
    
    # # Visualize 2D point trajectories

    # # Project to 2D in pixel coordinates
    # tracks_xy, infront_cameras = project_points_to_video_frame(
    #     tracks_xyz, intrinsics, video.shape[1], video.shape[2]
    # )
    # print(f"  tracks_xy: {tracks_xy.shape}")
    # print(f"  infront_cameras: {infront_cameras.shape}")

    # video2d_viz = plot_2d_tracks(
    #     video, tracks_xy, visibility, infront_cameras, show_occ=True
    # )  # (T, H, W, 3)
    # media.write_video(
    #     save_path / "tapvid3d_2d_point_tracks_viz.mp4", video2d_viz, fps=24
    # )

    # # Visualize 3D point trajectories (takes a long time if there are lots of trajectories...)!
    # video3d_viz = plot_3d_tracks(tracks_xyz, visibility, infront_cameras, show_occ=True)
    # media.write_video(
    #     save_path / "tapvid3d_3d_point_tracks_viz.mp4", video3d_viz, fps=24
    # )

    # # Visualize camera extrinsics if available

    # if extrinsics_w2c is not None:
    #     extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
    #     extrinsics_plot_video = plot_camera_trajectory(
    #         camera_rotations=extrinsics_c2w[:, :3, :3],
    #         camera_positions=extrinsics_c2w[:, :3, -1],
    #     )
    #     media.write_video(
    #         save_path / "tapvid3d_camera_trajectory_viz.mp4",
    #         extrinsics_plot_video,
    #         fps=24,
    #     )


if __name__ == "__main__":

    tapvid3d_dir = "/media/stefano/0D91176038319865/data/tapvid3d_dataset/pstudio"

    # list all .npz files in dataset_dir

    npz_files = [f for f in os.listdir(tapvid3d_dir) if f.endswith(".npz")]
    print(f"Found {len(npz_files)} .npz files in {tapvid3d_dir}")

    # get first file
    data_path = Path(tapvid3d_dir) / npz_files[0]
    scene_name = npz_files[0].replace(".npz", "")
    print(f"Loading first file: {data_path}")

    save_path = Path("./outputs/tapvid3d_viz")
    save_path.mkdir(parents=True, exist_ok=True)

    main(data_path, scene_name, save_path)
