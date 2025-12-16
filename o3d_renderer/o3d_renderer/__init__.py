"""
O3D Renderer - Open3D-based point cloud visualization library

This package provides tools for rendering point clouds with Open3D,
supporting both online (interactive GUI) and offline (headless) rendering.
"""

from .renderer import Renderer
from .online_renderer import OnlineRendererApp, run_open3d_viewer
from .offline_renderer import OffscreenRendererApp, run_open3d_offline_renderer
from .geometry_utils import (
    toOpen3dCloud,
    create_coordinate_frame,
    create_grid,
    create_camera_frustum,
    create_camera_trajectory,
    create_track_lines,
)
from .color_utils import (
    compute_tracks_colors,
    compute_instances_colors,
)

__version__ = "0.1.0"

__all__ = [
    "Renderer",
    "OnlineRendererApp",
    "OffscreenRendererApp",
    "run_open3d_viewer",
    "run_open3d_offline_renderer",
    "toOpen3dCloud",
    "create_coordinate_frame",
    "create_grid",
    "create_camera_frustum",
    "create_camera_trajectory",
    "create_track_lines",
    "compute_tracks_colors",
    "compute_instances_colors",
]
