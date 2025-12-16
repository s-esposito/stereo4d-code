"""Geometry utility functions for Open3D visualization."""

import numpy as np
import open3d as o3d


def toOpen3dCloud(points, colors=None, normals=None):
    """
    Convert numpy arrays to Open3D point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255 or 0-1)
        normals: (N, 3) array of normal vectors
    
    Returns:
        Open3D PointCloud object
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        # Check if colors array is not empty before accessing max()
        if colors.size > 0 and colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def create_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    """
    Create a coordinate frame showing X (red), Y (green), Z (blue) axes.
    
    Args:
        size: Length of each axis arrow
        origin: Origin point of the coordinate frame
    
    Returns:
        Open3D TriangleMesh representing the coordinate frame
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin
    )


def create_grid(size=10.0, n=10, plane='xy', height=0.0):
    """
    Create a grid in the specified plane.
    
    Args:
        size: Total size of the grid
        n: Number of grid lines in each direction
        plane: Plane to create grid in ('xy', 'xz', or 'yz')
        height: Height offset perpendicular to the plane
    
    Returns:
        Open3D LineSet representing the grid
    """
    points = []
    lines = []
    step = size / n
    start = -size / 2
    
    if plane == 'xy':
        # Grid in XY plane (perpendicular to Z)
        for i in range(n + 1):
            coord = start + i * step
            # Lines parallel to X axis
            points.append([start, coord, height])
            points.append([start + size, coord, height])
            lines.append([len(points) - 2, len(points) - 1])
            # Lines parallel to Y axis
            points.append([coord, start, height])
            points.append([coord, start + size, height])
            lines.append([len(points) - 2, len(points) - 1])
    elif plane == 'xz':
        # Grid in XZ plane (perpendicular to Y)
        for i in range(n + 1):
            coord = start + i * step
            # Lines parallel to X axis
            points.append([start, height, coord])
            points.append([start + size, height, coord])
            lines.append([len(points) - 2, len(points) - 1])
            # Lines parallel to Z axis
            points.append([coord, height, start])
            points.append([coord, height, start + size])
            lines.append([len(points) - 2, len(points) - 1])
    elif plane == 'yz':
        # Grid in YZ plane (perpendicular to X)
        for i in range(n + 1):
            coord = start + i * step
            # Lines parallel to Y axis
            points.append([height, start, coord])
            points.append([height, start + size, coord])
            lines.append([len(points) - 2, len(points) - 1])
            # Lines parallel to Z axis
            points.append([height, coord, start])
            points.append([height, coord, start + size])
            lines.append([len(points) - 2, len(points) - 1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    # Set grid color to gray
    colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def create_camera_frustum(c2w, K, size=0.2, color=[0, 0, 0]):
    """
    Create a camera frustum visualization from a camera-to-world matrix.
    
    Args:
        c2w: 4x4 camera-to-world transformation matrix
        K: 3x3 intrinsic camera matrix
        size: Size of the frustum
        color: RGB color for the frustum lines
    
    Returns:
        Open3D LineSet representing the camera frustum
    """
    # Frustum in image space (pixels)
    points_2d_screen = np.array([
        [0, 0],  # Bottom-left
        [K[0, 2] * 2, 0],  # Bottom-right
        [K[0, 2] * 2, K[1, 2] * 2],  # Top-right
        [0, K[1, 2] * 2],  # Top-left
    ])
    
    u = points_2d_screen[:, 0]
    v = points_2d_screen[:, 1]
    z = np.full(u.shape, size)
    
    # Flatten arrays
    u = u.reshape(-1)
    v = v.reshape(-1)
    z = z.reshape(-1)

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Apply pinhole camera model
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d_camera = np.stack([x, y, z], axis=1)
    
    # Append camera center to points_3d_camera
    points_3d_camera = np.vstack([np.array([[0, 0, 0]]), points_3d_camera])
    
    cam_points = points_3d_camera
    
    # Transform to world coordinates
    cam_points_hom = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])
    world_points = (c2w @ cam_points_hom.T).T[:, :3]
    
    # Define lines connecting the points
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Image plane rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    colors_list = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors_list)
    
    return line_set


def create_camera_trajectory(poses_c2w, color=[0.0, 0.0, 1.0]):
    """
    Create a line showing the camera trajectory through all poses.
    
    Args:
        poses_c2w: (T, 4, 4) array of camera-to-world matrices
        color: RGB color for the trajectory line, expected in [0.0, 1.0] range
    
    Returns:
        Open3D LineSet representing the camera trajectory
    """
    if len(poses_c2w) < 2:
        # Need at least 2 poses to form a line segment
        return o3d.geometry.LineSet()

    # Extract camera centers (translation vector is the 4th column, first 3 rows)
    centers = poses_c2w[:, :3, 3]
    
    # Create lines connecting consecutive camera positions
    points = centers
    lines = [[i, i + 1] for i in range(len(centers) - 1)]
    
    line_set = o3d.geometry.LineSet()
    
    # Convert points and lines to Open3D vector types
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines).astype(np.int32))
    
    # Create colors for each line segment
    num_lines = len(lines)
    color_np = np.array(color, dtype=np.float64).reshape(1, 3)
    colors_array = np.tile(color_np, (num_lines, 1))
    
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    
    return line_set


def create_track_lines(tracks3d, tracks_colors, current_frame, trail_length=10):
    """
    Create line visualization for 3D tracks showing trails from previous frames.
    
    Args:
        tracks3d: (N, T, 3) array of 3D track positions
        tracks_colors: (N, 3) array of RGB colors for each track
        current_frame: Current frame index
        trail_length: Number of previous frames to show in the trail
    
    Returns:
        Open3D LineSet representing the track trails or None if no valid tracks
    """
    n_tracks = tracks3d.shape[0]
    
    # Need at least 2 frames to draw lines
    if current_frame < 1:
        return None

    # Visible list by checking for non-NaN points and non-Inf points
    visible_list = ~np.isnan(tracks3d).any(axis=2) & ~np.isinf(tracks3d).any(axis=2)
    
    points = []
    lines = []
    colors = []
    
    # Determine frame range for trails (include current_frame)
    start_frame = max(0, current_frame - trail_length)
    
    for track_idx in range(n_tracks):
        track_points = []
        # Include current_frame in the range
        for frame_idx in range(start_frame, current_frame + 1):
            if visible_list[track_idx, frame_idx]:
                track_points.append(tracks3d[track_idx, frame_idx])
        
        # Only create lines if we have at least 2 visible points
        if len(track_points) >= 2:
            start_idx = len(points)
            points.extend(track_points)
            
            # Create line segments connecting consecutive points
            for i in range(len(track_points) - 1):
                lines.append([start_idx + i, start_idx + i + 1])
                color = tracks_colors[track_idx]
                colors.append(color)
    
    if len(points) == 0 or len(lines) == 0:
        return None
    
    # Ensure arrays are properly formed and have valid data
    points_array = np.array(points, dtype=np.float64)
    lines_array = np.array(lines, dtype=np.int32)
    colors_array = np.array(colors, dtype=np.float64)
    
    # Validate data
    if points_array.shape[0] < 2 or lines_array.shape[0] < 1:
        return None
    
    if np.any(np.isnan(points_array)) or np.any(np.isinf(points_array)):
        print("Warning: Invalid points detected (NaN or Inf)")
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_array)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    
    return line_set
