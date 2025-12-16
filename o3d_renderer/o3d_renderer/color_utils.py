"""Color computation utilities for tracks and instances."""

import numpy as np
import matplotlib.cm


def compute_tracks_colors(tracks3d):
    """
    Compute RGB colors for tracks based on their first valid 3D coordinate.
    
    Args:
        tracks3d: (N, T, 3) array of 3D track positions
    
    Returns:
        (N, 3) array of RGB colors in [0, 1] range
    """
    # Convert first coordinate of each track to color
    tracks_colors = np.zeros_like(tracks3d[:, 0, :])
    empty_color = np.ones_like(tracks_colors[:, 0], dtype=bool)
    
    for fid in range(tracks3d.shape[1]):
        points_at_fid = tracks3d[:, fid, :]
        valid_points_mask = ~np.isnan(points_at_fid).any(axis=1) & ~np.isinf(points_at_fid).any(axis=1)
        new_values_mask = valid_points_mask & empty_color
        tracks_colors[new_values_mask] = points_at_fid[new_values_mask]
        empty_color |= ~valid_points_mask
        # Check if all colors have been assigned
        if not np.any(empty_color):
            break
    
    # Normalize to [0, 1] range
    min_x, max_x = np.min(tracks_colors[:, 0]), np.max(tracks_colors[:, 0])
    min_y, max_y = np.min(tracks_colors[:, 1]), np.max(tracks_colors[:, 1])
    min_z, max_z = np.min(tracks_colors[:, 2]), np.max(tracks_colors[:, 2])
    
    tracks_colors[:, 0] = (tracks_colors[:, 0] - min_x) / (max_x - min_x + 1e-8)
    tracks_colors[:, 1] = (tracks_colors[:, 1] - min_y) / (max_y - min_y + 1e-8)
    tracks_colors[:, 2] = (tracks_colors[:, 2] - min_z) / (max_z - min_z + 1e-8)
    
    return tracks_colors


def compute_instances_colors(instances_masks):
    """
    Compute distinct RGB colors for each instance ID.
    
    Args:
        instances_masks: (T, H, W) array of instance IDs
    
    Returns:
        (num_instances+1, 3) array of RGB colors in [0, 255] range (uint8)
    """
    instance_colors = []
    # Append black for background (instance 0)
    instance_colors.append(np.array([0, 0, 0]))
    
    # Instances start from 1
    num_instances = np.max(instances_masks)
    cmap = matplotlib.cm.get_cmap('tab20', num_instances)
    
    for i in range(num_instances):
        color = cmap(i)[:3]  # Get RGB color
        instance_colors.append(np.array(color))
    
    instance_colors = np.array(instance_colors) * 255.0
    instance_colors = instance_colors.astype(np.uint8)
    
    return instance_colors


def srgb_to_linear(srgb_img_u8: np.ndarray) -> np.ndarray:
    """
    Converts an sRGB image (numpy array, uint8 in [0, 255]) to a linear 
    color space image (numpy array, float32 in [0.0, 1.0]).
    """
    # 1. Normalize to [0, 1] and convert to float32
    srgb_normalized = srgb_img_u8.astype(np.float32) / 255.0

    # 2. Apply the sRGB to Linear conversion formula
    
    # Define the linear and power functions
    # For S <= 0.04045, Linear = S / 12.92
    # For S > 0.04045, Linear = ((S + 0.055) / 1.055)**2.4
    
    # Create a mask for the linear part of the function
    mask = srgb_normalized <= 0.04045
    
    # Initialize the linear image array
    linear_img = np.empty_like(srgb_normalized, dtype=np.float32)
    
    # Apply the linear part
    linear_img[mask] = srgb_normalized[mask] / 12.92
    
    # Apply the power part
    s_pow = srgb_normalized[~mask]
    linear_img[~mask] = np.power((s_pow + 0.055) / 1.055, 2.4)
    
    # Convert back to u8
    linear_img = (linear_img * 255.0).astype(np.uint8)

    return linear_img