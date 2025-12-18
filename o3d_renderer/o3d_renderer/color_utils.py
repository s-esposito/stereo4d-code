"""Color computation utilities for tracks and instances."""

import numpy as np
import matplotlib.cm


def compute_tracks_colors(tracks3d):
    """
    Compute RGB colors for tracks using perceptually uniform colormap.
    Colors are based on spatial distribution of first valid 3D coordinate.
    
    Args:
        tracks3d: (N, T, 3) array of 3D track positions
    
    Returns:
        (N, 3) array of RGB colors in [0, 1] range
    """
    n_tracks = tracks3d.shape[0]
    
    # Get first valid coordinate for each track
    first_coords = np.zeros((n_tracks, 3))
    has_valid = np.zeros(n_tracks, dtype=bool)
    
    for fid in range(tracks3d.shape[1]):
        points_at_fid = tracks3d[:, fid, :]
        valid_points_mask = ~np.isnan(points_at_fid).any(axis=1) & ~np.isinf(points_at_fid).any(axis=1)
        new_values_mask = valid_points_mask & ~has_valid
        first_coords[new_values_mask] = points_at_fid[new_values_mask]
        has_valid |= valid_points_mask
        
        # Early exit if all tracks have valid coordinates
        if np.all(has_valid):
            break
    
    # Use perceptually uniform colormap (viridis) based on spatial hashing
    # Combine XYZ into a single value for colormap lookup
    spatial_hash = np.linalg.norm(first_coords, axis=1)
    
    # Normalize to [0, 1]
    min_val, max_val = np.min(spatial_hash), np.max(spatial_hash)
    normalized_hash = (spatial_hash - min_val) / (max_val - min_val + 1e-8)
    
    # Apply colormap for better visual distinction
    cmap = matplotlib.cm.get_cmap('turbo')
    tracks_colors = cmap(normalized_hash)[:, :3]  # Take RGB, discard alpha
    
    # Boost saturation for better visibility
    # Convert to HSV, increase saturation, convert back
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    tracks_colors_hsv = rgb_to_hsv(tracks_colors.reshape(-1, 1, 3))
    tracks_colors_hsv[:, :, 1] = np.clip(tracks_colors_hsv[:, :, 1] * 1.3, 0, 1)  # Increase saturation
    tracks_colors = hsv_to_rgb(tracks_colors_hsv).reshape(-1, 3)
    
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