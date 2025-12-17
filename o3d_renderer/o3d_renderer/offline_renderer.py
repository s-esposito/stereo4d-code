"""
Offline (headless) renderer for batch rendering of point clouds and tracks.
"""

import numpy as np
import cv2
from tqdm import tqdm
from open3d.visualization.rendering import OffscreenRenderer

from .renderer import Renderer
from .color_utils import srgb_to_linear


class OffscreenRendererApp(Renderer):
    """Headless renderer for batch rendering without GUI."""
    
    def __init__(
        self,
        width: int,
        height: int,
        nr_frames: int,
        rgbs: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        depths: np.ndarray | None = None,
        point_clouds: list | None = None,
        K: np.ndarray | None = None,
        poses_c2w: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        tracks3d: np.ndarray | None = None,
        instances_masks: np.ndarray | None = None,
        max_tracks: int = 3000
    ):
        super().__init__(nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks, max_tracks)
        
        # Create offscreen renderer
        self.height = height
        self.width = width
        self.o3d_renderer = OffscreenRenderer(width, height)
        
        # Setup Open3D Renderer
        self._setup_o3d_renderer()
        
        # Add coordinate frame at world origin
        self._init_coord_frame()
        
        # Add grid on XZ plane (ground plane, perpendicular to Y axis)
        self._init_grid_xz()
        
        # Init first frame
        self._init_frame(fid=0)
        
        # Set Camera View
        self.camera_distance = 8.0
        
        # Position the camera
        look_at = np.array([0.0, 0.0, 6.0])
        origin = np.array([0.0, 0.0, 0.0])
        eye = origin + np.array([2, -2, -2])  # Fixed position for better view
        self.up = np.array([0, 1, 0])  # Standard Up vector for world coordinates

        # Set the camera view
        self.o3d_renderer.setup_camera(
            60.0,      # vertical_field_of_view
            look_at,   # The point the camera looks at
            eye,       # The position of the camera
            self.up    # The vector defining 'up' for the camera
        )
        
    def render_frames(self) -> list[np.ndarray]:
        """Render all frames and return list of rendered images."""
        
        def render_image():
            """Render a single image from the current scene state."""
            # Render the image
            image = self.o3d_renderer.render_to_image()
            # Convert to numpy array and process in one go
            image = np.asarray(image)
            # Combined flip operation (more efficient than two separate flips)
            image = np.flipud(np.fliplr(image))
            # sRGB to linear color space conversion
            image = srgb_to_linear(image)
            # Only resize if needed
            if self.width != 512 or self.height != 512:
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            return image
        
        # Render keyframes (only once)
        
        # Clean scene
        self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        self.o3d_renderer.scene.remove_geometry(self.PCD_NAME)
        
        self.render_keyframes = True
        self.render_segmentation = False
        self.render_tracks = False
        self.render_time_color_coded = False
        self.render_bboxes = False
        
        self._clean_keyframes_from_scene()
        self._init_keyframes()
        
        keyframe_image = render_image()  # uint8 [0,255]
        
        # Render keyframes (time color coded, only once)
        
        self.render_keyframes = True
        self.render_segmentation = False
        self.render_tracks = False
        self.render_time_color_coded = True
        self.render_bboxes = False
        
        self._clean_keyframes_from_scene()
        self._init_keyframes()
        
        keyframe_time_color_coded_image = render_image()   # uint8 [0,255]
        
        # Concat keyframes renders vertically
        keyframe_frames = np.concatenate((keyframe_image, keyframe_time_color_coded_image), axis=0)  # (H*2, W, 3)
        
        self._clean_keyframes_from_scene()
        
        # Reset render settings
        self.render_keyframes = False
        self.render_segmentation = False
        self.render_tracks = False
        self.render_time_color_coded = False
        self.render_bboxes = False
        
        nr_frames = self.nr_frames
        
        # Preallocate arrays for better performance
        rgb_frames = np.empty((nr_frames, 512, 1024, 3), dtype=np.uint8)
        
        for fid in tqdm(range(nr_frames), desc="Rendering RGB"):
            # Generate the point cloud for the specified frame index (fid)
            self._update_geometry(fid)
            
            # Render the image
            image = render_image()   # uint8 [0,255]
            
            # Concat original rgb to the left of the rendered image (in-place)
            if self.rgbs is not None:
                rgb_frames[fid, :, :512, :] = self.rgbs[fid]
            # else do nothing
            
            rgb_frames[fid, :, 512:, :] = image
        
        # Render segmentation frames
        self.render_keyframes = False
        self.render_segmentation = True
        self.render_tracks = False
        self.render_time_color_coded = False
        self.render_bboxes = True
        
        # Preallocate segmentation frames
        segm_frames = np.empty((nr_frames, 512, 1024, 3), dtype=np.uint8)
        
        # Precompute segmentation RGB for all frames to avoid repeated reshaping
        if self.instances_masks is not None:
            segmentation_rgbs = self.instance_colors[self.instances_masks.reshape(nr_frames, -1)]
            segmentation_rgbs = segmentation_rgbs.reshape(nr_frames, self.instances_masks.shape[1], self.instances_masks.shape[2], -1)
        else:
            segmentation_rgbs = np.zeros((nr_frames, 512, 512, 3), dtype=np.uint8)
        
        for fid in tqdm(range(nr_frames), desc="Rendering Segmentation"):
            # Generate the point cloud for the specified frame index (fid)
            self._update_geometry(fid)
            
            # Render the image
            image = render_image()    # uint8 [0,255]
            
            # Concat segmentation image to the left of the rendered image (in-place)
            segm_frames[fid, :, :512, :] = segmentation_rgbs[fid]
            segm_frames[fid, :, 512:, :] = image
        
        # Combine all frames efficiently
        final_frames = np.concatenate((rgb_frames, segm_frames), axis=1)  # (T, H*2, W, 3)
        
        # Broadcast keyframe_frames efficiently
        keyframe_broadcast = np.broadcast_to(keyframe_frames[None, :, :, :], (nr_frames, *keyframe_frames.shape))
        final_frames = np.concatenate((final_frames, keyframe_broadcast), axis=2)  # (T, H*2, W*2, 3)
        
        print("Rendered frames shape:", final_frames.shape)
        
        # Convert to list of frames
        final_frames = [final_frames[i] for i in range(final_frames.shape[0])]
        
        return final_frames


def run_open3d_offline_renderer(
    width: int,
    height: int,
    nr_frames: int,
    rgbs: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    depths: np.ndarray | None = None,
    point_clouds: list | None = None,
    K: np.ndarray | None = None,
    poses_c2w: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None,
    max_tracks: int = 3000
) -> list[np.ndarray]:
    """
    Run the offline renderer to generate a batch of rendered frames.
    
    Args:
        rgbs: RGB images (T, H, W, 3)
        depths: Depth maps (T, H, W)
        intr_normalized: Normalized intrinsic matrix
        width: Image width
        height: Image height
        poses_c2w: Camera poses (world to camera)
        tracks3d: Optional 3D tracks (N, T, 3)
        instances_masks: Optional instance segmentation masks (T, H, W)
        generate_point_cloud_fn: Optional function to generate point clouds
        
    Returns:
        List of rendered frames
    """
    
    app = OffscreenRendererApp(
        width, height, nr_frames,
        rgbs=rgbs,
        depths=depths,
        point_clouds=point_clouds,
        K=K,
        poses_c2w=poses_c2w,
        tracks3d=tracks3d,
        instances_masks=instances_masks,
        max_tracks=max_tracks
    )
    
    final_frames = app.render_frames()
    
    return final_frames
