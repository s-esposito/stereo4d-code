"""Base Renderer class with common functionality for online and offline rendering."""

import numpy as np
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import MaterialRecord, OffscreenRenderer
from open3d.visualization.gui import SceneWidget
import matplotlib.cm

from .geometry_utils import (
    toOpen3dCloud,
    create_coordinate_frame,
    create_grid,
    create_camera_frustum,
    create_camera_trajectory,
    create_track_lines,
)
from .color_utils import compute_tracks_colors, compute_instances_colors


class Renderer:
    """Base class for Open3D point cloud rendering."""
    
    def __init__(self, nr_frames: int, rgbs=None, depths=None, point_clouds=None, K=None, 
                 poses_c2w=None, tracks3d=None, instances_masks=None, max_tracks=3000,
                 generate_point_cloud_fn=None, filter_bbox_outliers=True):
        """
        Initialize the renderer.
        
        Args:
            nr_frames: Number of frames to render
            rgbs: (T, H, W, 3) RGB images
            depths: (T, H, W) depth maps
            point_clouds: List of precomputed point clouds (dicts with 'xyz', 'rgb', 'inst_id')
            K: 3x3 intrinsic camera matrix
            poses_c2w: (T, 4, 4) camera-to-world poses or tuple of two (stereo)
            tracks3d: (N, T, 3) 3D track positions
            instances_masks: (T, H, W) instance segmentation masks
            max_tracks: Maximum number of tracks to render (for performance)
            generate_point_cloud_fn: Function to generate point clouds from (rgb, depth, K, pose, instances)
            filter_bbox_outliers: Whether to filter outlier points when computing bounding boxes
        """
        # Rendering flags
        self.render_tracks = False
        self.render_segmentation = False
        self.render_keyframes = False
        self.render_time_color_coded = False
        self.render_bboxes = False
        self.filter_bbox_outliers = filter_bbox_outliers
        
        # Initialize State
        self.state = {'fid': 0}
        self.nr_frames = nr_frames
        self.rgbs = rgbs
        self.depths = depths
        self.point_clouds = point_clouds
        self.K = K  # (3, 3) or (T, 3, 3)
        self.poses_c2w = poses_c2w  # (T, 4, 4) or tuple of two (stereo)
        self.tracks3d = tracks3d
        self.instances_masks = instances_masks
        self.nr_keyframes = 10
        self.keyframes_interval = max(10, self.nr_frames // self.nr_keyframes)
        self.generate_point_cloud_fn = generate_point_cloud_fn
        self.tracks_tail_length = 10
        
        # Check if data type is correct 
        # RGBs shoud be uint8 in [0, 255]
        if self.rgbs is not None:
            if isinstance(self.rgbs, tuple):
                assert len(self.rgbs) == 2, "If rgbs is a tuple, it must contain two elements for stereo."
                rgbs_left = self.rgbs[0]
                rgbs_right = self.rgbs[1]
                assert isinstance(rgbs_left, np.ndarray), "RGB images must be a numpy array."
                assert isinstance(rgbs_right, np.ndarray), "RGB images must be a numpy array."
                assert rgbs_left.dtype == np.uint8 and rgbs_right.dtype == np.uint8, "RGB images must be of type uint8."
            else:
                assert isinstance(self.rgbs, np.ndarray), "RGB images must be a numpy array."
                assert self.rgbs.dtype == np.uint8, "RGB images must be of type uint8."
        
        # instances_masks should be uint8
        if self.instances_masks is not None:
            assert isinstance(self.instances_masks, np.ndarray), "Instance masks must be a numpy array."
            assert self.instances_masks.dtype == np.uint8, "Instance masks must be of type uint8."
        
        # Precompute time color coding (one rgb per frame, turbo colormap)
        cmap = matplotlib.cm.get_cmap('turbo', self.nr_frames)
        time_values = np.linspace(0, 1, self.nr_frames)
        self.time_color_coding = (cmap(time_values)[:, :3] * 255.0).astype(np.uint8)
        
        # Precompute tracks colors
        if self.tracks3d is not None:
            self.tracks_colors = compute_tracks_colors(self.tracks3d)
            
            # Limit number of tracks for performance
            n_tracks = self.tracks3d.shape[0]
            if n_tracks > max_tracks:
                indices = np.linspace(0, n_tracks - 1, max_tracks).astype(int)
                self.tracks3d = self.tracks3d[indices]
                self.tracks_colors = self.tracks_colors[indices]
        
        # Precompute instance colors if instance masks are provided
        if self.instances_masks is not None:
            self.instance_colors = compute_instances_colors(self.instances_masks)
        
        # Precompute unique instance ids (global)
        self.unique_instance_ids = None
        if self.instances_masks is not None:
            self.unique_instance_ids = np.unique(self.instances_masks)
        
        # Setup Material and Initial Geometry
        self.point_material = MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = 2.0
        
        self.line_material = MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 3.0
        
        self.PCD_NAME = "current_point_cloud"
        self.CAMERA_TRAJECTORY_NAME = "camera_trajectory"
        self.CAMERA_FRUSTUM_NAME = "current_camera_frustum"
        self.TRACK_LINES_NAME = "track_lines"
        
        self.o3d_renderer = None  # to be initialized in subclass
    
    def _clean_keyframes_from_scene(self):
        """Helper to clean keyframes from the scene."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        keyframes_fids = list(range(0, self.nr_frames, self.keyframes_interval))
        for kf_fid in keyframes_fids:
            self.o3d_renderer.scene.remove_geometry(f"{self.PCD_NAME}_{kf_fid}")
    
    def _update_geometry(self, fid):
        """Helper to update the geometry based on frame index."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if fid < 0 or fid >= self.nr_frames:
            return

        self.state['fid'] = fid
        
        # Update camera frustum
        self._update_camera_frustum(fid)
        
        # Update camera trajectory
        self._update_camera_trajectory(fid)

        # Update track visualizations
        if self.render_tracks:
            self._update_tracks_3d(fid)
        else:
            self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        
        if self.render_keyframes:
            return
        
        # Update point cloud
        self._update_point_cloud(fid)
    
        # Update segmentation bounding boxes
        if self.render_bboxes:
            self._update_instances_bboxes(fid)

    def _setup_o3d_renderer(self):
        """Configure renderer settings."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        self.o3d_renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        
        view_instance = self.o3d_renderer.scene.view
        view_instance.set_post_processing(True)
        view_instance.set_antialiasing(True)

        color_grading_linear = rendering.ColorGrading(
            rendering.ColorGrading.Quality.MEDIUM,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        view_instance.set_color_grading(color_grading_linear)
    
    def _init_coord_frame(self):
        """Add coordinate frame to scene."""
        assert self.o3d_renderer is not None, "Renderer not initialized."

        coord_frame = create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord_material = MaterialRecord()
        coord_material.shader = "defaultUnlit"
        self.o3d_renderer.scene.add_geometry("coordinate_frame", coord_frame, coord_material)
    
    def _init_grid_xz(self):
        """Add grid to scene."""
        assert self.o3d_renderer is not None, "Renderer not initialized."

        grid = create_grid(size=75.0, n=75, plane='xz', height=2.0)
        grid_material = MaterialRecord()
        grid_material.shader = "unlitLine"
        grid_material.line_width = 0.25
        grid.paint_uniform_color([0.2, 0.2, 0.2])
        self.o3d_renderer.scene.add_geometry("grid", grid, grid_material)
    
    def _init_keyframes(self):
        """Initialize keyframe point clouds."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.point_clouds is None:
            return

        keyframes_fids = list(range(0, self.nr_frames, self.keyframes_interval))
        for kf_fid in keyframes_fids:
            pcd = self.point_clouds[kf_fid]
            rgb = pcd['rgb']
            xyz = pcd['xyz']
            
            if self.render_segmentation and self.instances_masks is not None:
                rgb = self.instance_colors[pcd['inst_id']]
            elif self.render_time_color_coded:
                rgb = self.time_color_coding[kf_fid]
                rgb = np.tile(rgb.reshape(1, 3), (xyz.shape[0], 1))
            
            pcd = toOpen3dCloud(xyz, rgb)
            self.o3d_renderer.scene.add_geometry(f"{self.PCD_NAME}_{kf_fid}", pcd, self.point_material)
    
    def _init_frame(self, fid=0):
        """Initialize single frame rendering."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        self._update_point_cloud(fid)
        self._update_camera_frustum(fid)
        self._update_camera_trajectory(fid)
        
        if self.render_bboxes:
            self._update_instances_bboxes(fid)
    
    def _remove_instances_bboxes(self):
        """Remove instance bounding boxes from scene."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.instances_masks is not None and self.unique_instance_ids is not None:
            instance_ids = self.unique_instance_ids
            for instance_id in instance_ids:
                if instance_id == 0:
                    continue
                geometry_name = f"{self.PCD_NAME}_aabb_{instance_id}"
                self.o3d_renderer.scene.remove_geometry(geometry_name)
    
    def _update_instances_bboxes(self, fid):
        """Update instance bounding boxes for current frame."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.instances_masks is None or self.point_clouds is None:
            return
        
        self._remove_instances_bboxes()
        
        # Get precomputed point cloud for this frame
        pcd = self.point_clouds[fid]
        xyz = pcd['xyz']
        rgb = pcd['rgb']
        inst_id = pcd['inst_id']
        
        # Compute axis-aligned bounding boxes for each instance
        instance_ids = np.unique(inst_id)
        
        for instance_id in instance_ids:
            if instance_id == 0:
                continue
            
            # Extract points belonging to the instance using inst_id
            mask = inst_id == instance_id
            if not np.any(mask):
                continue
            
            xyz_instance = xyz[mask]
            rgb_instance = rgb[mask]
            
            if len(xyz_instance) == 0:
                continue
            
            # Create point cloud from precomputed points
            pcd_instance = toOpen3dCloud(xyz_instance, rgb_instance)
            
            # Filter outliers if enabled (pc has to have enough points)
            if self.filter_bbox_outliers and len(pcd_instance.points) > 10:
                # Use statistical outlier removal
                # pcd_instance = pcd_instance.voxel_down_sample(voxel_size=1.0)
                # nb_neighbors: number of neighbors to analyze for each point
                # std_ratio: standard deviation ratio threshold (lower = more aggressive filtering)
                pcd_instance, inlier_indices = pcd_instance.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                # pcd_instance, inlier_indices = pcd_instance.remove_radius_outlier(nb_points=16, radius=0.05)
                
                # Skip if too few points remain after filtering
                if len(pcd_instance.points) < 3:
                    continue
            
            # Compute axis-aligned bounding box
            aabb = pcd_instance.get_axis_aligned_bounding_box()
            aabb.color = self.instance_colors[instance_id] / 255.0
            geometry_name = f"{self.PCD_NAME}_aabb_{instance_id}"
            self.o3d_renderer.scene.add_geometry(geometry_name, aabb, self.line_material)
    
    def _remove_point_cloud(self, fid):
        """Remove point cloud from scene."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        self.o3d_renderer.scene.remove_geometry(self.PCD_NAME)
    
    def _update_point_cloud(self, fid):
        """Update point cloud for current frame."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.point_clouds is None:
            return
        
        self._remove_point_cloud(fid)
        
        pcd = self.point_clouds[fid]
        rgb = pcd['rgb']
        xyz = pcd['xyz']
        
        if self.render_segmentation and self.instances_masks is not None:
            rgb = self.instance_colors[pcd['inst_id']]
        elif self.render_time_color_coded:
            rgb = self.time_color_coding[fid]
            rgb = np.tile(rgb.reshape(1, 3), (xyz.shape[0], 1))
        
        pcd = toOpen3dCloud(xyz, rgb)
        self.o3d_renderer.scene.add_geometry(self.PCD_NAME, pcd, self.point_material)
        
        return pcd
    
    def _update_camera_trajectory(self, fid):
        """Update camera trajectory up to current frame."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.poses_c2w is None:
            return
        
        self.o3d_renderer.scene.remove_geometry(self.CAMERA_TRAJECTORY_NAME)
        
        # Check if stereo camera
        if isinstance(self.poses_c2w, tuple):
            poses_c2w_left = self.poses_c2w[0]
            poses_c2w_right = self.poses_c2w[1]
            poses_c2w = poses_c2w_right.copy()
            poses_c2w[:, :3, 3] = (poses_c2w_left[:, :3, 3] + poses_c2w_right[:, :3, 3]) / 2.0
        else:
            poses_c2w = self.poses_c2w

        poses_c2w = poses_c2w[:fid+1]
        
        trajectory = create_camera_trajectory(poses_c2w, color=[1, 0, 1])
        self.o3d_renderer.scene.add_geometry(self.CAMERA_TRAJECTORY_NAME, trajectory, self.line_material)
        
    def _update_tracks_3d(self, fid):
        """Update 3D track lines with enhanced visualization."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.tracks3d is None:
            return
        
        self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        self.o3d_renderer.scene.remove_geometry(f"{self.TRACK_LINES_NAME}_points")
        
        # Create track lines with gradient
        track_lines = create_track_lines(self.tracks3d, self.tracks_colors, fid, trail_length=self.tracks_tail_length)
        if track_lines is not None:
            track_lines_material = MaterialRecord()
            track_lines_material.shader = "unlitLine"
            track_lines_material.line_width = 3.0  # Increased from 2.0 for better visibility
            self.o3d_renderer.scene.add_geometry(self.TRACK_LINES_NAME, track_lines, track_lines_material)
        
        # Add spheres at current track positions for better visibility
        current_points = []
        current_colors = []
        
        visible_list = ~np.isnan(self.tracks3d[:, fid]).any(axis=1) & ~np.isinf(self.tracks3d[:, fid]).any(axis=1)
        
        for track_idx in range(self.tracks3d.shape[0]):
            if visible_list[track_idx]:
                current_points.append(self.tracks3d[track_idx, fid])
                current_colors.append(self.tracks_colors[track_idx])
        
        if len(current_points) > 0:
            # Create point cloud for current positions
            current_pcd = toOpen3dCloud(np.array(current_points), np.array(current_colors))
            
            # Use larger points to make them visible
            current_points_material = MaterialRecord()
            current_points_material.shader = "defaultUnlit"
            current_points_material.point_size = 5.0  # Larger points for visibility
            self.o3d_renderer.scene.add_geometry(f"{self.TRACK_LINES_NAME}_points", current_pcd, current_points_material)
    
    def _update_camera_frustum(self, fid):
        """Update camera frustum for current frame."""
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.poses_c2w is None and self.K is None:
            return
        
        poses_c2w = self.poses_c2w
        
        if poses_c2w is None and self.K is not None:
            # make identity poses if not provided
            poses_c2w = np.eye(4, dtype=np.float32)
        
        if self.K.ndim == 3:
            K = self.K[fid]
        else:
            K = self.K
        
        if isinstance(poses_c2w, tuple):
            # Stereo camera
            self.o3d_renderer.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left")
            self.o3d_renderer.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right")
            
            pose_c2w_left = poses_c2w[0][fid]
            pose_c2w_right = poses_c2w[1][fid]
                
            orange_color = [1, 0.5, 0]
            blue_color = [0, 0.5, 1]
            frustum_left = create_camera_frustum(pose_c2w_left, K, color=orange_color)
            frustum_right = create_camera_frustum(pose_c2w_right, K, color=blue_color)
            self.o3d_renderer.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left", frustum_left, self.line_material)
            self.o3d_renderer.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right", frustum_right, self.line_material)        
        else:
            # Mono camera
            self.o3d_renderer.scene.remove_geometry(self.CAMERA_FRUSTUM_NAME)
            
            # check if fixed pose
            if poses_c2w.ndim == 2:
                pose_c2w = poses_c2w
            else:
                pose_c2w = poses_c2w[fid]
                
            frustum = create_camera_frustum(pose_c2w, K)
            self.o3d_renderer.scene.add_geometry(self.CAMERA_FRUSTUM_NAME, frustum, self.line_material)
