import utils
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.gui as gui
from open3d.visualization.gui import SceneWidget, Application
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord, Open3DScene
import numpy as np
import time
import cv2
import matplotlib

def toOpen3dCloud(points,colors=None,normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        # Check if colors array is not empty before accessing max()
        if colors.size > 0 and colors.max() > 1:
            colors = colors/255.0
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
        size: Size of the frustum
        color: RGB color for the frustum lines
    
    Returns:
        Open3D LineSet representing the camera frustum
    """
    
    # Frustum in image space (pixels)
    
    points_2d_screen = np.array([
        [0, 0], # Bottom-left
        [K[0,2]*2, 0], # Bottom-right
        [K[0,2]*2, K[1,2]*2], # Top-right
        [0, K[1,2]*2],  # Top-left
    ])
    
    u = points_2d_screen[:, 0]
    v = points_2d_screen[:, 1]
    z = np.full(u.shape, size)
    
    # Flatten arrays
    u = u.reshape(-1)  # (H*W,)
    v = v.reshape(-1)  # (H*W,)
    z = z.reshape(-1)  # (H*W,)

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Apply pinhole camera model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d_camera = np.stack([x, y, z], axis=1)
    
    # append camera center to points_3d_camera
    points_3d_camera = np.vstack([np.array([[0, 0, 0]]), points_3d_camera])
    
    # print("points_3d_camera:", points_3d_camera)
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
        color: RGB color for the trajectory line, expected in [0.0, 1.0] range.
    
    Returns:
        Open3D LineSet representing the camera trajectory
    """
    if len(poses_c2w) < 2:
        # Need at least 2 poses to form a line segment
        return o3d.geometry.LineSet() 

    # Extract camera centers (translation vector is the 4th column, first 3 rows)
    centers = poses_c2w[:, :3, 3]
    
    # Create lines connecting consecutive camera positions
    points = centers # (N, 3) float array of camera centers
    lines = [[i, i+1] for i in range(len(centers) - 1)]
    # lines is a list of lists/tuples, e.g., [[0, 1], [1, 2], ...]
    
    line_set = o3d.geometry.LineSet()
    
    # Convert points (float) and lines (int) to Open3D vector types
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines).astype(np.int32))
    
    # Create colors for each line segment
    num_lines = len(lines)
    
    # Ensure color is a numpy array of floats (Open3D standard for Vector3dVector)
    color_np = np.array(color, dtype=np.float64).reshape(1, 3) 
    colors_array = np.tile(color_np, (num_lines, 1)) # (num_lines, 3) array
    
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    
    return line_set

def create_track_lines(tracks3d, tracks_colors, current_frame, trail_length=10):
    """
    Create line visualization for 3D tracks showing trails from previous frames.
    
    Args:
        tracks3d: (N, T, 3) array of 3D track positions
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
                # Color with alpha gradient (older = more transparent)
                # alpha = (i + 1) / len(track_points)
                color = tracks_colors[track_idx] # * alpha + (1 - alpha) * np.array([0.5, 0.5, 0.5])
                # convert point coordinate to color in [0,1]
                # color = matplotlib.cm.get_cmap('hsv')(track_idx / n_tracks)[:3]
                colors.append(color)
    
    if len(points) == 0 or len(lines) == 0:
        # Return None if no visible tracks
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
        # return None
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_array)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    
    return line_set

# def generate_o3d_point_cloud(rgb, depth, K, pose_c2w):
#     """
#     Generates a dummy point cloud that changes only its position and color 
#     with the frame index (fid), maintaining a consistent scale.
#     """
    
#     pcd = utils.generate_point_cloud(rgb, depth, K, pose_c2w)
#     rgb = pcd['rgb']
#     xyz = pcd['xyz']
#     scales = pcd['scales']
    
#     # Create the Open3D point cloud object
#     pcd = toOpen3dCloud(xyz, rgb)
#     return pcd

def compute_tracks_colors(tracks3d):

    # Convert first coordinate of each track to color
    tracks_colors = np.zeros_like(tracks3d[:, 0, :])
    empty_color = np.ones_like(tracks_colors[:, 0], dtype=bool)
    for fid in range(tracks3d.shape[1]):
        points_at_fid = tracks3d[:, fid, :]
        valid_points_mask = ~np.isnan(points_at_fid).any(axis=1) & ~np.isinf(points_at_fid).any(axis=1)
        new_values_mask = valid_points_mask & empty_color
        tracks_colors[new_values_mask] = points_at_fid[new_values_mask]
        empty_color |= ~valid_points_mask
        # check if all colors have been assigned
        if not np.any(empty_color):
            break
    min_x, max_x = np.min(tracks_colors[:, 0]), np.max(tracks_colors[:, 0])
    min_y, max_y = np.min(tracks_colors[:, 1]), np.max(tracks_colors[:, 1])
    min_z, max_z = np.min(tracks_colors[:, 2]), np.max(tracks_colors[:, 2])
    # print(min_x, max_x, min_y, max_y, min_z, max_z)
    tracks_colors[:, 0] = (tracks_colors[:, 0] - min_x) / (max_x - min_x + 1e-8)
    tracks_colors[:, 1] = (tracks_colors[:, 1] - min_y) / (max_y - min_y + 1e-8)
    tracks_colors[:, 2] = (tracks_colors[:, 2] - min_z) / (max_z - min_z + 1e-8)
    
    return tracks_colors

def compute_instances_colors(instances_masks):

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

class Renderer:
    def __init__(self, nr_frames: int, rgbs=None, depths=None, point_clouds=None, K=None, poses_c2w=None, tracks3d=None, instances_masks=None, max_tracks=3000):
        
        # Rendering flags
        self.render_tracks = False
        self.render_segmentation = False
        self.render_keyframes = False
        self.render_time_color_coded = False
        self.render_bboxes = False
        
        # Initialize State
        self.state = {'fid': 0}
        self.nr_frames = nr_frames
        self.rgbs = rgbs  # (T, H, W, 3)
        self.depths = depths  # (T, H, W)
        self.point_clouds = point_clouds  # List of point clouds
        self.K = K
        self.poses_c2w = poses_c2w  # (T, 4, 4) or tuple of two (T, 4, 4)
        self.tracks3d = tracks3d  # (N, T, 3) or None
        self.instances_masks = instances_masks  # (T, H, W) or None
        self.nr_keyframes = 10
        self.keyframes_interval = max(10, self.nr_frames // self.nr_keyframes)  # one every 
        
        # Precompute time color coding (one rgb per frame, turbo colormap)
        cmap = matplotlib.cm.get_cmap('turbo', self.nr_frames)
        # Generate normalized values from 0 to 1
        time_values = np.linspace(0, 1, self.nr_frames)
        # Get RGB values (shape: (N, 4) with RGBA)
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
        self.point_material.shader = "defaultUnlit"  # Unlit shader bypasses lighting calculations
        self.point_material.point_size = 2.0
        
        self.line_material = MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 3.0
        
        self.PCD_NAME = "current_point_cloud"
        self.CAMERA_TRAJECTORY_NAME = "camera_trajectory"
        self.CAMERA_FRUSTUM_NAME = "current_camera_frustum"
        self.TRACK_LINES_NAME = "track_lines"
        
        self.o3d_renderer: OffscreenRenderer | SceneWidget = None  # to be initialized in the viewer setup
    
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
            # Update track lines
            self._update_tracks_3d(fid)
        else:
            # Remove existing track lines if any
            self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        
        if self.render_keyframes:
            # If rendering keyframes, just return
            return
        
        # Update point cloud
        self._update_point_cloud(fid)
    
        # Update segmentation bounding boxes
        if self.render_bboxes:
            self._update_instances_bboxes(fid)

    def _setup_o3d_renderer(self):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        # Disable color correction/gamma correction by setting background to white
        # and disabling post-processing
        self.o3d_renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  #  0.75
        
        # Access the View instance via the scene object
        view_instance = self.o3d_renderer.scene.view 
        
        # Ensure post-processing is ON for anti-aliasing
        view_instance.set_post_processing(True)

        # The following config does not seem to be working as expected
        
        # Configure Anti-Aliasing
        # Use True for Fast Approximate Anti-Aliasing (FXAA) or other methods
        view_instance.set_antialiasing(True) 

        # Configure Color Grading to use LINEAR tone mapping
        color_grading_linear = rendering.ColorGrading(
            rendering.ColorGrading.Quality.MEDIUM,
            # This is the crucial step: use LINEAR to skip gamma correction
            rendering.ColorGrading.ToneMapping.LINEAR,
        )

        # Apply the linear color grading to the view
        view_instance.set_color_grading(color_grading_linear)
    
    def _init_coord_frame(self):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."

        coord_frame = create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord_material = MaterialRecord()
        coord_material.shader = "defaultUnlit"
        self.o3d_renderer.scene.add_geometry("coordinate_frame", coord_frame, coord_material)
    
    def _init_grid_xz(self):
        assert self.o3d_renderer is not None, "Renderer not initialized."

        grid = create_grid(size=75.0, n=75, plane='xz', height=2.0)
        grid_material = MaterialRecord()
        grid_material.shader = "unlitLine"
        grid_material.line_width = 0.25
        # set color
        grid.paint_uniform_color([0.2, 0.2, 0.2])
        self.o3d_renderer.scene.add_geometry("grid", grid, grid_material)
    
    def _init_keyframes(self):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.point_clouds is None:
            return

        # Add keyframes point clouds
        keyframes_fids = list(range(0, self.nr_frames, self.keyframes_interval))
        for kf_fid in keyframes_fids:

            pcd = self.point_clouds[kf_fid]
            rgb = pcd['rgb']
            xyz = pcd['xyz']
            
            # Add initial point cloud
            if self.render_segmentation and self.instances_masks is not None:
                # rgb = self.instance_colors[self.instances_masks[kf_fid].reshape(-1)].reshape(self.instances_masks[kf_fid].shape[0], self.instances_masks[kf_fid].shape[1], -1)
                rgb = self.instance_colors[pcd['inst_id']]
            elif self.render_time_color_coded:
                rgb = self.time_color_coding[kf_fid]  # (3)
                # reshape to (N, 3)
                rgb = np.tile(rgb.reshape(1, 3), (xyz.shape[0], 1))
                
            # create open3d point cloud
            pcd = toOpen3dCloud(xyz, rgb)
            self.o3d_renderer.scene.add_geometry(f"{self.PCD_NAME}_{kf_fid}", pcd, self.point_material)
    
    def _init_frame(self, fid=0):

        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        # Add current point cloud 
        
        self._update_point_cloud(fid)
        
        # Add current camera frustum
        
        self._update_camera_frustum(fid)
        
        # Add current camera trajectory
        
        self._update_camera_trajectory(fid)
        
        # Add segmentation bounding boxes
        
        if self.render_bboxes:
            self._update_instances_bboxes(fid)
    
    def _remove_instances_bboxes(self):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        # Remove bounding box per instance if segmentation is enabled
        if self.instances_masks is not None and self.unique_instance_ids is not None:
            instance_ids = self.unique_instance_ids
            for instance_id in instance_ids:
                if instance_id == 0:
                    continue  # Skip background
                geometry_name = f"{self.PCD_NAME}_aabb_{instance_id}"
                self.o3d_renderer.scene.remove_geometry(geometry_name)
    
    def _update_instances_bboxes(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.instances_masks is None:
            return
        
        if self.point_clouds is None:
            return
        
        # Remove existing bounding boxes
        self._remove_instances_bboxes()
        
        # Add bounding box per instance if segmentation is enabled
        
        # TODO Stefano: avoid unprojecting again, use precomputed point cloud with instance ids
            
        # Check if stereo camera
        if isinstance(self.poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = self.poses_c2w[1][fid]
        else:
            pose_c2w = self.poses_c2w[fid]
        
        # Compute axis-aligned bounding boxes for each instance
        instance_ids = np.unique(self.instances_masks[fid])
        h, w = self.instances_masks[fid].shape
        # Get rgb image colored by instance ids
        rgb = self.instance_colors[self.instances_masks[fid].reshape(-1)].reshape(h, w, -1)
        
        for instance_id in instance_ids:
            
            if instance_id == 0:
                continue  # Skip background
            
            mask = self.instances_masks[fid] == instance_id
            
            # Skip if instance has no pixels
            if not np.any(mask):
                continue
            
            # Extract points belonging to the instance
            depth_instance = np.where(mask, self.depths[fid], 0)
            rgb_instance = np.where(mask[..., None], rgb, 0)
            # pcd_instance = generate_o3d_point_cloud(
            #     rgb=rgb_instance,
            #     depth=depth_instance,
            #     K=self.K,
            #     pose_c2w=pose_c2w
            # )
            pcd = utils.generate_point_cloud(rgb_instance, depth_instance, self.K, pose_c2w)
            # Create the Open3D point cloud object
            pcd_instance = toOpen3dCloud(pcd['xyz'], pcd['rgb'])
            
            # Skip if point cloud is empty
            if len(pcd_instance.points) == 0:
                continue
            
            aabb = pcd_instance.get_axis_aligned_bounding_box()
            aabb.color = self.instance_colors[instance_id] / 255.0  # Normalize color to [0,1]
            geometry_name = f"{self.PCD_NAME}_aabb_{instance_id}"
            self.o3d_renderer.scene.add_geometry(geometry_name, aabb, self.line_material)
    
    def _remove_point_cloud(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        # Remove existing point cloud
        self.o3d_renderer.scene.remove_geometry(self.PCD_NAME)
    
    def _update_point_cloud(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.point_clouds is None:
            return
        
        self._remove_point_cloud(fid)
        
        pcd = self.point_clouds[fid]
        rgb = pcd['rgb']
        xyz = pcd['xyz']
        
        # Add initial point cloud
        if self.render_segmentation and self.instances_masks is not None:
            rgb = self.instance_colors[pcd['inst_id']]
        elif self.render_time_color_coded:
            rgb = self.time_color_coding[fid]  # (3)
            # reshape to (N, 3)
            rgb = np.tile(rgb.reshape(1, 3), (xyz.shape[0], 1))
        
        # create open3d point cloud
        pcd = toOpen3dCloud(xyz, rgb)
        self.o3d_renderer.scene.add_geometry(self.PCD_NAME, pcd, self.point_material)
        
        return pcd
    
    def _update_camera_trajectory(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.poses_c2w is None:
            return
        
        # Remove existing trajectory if any
        self.o3d_renderer.scene.remove_geometry(self.CAMERA_TRAJECTORY_NAME)
        
        # Add camera trajectory (line connecting all camera positions)
            
        # Check if stereo camera
        if isinstance(self.poses_c2w, tuple):
            # get average between left and right camera poses
            poses_c2w_left = self.poses_c2w[0]
            poses_c2w_right = self.poses_c2w[1]
            # poses_c2w = (poses_c2w_left + poses_c2w_right) / 2.0 (only for translation)
            poses_c2w = poses_c2w_right.copy()
            poses_c2w[:, :3, 3] = (poses_c2w_left[:, :3, 3] + poses_c2w_right[:, :3, 3]) / 2.0
            # poses_c2w = poses_c2w_right
        else:
            poses_c2w = self.poses_c2w

        # filter out poses after current fid
        poses_c2w = poses_c2w[:fid+1]
        
        trajectory = create_camera_trajectory(poses_c2w, color=[1, 0, 1])
        self.o3d_renderer.scene.add_geometry(self.CAMERA_TRAJECTORY_NAME, trajectory, self.line_material)
        
    def _update_tracks_3d(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.tracks3d is None:
            return
        
        # Remove existing track lines if any
        self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        
        # Update track lines (trails)
        track_lines = create_track_lines(self.tracks3d, self.tracks_colors, fid, trail_length=30)
        if track_lines is not None:
            track_lines_material = MaterialRecord()
            track_lines_material.shader = "unlitLine"
            track_lines_material.line_width = 1.0
            self.o3d_renderer.scene.add_geometry(self.TRACK_LINES_NAME, track_lines, track_lines_material)
    
    def _update_camera_frustum(self, fid):
        
        assert self.o3d_renderer is not None, "Renderer not initialized."
        
        if self.poses_c2w is None:
            return
        
        if self.K is None:
            return
        
        # Check if stereo camera
        
        if isinstance(self.poses_c2w, tuple):
            
            # Remove existing frustums if any
            self.o3d_renderer.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left")
            self.o3d_renderer.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right")
            
            # Add left and right camera frustums if stereo
            pose_c2w_left = self.poses_c2w[0][fid]
            pose_c2w_right = self.poses_c2w[1][fid]
            orange_color = [1, 0.5, 0]
            blue_color = [0, 0.5, 1]
            frustum_left = create_camera_frustum(pose_c2w_left, self.K, color=orange_color)
            frustum_right = create_camera_frustum(pose_c2w_right, self.K, color=blue_color)
            self.o3d_renderer.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left", frustum_left, self.line_material)
            self.o3d_renderer.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right", frustum_right, self.line_material)        
        
        else:
            
            # Remove existing frustum if any
            self.o3d_renderer.scene.remove_geometry(self.CAMERA_FRUSTUM_NAME)
            
            # Add single camera frustum if not stereo
            pose_c2w = self.poses_c2w[fid]
            frustum = create_camera_frustum(pose_c2w, self.K)
            self.o3d_renderer.scene.add_geometry(self.CAMERA_FRUSTUM_NAME, frustum, self.line_material)
    
class OnlineRendererApp(Renderer):
    def __init__(self, nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d=None, instances_masks=None, max_tracks=3000):
        super().__init__(nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks, max_tracks)
        
        self.is_running = True
        self.last_update_time = time.time()
        self.update_interval = 0.03 # 30ms  

        # Setup Window
        self.window = Application.instance.create_window(
            "Open3D Video Point Cloud Viewer", 1920, 1080)
        self.window.set_on_close(self._on_close)
        
        # Setup Scene Widget (3D Viewport)
        self.o3d_renderer = SceneWidget()
        self.o3d_renderer.scene = Open3DScene(self.window.renderer)
        self.window.add_child(self.o3d_renderer)
        
        # Setup UI Panel (Slider)
        self._setup_ui()
        
        # Setup Open3D Renderer
        self._setup_o3d_renderer()
        
        # Add coordinate frame at world origin
        self._init_coord_frame()
        
        # Add grid on XZ plane (ground plane, perpendicular to Y axis)
        self._init_grid_xz()
        
        # Init first frame
        self._init_frame(fid=0)
        
        # Set Camera View - store bounds and center for camera view switching
        if self.point_clouds is not None:
            initial_pcd = self.point_clouds[0]
            # to open3d point cloud
            initial_pcd = toOpen3dCloud(initial_pcd['xyz'], initial_pcd['rgb'])
            self.bounds = initial_pcd.get_axis_aligned_bounding_box()
        else:
            # set bounds to fixed size
            self.bounds = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([-5.0, -5.0, -5.0]),
                max_bound=np.array([5.0, 5.0, 5.0])
            )
            
        self.scene_center = self.bounds.get_center()
        self.o3d_renderer.setup_camera(60, self.bounds, self.scene_center)
        
        # # Estimate the size of the scene to position the camera far enough away
        # scene_size = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        # camera_distance = scene_size * 0.75
        
        # # Position the eye (viewer camera) relative to the center
        # # Example: Looking down and slightly in front (positive Z, positive Y)
        # eye = scene_center + np.array([camera_distance * 0.5, -camera_distance * 0.5, -camera_distance * 0.8])
        # up = np.array([0, 1, 0]) # Standard Up vector for world coordinates

        # # Set the camera view using OffscreenRenderer's supported overload 1
        # self.o3d_renderer.setup_camera(
        #     60.0,    # vertical_field_of_view (e.g., 60 degrees)
        #     scene_center,  # The point the camera looks at
        #     eye,     # The position of the camera
        #     up       # The vector defining 'up' for the camera
        # )

        # Register the tick callback
        # FIX: Use set_on_tick_event on the window instead of add_timer on the app
        self.window.set_on_tick_event(self._on_tick)
    
    def _setup_ui(self):
        
        em = self.window.theme.font_size
        self.panel = gui.Widget()
        # Store the layout as a member variable so we can resize it in _on_layout
        self.layout = gui.Vert(em, gui.Margins(em, em, em, em)) 
        
        # Frame slider section
        frame_label = gui.Label("Frame")
        self.layout.add_child(frame_label)
        
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, self.nr_frames - 1)
        self.slider.double_value = 0.0 
        self.slider.set_on_value_changed(self._on_slider_changed)
        self.layout.add_child(self.slider)
        
        # Add spacing
        self.layout.add_fixed(em * 0.5)
        
        # Rendering options section
        options_label = gui.Label("Rendering Options")
        self.layout.add_child(options_label)
        
        # Stack checkboxes vertically
        self.show_tracks_checkbox = gui.Checkbox("3D Tracks")
        self.show_tracks_checkbox.checked = self.render_tracks
        self.show_tracks_checkbox.set_on_checked(self._on_show_tracks_toggled)
        self.layout.add_child(self.show_tracks_checkbox)
        
        self.show_segmentation_checkbox = gui.Checkbox("Segmentation")
        self.show_segmentation_checkbox.checked = self.render_segmentation
        self.show_segmentation_checkbox.set_on_checked(self._on_show_segmentation_toggled)
        self.layout.add_child(self.show_segmentation_checkbox)
        
        self.show_bboxes_checkbox = gui.Checkbox("Bounding Boxes")
        self.show_bboxes_checkbox.checked = self.render_bboxes
        self.show_bboxes_checkbox.set_on_checked(self._on_show_bboxes_toggled)
        self.layout.add_child(self.show_bboxes_checkbox)
        
        self._on_show_time_color_coded_toggled(self.render_time_color_coded)
        self.show_time_color_coded_checkbox = gui.Checkbox("Time Color Coded")
        self.show_time_color_coded_checkbox.checked = self.render_time_color_coded
        self.show_time_color_coded_checkbox.set_on_checked(self._on_show_time_color_coded_toggled)
        self.layout.add_child(self.show_time_color_coded_checkbox)
        
        self.show_keyframes_checkbox = gui.Checkbox("Keyframes")
        self.show_keyframes_checkbox.checked = self.render_keyframes
        self.show_keyframes_checkbox.set_on_checked(self._on_show_keyframes_toggled)
        self.layout.add_child(self.show_keyframes_checkbox)
        
        # Add spacing
        self.layout.add_fixed(em * 0.5)
        
        # Camera views section
        camera_label = gui.Label("Camera Views")
        self.layout.add_child(camera_label)
        
        # Create two rows of camera view buttons
        camera_view_row1 = gui.Horiz(em * 0.5)
        
        self.view_top_button = gui.Button("Top")
        self.view_top_button.set_on_clicked(self._on_view_top)
        camera_view_row1.add_child(self.view_top_button)
        
        self.view_front_button = gui.Button("Front")
        self.view_front_button.set_on_clicked(self._on_view_front)
        camera_view_row1.add_child(self.view_front_button)
        
        self.view_right_button = gui.Button("Right")
        self.view_right_button.set_on_clicked(self._on_view_right)
        camera_view_row1.add_child(self.view_right_button)
        
        self.layout.add_child(camera_view_row1)
        
        camera_view_row2 = gui.Horiz(em * 0.5)
        
        self.view_bottom_button = gui.Button("Bottom")
        self.view_bottom_button.set_on_clicked(self._on_view_bottom)
        camera_view_row2.add_child(self.view_bottom_button)
        
        self.view_back_button = gui.Button("Back")
        self.view_back_button.set_on_clicked(self._on_view_back)
        camera_view_row2.add_child(self.view_back_button)
        
        self.view_left_button = gui.Button("Left")
        self.view_left_button.set_on_clicked(self._on_view_left)
        camera_view_row2.add_child(self.view_left_button)
        
        self.layout.add_child(camera_view_row2)
        
        self.panel.add_child(self.layout)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 250  # Width of the left-side panel
        # Panel is positioned on the left side
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        # Scene widget takes the right portion
        self.o3d_renderer.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    def _on_slider_changed(self, new_val):
        """Handle manual slider movement immediately."""
        val = int(new_val)
        if val != self.state['fid']:
            self._update_geometry(val)
            
    def _on_show_tracks_toggled(self, is_checked):
        """Handle toggling of 3D tracks visibility."""
        self.render_tracks = is_checked
        # Remove previously added tracks if any
        self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
        # Update geometry to reflect change
        self._update_geometry(self.state['fid'])
    
    def _on_show_segmentation_toggled(self, is_checked):
        """Handle toggling of segmentation rendering."""
        self.render_segmentation = is_checked
        # Update geometry to reflect change
        if self.render_keyframes:
            # Clean scene from keyframes
            self._clean_keyframes_from_scene()
            # Re-init keyframe rendering
            self._init_keyframes()
        else:
            self._update_geometry(self.state['fid'])
            
    def _on_show_time_color_coded_toggled(self, is_checked):
        """Handle toggling of time color coding rendering."""
        self.render_time_color_coded = is_checked
        # Update geometry to reflect change
        if self.render_keyframes:
            # Clean scene from keyframes
            self._clean_keyframes_from_scene()
            # Re-init keyframe rendering
            self._init_keyframes()
        else:
            self._update_geometry(self.state['fid'])
            
    def _on_show_bboxes_toggled(self, is_checked):
        """Handle toggling of bounding boxes visibility."""
        self.render_bboxes = is_checked
        
        # Remove existing bounding boxes
        self._remove_instances_bboxes()
        
        # Update geometry to reflect change
        if self.render_keyframes:
            pass
        else:
            self._update_geometry(self.state['fid'])
        
    def _on_show_keyframes_toggled(self, is_checked):
        """Handle toggling of keyframes visibility."""
        self.render_keyframes = is_checked
        
        # Update geometry to reflect change
        if self.render_keyframes:
            # Clean scene
            self.o3d_renderer.scene.remove_geometry(self.TRACK_LINES_NAME)
            self._remove_instances_bboxes()
            self._remove_point_cloud(self.state['fid'])
            # Init keyframes
            self._init_keyframes()  # init keyframe rendering
        else:
            # Clean scene from keyframes
            self._clean_keyframes_from_scene()
            # Re-init single frame rendering
            self._init_frame(fid=self.state['fid'])  # re-init single frame rendering
            self._update_geometry(self.state['fid'])
    
    def _set_camera_view(self, eye_offset, up_vector):
        """Helper method to set camera view from a given offset and up vector."""
        scene_size = np.linalg.norm(self.bounds.get_max_bound() - self.bounds.get_min_bound())
        distance = scene_size * 1.5
        eye = self.scene_center + eye_offset * distance
        
        # Use look_at if available, otherwise fall back to setup_camera
        try:
            self.o3d_renderer.look_at(self.scene_center, eye, up_vector)
        except AttributeError:
            # Fallback: just reset to default view with bounds
            self.o3d_renderer.setup_camera(60, self.bounds, self.scene_center)
    
    def _on_view_top(self):
        """Set camera to top view (looking down on XZ plane)."""
        self._set_camera_view(np.array([0, 1, 0]), np.array([0, 0, -1]))
    
    def _on_view_bottom(self):
        """Set camera to bottom view (looking up from below)."""
        self._set_camera_view(np.array([0, -1, 0]), np.array([0, 0, 1]))
    
    def _on_view_left(self):
        """Set camera to left view (looking along +X axis)."""
        self._set_camera_view(np.array([-1, 0, 0]), np.array([0, 1, 0]))
    
    def _on_view_right(self):
        """Set camera to right view (looking along -X axis)."""
        self._set_camera_view(np.array([1, 0, 0]), np.array([0, 1, 0]))
    
    def _on_view_front(self):
        """Set camera to front view (looking along +Z axis)."""
        self._set_camera_view(np.array([0, 0, 1]), np.array([0, 1, 0]))
    
    def _on_view_back(self):
        """Set camera to back view (looking along -Z axis)."""
        self._set_camera_view(np.array([0, 0, -1]), np.array([0, 1, 0]))

    def _on_tick(self):
        """
        Called every frame by the window loop. 
        We enforce a time check to simulate the 30ms timer.
        """
        if not self.is_running:
            return False

        now = time.time()
        # Rate limiting: only update if 30ms has passed
        if now - self.last_update_time >= self.update_interval:
            self.last_update_time = now
            
            # Logic: If you want auto-play, increment frame here.
            # If you only want the slider to control it, check if slider changed.
            
            slider_val = int(self.slider.double_value)
            if slider_val != self.state['fid']:
                self._update_geometry(slider_val)
                return True # Needs redraw
                
        return False # No redraw needed

    def _on_close(self):
        self.is_running = False
        gui.Application.instance.quit()
        return True

def run_open3d_viewer(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intr_normalized: dict,
    width: int, height: int,
    poses_c2w: tuple[np.ndarray, np.ndarray] | np.ndarray,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None,
):
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height
    print("Intrinsic Matrix K:\n", K)
        
    point_clouds = []
    for fid in range(len(rgbs)):
        # Check if stereo camera
        if isinstance(poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = poses_c2w[1][fid]
        else:
            pose_c2w = poses_c2w[fid]
        
        rgb = rgbs[fid]
        depth = depths[fid]
        instances = instances_masks[fid] if instances_masks is not None else None
        #
        pcd = utils.generate_point_cloud(rgb, depth, K, pose_c2w, instances=instances)
        point_clouds.append(pcd)
    
    nr_frames = len(rgbs)
    
    gui.Application.instance.initialize()
    app = OnlineRendererApp(nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks)
    gui.Application.instance.run()
    
    
class OffscreenRendererApp(Renderer):
    def __init__(self, width, height, nr_frames, rgbs=None, depths=None, point_clouds=None, K=None, poses_c2w=None, tracks3d=None, instances_masks=None, max_tracks=3000):
        super().__init__(nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks, max_tracks)
        
        # 
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
        # bounds = initial_pcd.get_axis_aligned_bounding_box()
        # self.o3d_renderer.setup_camera(60, bounds, bounds.get_center())
        
        # Calculate the view matrix (world-to-camera) from the camera-to-world pose
        # We want a view that encompasses the scene, typically centered on the point cloud.
        # bounds = initial_pcd.get_axis_aligned_bounding_box()
        # self.scene_center = bounds.get_center()
        
        
        # Determine a suitable camera position (eye) and up vector
        # This example places the viewer camera (eye) slightly above and behind the scene
        # to get a good overview, looking towards the center.
        
        # Estimate the size of the scene to position the camera far enough away
        # scene_size = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        # self.camera_distance = scene_size * 1.0
        self.camera_distance = 8.0
        
        # Position the eye (viewer camera) relative to the center
        # Example: Looking down and slightly in front (positive Z, positive Y)
        # eye = self.scene_center + np.array([self.camera_distance * 0.5, -self.camera_distance * 0.5, -self.camera_distance * 0.5])
        look_at = np.array([0.0, 0.0, 6.0])
        origin = np.array([0.0, 0.0, 0.0])
        eye = origin + np.array([2, -2, -2])  # Fixed position for better view
        self.up = np.array([0, 1, 0]) # Standard Up vector for world coordinates

        # Set the camera view using OffscreenRenderer's supported overload 1
        self.o3d_renderer.setup_camera(
            60.0,    # vertical_field_of_view (e.g., 60 degrees)
            look_at,  # The point the camera looks at
            eye,     # The position of the camera
            self.up       # The vector defining 'up' for the camera
        )
    
    
def run_open3d_offline_renderer(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intr_normalized: dict,
    width: int, height: int,
    poses_c2w: tuple[np.ndarray, np.ndarray] | np.ndarray,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None,
) -> list[np.ndarray]:
    
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height
    
    point_clouds = []
    for fid in range(len(rgbs)):
        # Check if stereo camera
        if isinstance(poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = poses_c2w[1][fid]
        else:
            pose_c2w = poses_c2w[fid]
        
        rgb = rgbs[fid]
        depth = depths[fid]
        instances = instances_masks[fid] if instances_masks is not None else None
        #
        pcd = utils.generate_point_cloud(rgb, depth, K, pose_c2w, instances=instances)
        point_clouds.append(pcd)
    
    # Reduced resolution for faster rendering
    nr_frames = len(rgbs)
    base_res = 512
    res_scale = 1
    res = base_res * res_scale
    app = OffscreenRendererApp(res, res, nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks)
    
    def render_image():
        # Render the image
        image = app.o3d_renderer.render_to_image()
        # Convert to numpy array and process in one go
        image = np.asarray(image)
        # Combined flip operation (more efficient than two separate flips)
        image = np.flipud(np.fliplr(image))
        # sRGB to linear color space conversion
        image = utils.srgb_to_linear(image)
        # Only resize if needed
        if res != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        return image
    
    # render keyframes (only once)
    
    # Clean scene
    app.o3d_renderer.scene.remove_geometry(app.TRACK_LINES_NAME)
    app.o3d_renderer.scene.remove_geometry(app.PCD_NAME)
    
    app.render_keyframes = True
    app.render_segmentation = False
    app.render_tracks = False
    app.render_time_color_coded = False
    app.render_bboxes = False
    
    app._clean_keyframes_from_scene()
    app._init_keyframes()
    
    keyframe_image = render_image()
    
    # render keyframes (time color coded, only once)
    
    app.render_keyframes = True
    app.render_segmentation = False
    app.render_tracks = False
    app.render_time_color_coded = True
    app.render_bboxes = False
    
    app._clean_keyframes_from_scene()
    app._init_keyframes()
    
    keyframe_time_color_coded_image = render_image()
    
    # concat keyframes renders vertically
    keyframe_frames = np.concatenate((keyframe_image, keyframe_time_color_coded_image), axis=0)  # (H*2, W, 3)
    
    app._clean_keyframes_from_scene()
    
    # 
    app.render_keyframes = False
    app.render_segmentation = False
    app.render_tracks = False
    app.render_time_color_coded = False
    app.render_bboxes = False
    
    nr_frames = rgbs.shape[0]
    
    # Preallocate arrays for better performance
    rgb_frames = np.empty((nr_frames, 512, 1024, 3), dtype=rgbs.dtype)
    
    for fid in tqdm(range(nr_frames), desc="Rendering RGB"):
        # Generate the point cloud for the specified frame index (fid)
        app._update_geometry(fid)
        
        # Render the image
        image = render_image()
        
        # concat original rgb to the left of the rendered image (in-place)
        rgb_frames[fid, :, :512, :] = rgbs[fid]
        rgb_frames[fid, :, 512:, :] = image
    
    # render segmentation frames
    app.render_keyframes = False
    app.render_segmentation = True
    app.render_tracks = False
    app.render_time_color_coded = False
    app.render_bboxes = True
    
    # Preallocate segmentation frames
    segm_frames = np.empty((nr_frames, 512, 1024, 3), dtype=rgbs.dtype)
    
    # Precompute segmentation RGB for all frames to avoid repeated reshaping
    if instances_masks is not None:
        segmentation_rgbs = app.instance_colors[instances_masks.reshape(nr_frames, -1)]
        segmentation_rgbs = segmentation_rgbs.reshape(nr_frames, instances_masks.shape[1], instances_masks.shape[2], -1)
    else:
        segmentation_rgbs = np.zeros((nr_frames, 512, 512, 3), dtype=rgbs.dtype)
    
    for fid in tqdm(range(nr_frames), desc="Rendering Segmentation"):
        # Generate the point cloud for the specified frame index (fid)
        app._update_geometry(fid)
        
        # Render the image
        image = render_image()
        
        # concat segmentation image to the left of the rendered image (in-place)
        segm_frames[fid, :, :512, :] = segmentation_rgbs[fid]
        segm_frames[fid, :, 512:, :] = image
    
    # Combine all frames efficiently
    final_frames = np.concatenate((rgb_frames, segm_frames), axis=1)  # (T, H*2, W, 3)
    
    # Broadcast keyframe_frames efficiently
    keyframe_broadcast = np.broadcast_to(keyframe_frames[None, :, :, :], (nr_frames, *keyframe_frames.shape))
    final_frames = np.concatenate((final_frames, keyframe_broadcast), axis=2)  # (T, H*2, W*2, 3)
    
    print("Rendered frames shape:", final_frames.shape)
    
    # convert to list of frames
    final_frames = [final_frames[i] for i in range(final_frames.shape[0])]
    
    return final_frames