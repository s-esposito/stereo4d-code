import utils
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import time
import matplotlib

def toOpen3dCloud(points,colors=None,normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max()>1:
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

def create_camera_trajectory(poses_c2w, color=[0, 0, 1]):
    """
    Create a line showing the camera trajectory through all poses.
    
    Args:
        poses_c2w: (T, 4, 4) array of camera-to-world matrices
        color: RGB color for the trajectory line
    
    Returns:
        Open3D LineSet representing the camera trajectory
    """
    # Extract camera centers from poses
    centers = poses_c2w[:, :3, 3]
    
    # Create lines connecting consecutive camera positions
    points = centers
    lines = [[i, i+1] for i in range(len(centers) - 1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    colors_list = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors_list)
    
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

def generate_o3d_point_cloud(rgb, depth, K, pose_c2w):
    """
    Generates a dummy point cloud that changes only its position and color 
    with the frame index (fid), maintaining a consistent scale.
    """
    
    xyz, rgb, _ = utils.generate_point_cloud(rgb, depth, K, pose_c2w)
    
    # Create the Open3D point cloud object
    pcd = toOpen3dCloud(xyz, rgb)
    return pcd


class VideoPointCloudApp:
    def __init__(self, rgbs, depths, K, poses_c2w, tracks3d=None, instances_masks=None, max_tracks=3000):
        # Initialize State
        self.state = {'fid': 0}
        self.rgbs = rgbs  # (T, H, W, 3)
        self.depths = depths  # (T, H, W)
        self.K = K
        self.poses_c2w = poses_c2w  # (T, 4, 4) or tuple of two (T, 4, 4)
        self.tracks3d = tracks3d  # (N, T, 3) or None
        self.instances_masks = instances_masks  # (T, H, W) or None
        self.is_running = True
        self.last_update_time = time.time()
        self.update_interval = 0.03 # 30ms
        self.keyframes_interval = 10
        
        # Rendering flags
        self.render_tracks = False
        self.render_segmentation = False
        self.render_keyframes = False
        
        # Precompute tracks colors
        if self.tracks3d is not None:
            # Convert first coordinate of each track to color
            self.tracks_colors = np.zeros_like(self.tracks3d[:, 0, :])
            empty_color = np.ones_like(self.tracks_colors[:, 0], dtype=bool)
            for fid in range(self.tracks3d.shape[1]):
                points_at_fid = self.tracks3d[:, fid, :]
                valid_points_mask = ~np.isnan(points_at_fid).any(axis=1) & ~np.isinf(points_at_fid).any(axis=1)
                new_values_mask = valid_points_mask & empty_color
                self.tracks_colors[new_values_mask] = points_at_fid[new_values_mask]
                empty_color |= ~valid_points_mask
                # check if all colors have been assigned
                if not np.any(empty_color):
                    break
            min_x, max_x = np.min(self.tracks_colors[:, 0]), np.max(self.tracks_colors[:, 0])
            min_y, max_y = np.min(self.tracks_colors[:, 1]), np.max(self.tracks_colors[:, 1])
            min_z, max_z = np.min(self.tracks_colors[:, 2]), np.max(self.tracks_colors[:, 2])
            # print(min_x, max_x, min_y, max_y, min_z, max_z)
            self.tracks_colors[:, 0] = (self.tracks_colors[:, 0] - min_x) / (max_x - min_x + 1e-8)
            self.tracks_colors[:, 1] = (self.tracks_colors[:, 1] - min_y) / (max_y - min_y + 1e-8)
            self.tracks_colors[:, 2] = (self.tracks_colors[:, 2] - min_z) / (max_z - min_z + 1e-8)
            # print(self.tracks_colors)
            # exit(0)
            # Limit number of tracks for performance
            n_tracks = self.tracks3d.shape[0]
            if n_tracks > max_tracks:
                indices = np.linspace(0, n_tracks - 1, max_tracks).astype(int)
                self.tracks3d = self.tracks3d[indices]
                self.tracks_colors = self.tracks_colors[indices]

        # Precompute instance colors if instance masks are provided
        if self.instances_masks is not None:
            self.instance_colors = []
            # Append black for background (instance 0)
            self.instance_colors.append(np.array([0, 0, 0]))
            # Instances start from 1
            num_instances = np.max(self.instances_masks)
            cmap = matplotlib.cm.get_cmap('tab20', num_instances)
            for i in range(num_instances):
                color = cmap(i)[:3]  # Get RGB color
                self.instance_colors.append(np.array(color))
            self.instance_colors = np.array(self.instance_colors) * 255.0
            self.instance_colors = self.instance_colors.astype(np.uint8)
            print("Instance colors:", self.instance_colors)

        # Setup Window
        self.window = gui.Application.instance.create_window(
            "Open3D Video Point Cloud Viewer", 1920, 1080)
        self.window.set_on_close(self._on_close)
        
        # Setup Scene Widget (3D Viewport)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)
        
        # Disable color correction/gamma correction by setting background to white
        # and disabling post-processing
        self.scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        
        # Setup Material and Initial Geometry
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"  # Unlit shader bypasses lighting calculations
        self.material.point_size = 3.0 
        
        self.PCD_NAME = "current_point_cloud"
        self.CAMERA_TRAJECTORY_NAME = "camera_trajectory"
        self.CAMERA_FRUSTUM_NAME = "current_camera_frustum"
        self.TRACK_LINES_NAME = "track_lines"
        
        # Add coordinate frame at world origin
        coord_frame = create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord_material = rendering.MaterialRecord()
        coord_material.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("coordinate_frame", coord_frame, coord_material)
        
        # Add grid on XZ plane (ground plane, perpendicular to Y axis)
        grid = create_grid(size=20.0, n=20, plane='xz', height=0.0)
        grid_material = rendering.MaterialRecord()
        grid_material.shader = "unlitLine"
        grid_material.line_width = 1.0
        self.scene_widget.scene.add_geometry("grid", grid, grid_material)
        
        # Init first frame
        initial_pcd = self._init_frame(fid=0)
        
        # Set Camera View
        bounds = initial_pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        # Setup UI Panel (Slider)
        self._setup_ui()
        
        # Register the tick callback
        # FIX: Use set_on_tick_event on the window instead of add_timer on the app
        self.window.set_on_tick_event(self._on_tick)
        
        print("Open3D GUI launched. Use the 'Frame Index' slider to navigate.")
    
    def _init_keyframes(self):

        # Add keyframes point clouds
        keyframes_fids = list(range(0, len(self.rgbs), self.keyframes_interval))
        for kf_fid in keyframes_fids:
        
            # Add initial point cloud
            if self.render_segmentation and self.instances_masks is not None:
                rgb = self.instance_colors[self.instances_masks[kf_fid].reshape(-1)].reshape(self.instances_masks[kf_fid].shape[0], self.instances_masks[kf_fid].shape[1], -1)
            else:
                rgb = self.rgbs[kf_fid]
            
            # Check if stereo camera
            if isinstance(self.poses_c2w, tuple):
                # Use right camera pose
                pose_c2w = self.poses_c2w[1][kf_fid]
            else:
                pose_c2w = self.poses_c2w[kf_fid]
                
            pcd = generate_o3d_point_cloud(
                rgb=rgb,
                depth=self.depths[kf_fid],
                K=self.K,
                pose_c2w=pose_c2w
            )
            self.scene_widget.scene.add_geometry(f"{self.PCD_NAME}_{kf_fid}", pcd, self.material)
    
    def _init_frame(self, fid=0):

        # Add current point cloud 
        
        pcd = self._update_point_cloud(fid)
        
        # Add current camera frustum
        
        self._update_camera_frustum(fid)
        
        # Add current camera trajectory
        
        self._update_camera_trajectory(fid)
        
        return pcd
    
    def _update_point_cloud(self, fid):
        
        # Remove existing point cloud
        self.scene_widget.scene.remove_geometry(self.PCD_NAME)
        
        # Add initial point cloud
        if self.render_segmentation and self.instances_masks is not None:
            rgb = self.instance_colors[self.instances_masks[fid].reshape(-1)].reshape(self.instances_masks[fid].shape[0], self.instances_masks[fid].shape[1], -1)
        else:
            rgb = self.rgbs[fid]
        
        # Check if stereo camera
        if isinstance(self.poses_c2w, tuple):
            # Use right camera pose
            pose_c2w = self.poses_c2w[1][fid]
        else:
            pose_c2w = self.poses_c2w[fid]
            
        pcd = generate_o3d_point_cloud(
            rgb=rgb,
            depth=self.depths[fid],
            K=self.K,
            pose_c2w=pose_c2w
        )
        self.scene_widget.scene.add_geometry(self.PCD_NAME, pcd, self.material)
        
        return pcd
    
    def _update_camera_trajectory(self, fid):
        
        # Remove existing trajectory if any
        self.scene_widget.scene.remove_geometry(self.CAMERA_TRAJECTORY_NAME)
        
        # Add camera trajectory (line connecting all camera positions)
            
        # Check if stereo camera
        if isinstance(self.poses_c2w, tuple):
            # get average between left and right camera poses
            # poses_c2w_left = self.poses_c2w[0]
            poses_c2w_right = self.poses_c2w[1]
            # poses_c2w = (poses_c2w_left + poses_c2w_right) / 2.0 (only for translation)
            # poses_c2w = poses_c2w_right.copy()
            # poses_c2w[:, :3, 3] = (poses_c2w_left[:, :3, 3] + poses_c2w_right[:, :3, 3]) / 2.0
            poses_c2w = poses_c2w_right
        else:
            poses_c2w = self.poses_c2w

        if len(poses_c2w) > 1:
            trajectory = create_camera_trajectory(poses_c2w, color=[0, 0.5, 1])
            traj_material = rendering.MaterialRecord()
            traj_material.shader = "unlitLine"
            traj_material.line_width = 2.0
            self.scene_widget.scene.add_geometry(self.CAMERA_TRAJECTORY_NAME, trajectory, traj_material)
        
    def _update_tracks_3d(self, fid):
        
        # Remove existing track lines if any
        self.scene_widget.scene.remove_geometry(self.TRACK_LINES_NAME)
        
        # Update track lines (trails)
        track_lines = create_track_lines(self.tracks3d, self.tracks_colors, fid, trail_length=30)
        if track_lines is not None:
            track_lines_material = rendering.MaterialRecord()
            track_lines_material.shader = "unlitLine"
            track_lines_material.line_width = 4.0
            self.scene_widget.scene.add_geometry(self.TRACK_LINES_NAME, track_lines, track_lines_material)
    
    def _update_camera_frustum(self, fid):
        
        # Check if stereo camera
        
        if isinstance(self.poses_c2w, tuple):
            
            # Remove existing frustums if any
            self.scene_widget.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left")
            self.scene_widget.scene.remove_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right")
            
            # Add left and right camera frustums if stereo
            pose_c2w_left = self.poses_c2w[0][fid]
            pose_c2w_right = self.poses_c2w[1][fid]
            orange_color = [1, 0.5, 0]
            blue_color = [0, 0.5, 1]
            frustum_left = create_camera_frustum(pose_c2w_left, self.K, color=orange_color)
            frustum_right = create_camera_frustum(pose_c2w_right, self.K, color=blue_color)
            frustum_material = rendering.MaterialRecord()
            frustum_material.shader = "unlitLine"
            frustum_material.line_width = 2.0
            self.scene_widget.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_left", frustum_left, frustum_material)
            self.scene_widget.scene.add_geometry(f"{self.CAMERA_FRUSTUM_NAME}_right", frustum_right, frustum_material)        
        
        else:
            
            # Remove existing frustum if any
            self.scene_widget.scene.remove_geometry(self.CAMERA_FRUSTUM_NAME)
            
            # Add single camera frustum if not stereo
            pose_c2w = self.poses_c2w[fid]
            frustum = create_camera_frustum(pose_c2w, self.K)
            frustum_material = rendering.MaterialRecord()
            frustum_material.shader = "unlitLine"
            frustum_material.line_width = 2.0
            self.scene_widget.scene.add_geometry(self.CAMERA_FRUSTUM_NAME, frustum, frustum_material)
    
    def _setup_ui(self):
        em = self.window.theme.font_size
        self.panel = gui.Widget()
        # Store the layout as a member variable so we can resize it in _on_layout
        self.layout = gui.Vert(em, gui.Margins(em, em, em, em)) 
        
        # Slider for frame index
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, len(self.rgbs) - 1)
        self.slider.double_value = 0.0 
        self.slider.set_on_value_changed(self._on_slider_changed)
        self.layout.add_child(self.slider)
        
        # Toggle buttons to show tracks
        self.show_tracks_checkbox = gui.Checkbox("Show 3D Tracks")
        self.show_tracks_checkbox.checked = self.render_tracks
        self.show_tracks_checkbox.set_on_checked(self._on_show_tracks_toggled)
        self.layout.add_child(self.show_tracks_checkbox)
        
        # Toggle button to show segmentation
        self.show_segmentation_checkbox = gui.Checkbox("Show Segmentation")
        self.show_segmentation_checkbox.checked = self.render_segmentation
        self.show_segmentation_checkbox.set_on_checked(self._on_show_segmentation_toggled)
        self.layout.add_child(self.show_segmentation_checkbox)
        
        # Toggle button to show keyframes
        self.show_keyframes_checkbox = gui.Checkbox("Show Keyframes")
        self.show_keyframes_checkbox.checked = self.render_keyframes
        self.show_keyframes_checkbox.set_on_checked(self._on_show_keyframes_toggled)
        self.layout.add_child(self.show_keyframes_checkbox)
        
        self.panel.add_child(self.layout)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_height = 160
        # Panel is positioned at the top
        self.panel.frame = gui.Rect(r.x, r.y, r.width, panel_height)
        # Scene widget takes the bottom portion
        self.scene_widget.frame = gui.Rect(r.x, r.y + panel_height, r.width, r.height - panel_height)

    def _update_geometry(self, fid):
        """Helper to update the geometry based on frame index."""
        
        if fid < 0 or fid >= len(self.rgbs):
            return

        self.state['fid'] = fid
        
        # Update camera frustum
        self._update_camera_frustum(fid)
        
        # Update camera trajectory
        self._update_camera_trajectory(fid)

        # Update track visualizations
        
        if self.tracks3d is not None and self.render_tracks:
            # Update track lines
            self._update_tracks_3d(fid)
        
        if self.render_keyframes:
            # If rendering keyframes, just return
            return
        
        # Update point cloud
        self._update_point_cloud(fid)

    def _on_slider_changed(self, new_val):
        """Handle manual slider movement immediately."""
        val = int(new_val)
        if val != self.state['fid']:
            self._update_geometry(val)
            
    def _on_show_tracks_toggled(self, is_checked):
        """Handle toggling of 3D tracks visibility."""
        self.render_tracks = is_checked
        # Remove previously added tracks if any
        self.scene_widget.scene.remove_geometry(self.TRACK_LINES_NAME)
        # Update geometry to reflect change
        self._update_geometry(self.state['fid'])
        
    def _on_show_segmentation_toggled(self, is_checked):
        """Handle toggling of segmentation rendering."""
        self.render_segmentation = is_checked
        # Update geometry to reflect change
        if self.render_keyframes:
            # Clean scene from keyframes
            keyframes_fids = list(range(0, len(self.rgbs), self.keyframes_interval))
            for kf_fid in keyframes_fids:
                self.scene_widget.scene.remove_geometry(f"{self.PCD_NAME}_{kf_fid}")
            # Re-init keyframe rendering
            self._init_keyframes()
        else:
            self._update_geometry(self.state['fid'])
        
    def _on_show_keyframes_toggled(self, is_checked):
        """Handle toggling of keyframes visibility."""
        self.render_keyframes = is_checked
        
        # Update geometry to reflect change
        if self.render_keyframes:
            # Clean scene
            self.scene_widget.scene.remove_geometry(self.TRACK_LINES_NAME)
            self.scene_widget.scene.remove_geometry(self.PCD_NAME)
            # Init keyframes
            self._init_keyframes()  # init keyframe rendering
        else:
            # Clean scene from keyframes
            keyframes_fids = list(range(0, len(self.rgbs), self.keyframes_interval))
            for kf_fid in keyframes_fids:
                self.scene_widget.scene.remove_geometry(f"{self.PCD_NAME}_{kf_fid}")
            # Re-init single frame rendering
            self._init_frame(fid=self.state['fid'])  # re-init single frame rendering
            self._update_geometry(self.state['fid'])

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
    
    gui.Application.instance.initialize()
    app = VideoPointCloudApp(rgbs, depths, K, poses_c2w, tracks3d, instances_masks)
    gui.Application.instance.run()