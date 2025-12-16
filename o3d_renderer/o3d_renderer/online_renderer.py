"""
Online renderer with interactive GUI for visualizing 3D point clouds and tracks.
"""

import time
import numpy as np
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization.gui import Application, SceneWidget
from open3d.visualization.rendering import Open3DScene

from .renderer import Renderer
from .geometry_utils import toOpen3dCloud


class OnlineRendererApp(Renderer):
    """Interactive renderer with GUI controls for real-time visualization."""
    
    def __init__(
        self,
        nr_frames: int,
        rgbs: np.ndarray | None = None,
        depths: np.ndarray | None = None,
        point_clouds: list | None = None,
        K: np.ndarray | None = None,  # (3, 3) or (T, 3, 3)
        poses_c2w: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        tracks3d: np.ndarray | None = None,  # (N, T, 3)
        instances_masks: np.ndarray | None = None,
        max_tracks: int = 3000
    ):
        super().__init__(nr_frames, rgbs, depths, point_clouds, K, poses_c2w, tracks3d, instances_masks, max_tracks)
        
        self.is_running = True
        self.last_update_time = time.time()
        self.update_interval = 0.03  # 30ms  

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
        # self.o3d_renderer.setup_camera(60, self.bounds, self.scene_center)

        # setup camera to position [0, -5, 0]
        self.o3d_renderer.scene.camera.look_at(
            np.array([0.0, 0.0, 6.0]), # self.scene_center,
            np.array([2, -2, -2]),
            np.array([0, -1, 0])
        )

        # Register the tick callback
        self.window.set_on_tick_event(self._on_tick)
    
    def _setup_ui(self):
        """Setup the UI panel with controls."""
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
        """Handle window layout changes."""
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
        distance = scene_size * 1.0
        eye = self.scene_center + eye_offset * distance
        
        # Use look_at if available, otherwise fall back to setup_camera
        try:
            self.o3d_renderer.look_at(self.scene_center, eye, up_vector)
        except AttributeError:
            # Fallback: just reset to default view with bounds
            self.o3d_renderer.setup_camera(60, self.bounds, self.scene_center)
    
    def _on_view_top(self):
        """Set camera to top view (looking down on XZ plane)."""
        self._set_camera_view(np.array([0, -1, 0]), np.array([0, 0, 1]))
    
    def _on_view_bottom(self):
        """Set camera to bottom view (looking up from below)."""
        self._set_camera_view(np.array([0, 1, 0]), np.array([0, 0, -1]))
    
    def _on_view_left(self):
        """Set camera to left view (looking along +X axis)."""
        self._set_camera_view(np.array([-1, 0, 0]), np.array([0, -1, 0]))
    
    def _on_view_right(self):
        """Set camera to right view (looking along -X axis)."""
        self._set_camera_view(np.array([1, 0, 0]), np.array([0, -1, 0]))
    
    def _on_view_front(self):
        """Set camera to front view (looking along +Z axis)."""
        self._set_camera_view(np.array([0, 0, -1]), np.array([0, -1, 0]))
    
    def _on_view_back(self):
        """Set camera to back view (looking along -Z axis)."""
        self._set_camera_view(np.array([0, 0, 1]), np.array([0, -1, 0]))

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
            
            slider_val = int(self.slider.double_value)
            if slider_val != self.state['fid']:
                self._update_geometry(slider_val)
                return True  # Needs redraw
                
        return False  # No redraw needed

    def _on_close(self):
        """Handle window close event."""
        self.is_running = False
        gui.Application.instance.quit()
        return True


def run_open3d_viewer(
    nr_frames: int,
    rgbs: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    point_clouds: list | None = None,
    K: np.ndarray | None = None,
    poses_c2w: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    tracks3d: np.ndarray | None = None,
    instances_masks: np.ndarray | None = None,
    max_tracks: int = 3000
):
    """
    Run the interactive Open3D viewer for visualizing point clouds and tracks.
    
    Args:
        nr_frames: Number of frames in the sequence
        rgbs: RGB images (T, H, W, 3)
        depths: Depth maps (T, H, W)
        point_clouds: List of point cloud data for each frame
        K: Intrinsic matrix (3, 3) or (T, 3, 3)
        poses_c2w: Camera poses (world to camera)
        tracks3d: Optional 3D tracks (N, T, 3)
        instances_masks: Optional instance segmentation masks (T, H, W)
        max_tracks: Maximum number of tracks to visualize
    """
    
    gui.Application.instance.initialize()
    app = OnlineRendererApp(
        nr_frames=nr_frames,
        rgbs=rgbs,
        depths=depths,
        point_clouds=point_clouds,
        K=K,
        poses_c2w=poses_c2w,
        tracks3d=tracks3d,
        instances_masks=instances_masks,
        max_tracks=max_tracks
    )
    gui.Application.instance.run()
