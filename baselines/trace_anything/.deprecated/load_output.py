import torch
import matplotlib.pyplot as plt
import numpy as np
import baselines.trace_anything.utils as utils


def visualize_point_clouds_interactive(t_vals, frames):
    """
    Interactive visualization with frame slider and trajectories.
    
    Args:
        t_vals: Timeline values (T,)
        frames: List of frame dictionaries with point cloud data
    """
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    
    scale = 50.0  # Scale factor for better visibility
    
    T = len(t_vals)
    n_frames = len(frames)
    
    print("\nVisualization Info:")
    print(f"  Number of timesteps: {T}")
    print(f"  Number of frames: {n_frames}")
    
    # Precompute trajectories for visualization
    # Sample a subset of points to track (for performance)
    max_tracks = 500
    trajectories = []
    trajectory_colors = []
    
    # Get some foreground points from first frame and track them
    first_frame = frames[0]
    if len(first_frame['pts_fg_per_t']) > 0 and first_frame['pts_fg_per_t'][0].shape[0] > 0:
        n_points = min(max_tracks, first_frame['pts_fg_per_t'][0].shape[0])
        # Sample evenly across points
        sample_indices = np.linspace(0, first_frame['pts_fg_per_t'][0].shape[0] - 1, n_points, dtype=int)
        
        # Get XYZ coordinates at t=0 and convert to RGB colors
        pts_t0 = first_frame['pts_fg_per_t'][0][sample_indices]  # XYZ at time 0
        
        # Normalize XYZ to [0, 1] range for RGB coloring
        # Find global min/max across all coordinates
        xyz_min = pts_t0.min(axis=0)
        xyz_max = pts_t0.max(axis=0)
        xyz_range = xyz_max - xyz_min
        # Avoid division by zero
        xyz_range = np.where(xyz_range == 0, 1, xyz_range)
        
        # Normalize to [0, 1]
        track_colors = (pts_t0 - xyz_min) / xyz_range
        
        # Build trajectories across timesteps
        traj_idx = 0
        for pt_idx in sample_indices:
            trajectory = []
            for t_idx in range(T):
                if pt_idx < first_frame['pts_fg_per_t'][t_idx].shape[0]:
                    pt = first_frame['pts_fg_per_t'][t_idx][pt_idx] * scale
                    trajectory.append(pt)
            if len(trajectory) > 1:
                trajectories.append(np.array(trajectory))
                # Assign color from XYZ position at t=0
                trajectory_colors.append(track_colors[traj_idx])
                traj_idx += 1
    
    print(f"  Trajectories: {len(trajectories)}")
    
    # Create application and window
    app = gui.Application.instance
    app.initialize()
    
    window = app.create_window("Trace Anything - Interactive Viewer", 1600, 900)
    
    # Create 3D scene widget
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([1, 1, 1, 1])
    
    # Create UI panel
    em = window.theme.font_size
    panel = gui.Vert(em, gui.Margins(em, em, em, em))
    
    # Timestep slider
    timestep_label = gui.Label("Timestep")
    panel.add_child(timestep_label)
    
    timestep_slider = gui.Slider(gui.Slider.INT)
    timestep_slider.set_limits(0, T - 1)
    timestep_slider.int_value = T // 2
    panel.add_child(timestep_slider)
    
    timestep_value_label = gui.Label(f"t = {t_vals[T // 2]:.3f}")
    panel.add_child(timestep_value_label)
    
    panel.add_fixed(em)
    
    # Show trajectories checkbox
    show_trajectories_checkbox = gui.Checkbox("Show Trajectories")
    show_trajectories_checkbox.checked = True
    panel.add_child(show_trajectories_checkbox)
    
    # Show background checkbox
    show_background_checkbox = gui.Checkbox("Show Background")
    show_background_checkbox.checked = True
    panel.add_child(show_background_checkbox)
    
    panel.add_fixed(em)
    
    # Info labels
    info_label = gui.Label(f"Frames: {n_frames}")
    panel.add_child(info_label)
    
    # Material for point clouds
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    
    # Material for lines
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.0
    
    def update_visualization(t_idx):
        """Update the 3D scene based on current timestep."""
        scene_widget.scene.clear_geometry()
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        scene_widget.scene.add_geometry("coord_frame", coord_frame, mat)
        
        # Collect points from all frames at this timestep
        all_fg_points = []
        all_fg_colors = []
        all_bg_points = []
        all_bg_colors = []
        
        for frame in frames:
            # Foreground points
            pts_fg = frame['pts_fg_per_t'][t_idx]
            if pts_fg.shape[0] > 0:
                fg_mask = frame['fg_mask_flat']
                img_rgb = frame['img_rgb_float']
                fg_colors = img_rgb[fg_mask][:pts_fg.shape[0]]
                all_fg_points.append(pts_fg * scale)
                all_fg_colors.append(fg_colors)
            
            # Background points
            if show_background_checkbox.checked:
                bg_pts = frame['bg_pts']
                if bg_pts.shape[0] > 0:
                    bg_mask = frame['bg_mask_flat']
                    img_rgb = frame['img_rgb_float']
                    bg_colors = img_rgb[bg_mask][:bg_pts.shape[0]]
                    all_bg_points.append(bg_pts * scale)
                    all_bg_colors.append(bg_colors)
        
        # Add foreground point cloud
        if all_fg_points:
            fg_points = np.concatenate(all_fg_points, axis=0)
            fg_colors = np.concatenate(all_fg_colors, axis=0)
            pcd_fg = o3d.geometry.PointCloud()
            pcd_fg.points = o3d.utility.Vector3dVector(fg_points.astype(np.float64))
            pcd_fg.colors = o3d.utility.Vector3dVector(fg_colors.astype(np.float64))
            scene_widget.scene.add_geometry("fg_points", pcd_fg, mat)
        
        # Add background point cloud
        if all_bg_points and show_background_checkbox.checked:
            bg_points = np.concatenate(all_bg_points, axis=0)
            bg_colors = np.concatenate(all_bg_colors, axis=0)
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_points.astype(np.float64))
            pcd_bg.colors = o3d.utility.Vector3dVector(bg_colors.astype(np.float64))
            scene_widget.scene.add_geometry("bg_points", pcd_bg, mat)
        
        # Add trajectories if enabled
        if show_trajectories_checkbox.checked and trajectories:
            for i, traj in enumerate(trajectories):
                if len(traj) < 2:
                    continue
                
                # Create line set for trajectory
                lines = [[j, j+1] for j in range(len(traj)-1)]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(traj.astype(np.float64))
                line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
                
                # Color the trajectory
                if i < len(trajectory_colors):
                    colors = [trajectory_colors[i] for _ in range(len(lines))]
                    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
                
                scene_widget.scene.add_geometry(f"trajectory_{i}", line_set, line_mat)
    
    def on_timestep_changed(value):
        """Callback when timestep slider changes."""
        t_idx = int(value)
        timestep_value_label.text = f"t = {t_vals[t_idx]:.3f}"
        update_visualization(t_idx)
    
    def on_trajectories_toggled(checked):
        """Callback when trajectories checkbox is toggled."""
        t_idx = timestep_slider.int_value
        update_visualization(t_idx)
    
    def on_background_toggled(checked):
        """Callback when background checkbox is toggled."""
        t_idx = timestep_slider.int_value
        update_visualization(t_idx)
    
    timestep_slider.set_on_value_changed(on_timestep_changed)
    show_trajectories_checkbox.set_on_checked(on_trajectories_toggled)
    show_background_checkbox.set_on_checked(on_background_toggled)
    
    # Layout
    window.add_child(scene_widget)
    window.add_child(panel)
    
    def on_layout(layout_context):
        r = window.content_rect
        panel_width = 15 * em
        scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
    
    window.set_on_layout(on_layout)
    
    # Initial visualization
    initial_t_idx = timestep_slider.int_value
    update_visualization(initial_t_idx)
    
    # Setup camera
    bounds = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
    scene_widget.setup_camera(60, bounds, [0, 0, 0])
    
    print("\nInteractive controls:")
    print("  - Use slider to change timestep")
    print("  - Toggle trajectories/background with checkboxes")
    print("  - Drag to rotate, scroll to zoom")
    
    app.run()
    
if __name__ == "__main__":
    
    # filepath = "/home/stefano/Codebase/stereo4d-code/baselines/trace_anything/outputs/H5xOyNqJkPs_38738739-right_rectified/output.pt"
    filepath = "/home/stefano/Codebase/stereo4d-code/baselines/trace_anything/outputs/H5xOyNqJkPs_38738739-right_rectified/output.pt"
    preds = utils.load_output(filepath)