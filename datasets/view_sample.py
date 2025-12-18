import os
import numpy as np


if __name__ == "__main__":

    aria_dir = "/media/stefano/0D91176038319865/data/aria_gen_2_pilot"
    
    # list all folders in dataset_dir
    scene_dirs = [
        f for f in os.listdir(aria_dir)
        if os.path.isdir(os.path.join(aria_dir, f))
    ]
    print(f"Found {len(scene_dirs)} scenes in {aria_dir}")
    
    # get first scene
    scene_dir = os.path.join(aria_dir, scene_dirs[0])
    scene_name = scene_dirs[0]
    print(f"Loading first scene: {scene_dir}")
    
    from aria.utils import load_data
    from aria.utils import view_with_open3d_viewer
    
    data = load_data(scene_dir, scene_name)
    view_with_open3d_viewer(data)
    
    
    