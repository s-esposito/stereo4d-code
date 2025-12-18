import os
import numpy as np
import imageio
from tqdm import tqdm


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
    
    # load data
    depth_dir = os.path.join(scene_dir, "depth")
    rectified_dir = os.path.join(depth_dir, "rectified_images")
    # load all .png files in rectified_dir
    depth_files = [f for f in os.listdir(rectified_dir) if f.endswith(".png")]
    depth_files.sort()
    print(f"Found {len(depth_files)} frames in {rectified_dir}")
    
    # Load only a few frames for testing
    MAX_SEQ_LEN = 50
    depth_files = depth_files[:MAX_SEQ_LEN]
    
    # load all depth maps
    depth_maps = []
    for depth_path in tqdm(depth_files):
        # load depth from .png file
        depth_path = os.path.join(rectified_dir, depth_path)
        # load .png as numpy array using imageio  
        depth_map = imageio.imread(depth_path)
        # convert to float32
        depth_map = depth_map.astype(np.float32)
        depth_maps.append(depth_map)
    depth_maps = np.stack(depth_maps, axis=0)  # (T, H, W)
    print(f"Loaded depth maps shape: {depth_maps.shape}")
    
    # # plot first depth map
    # import matplotlib.pyplot as plt
    # plt.imshow(depth_maps[0], cmap='plasma')
    # plt.colorbar()
    # plt.title("First Depth Map")
    # plt.savefig("first_depth_map.png")
    # plt.close()
    
    # diarization 
    # hand_object_interaction 
    # heart_rate
    # mps
    # scene
    # video.vrs
    
    