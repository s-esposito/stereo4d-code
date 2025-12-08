import pickle
import numpy as np
from utils import load_video_frames
import matplotlib.pyplot as plt
import open3d as o3d
import os
import math
import time
import utils
import o3d_utils
from tqdm import tqdm
from load_data import load_data

def view_data(input_dict):
    
    rgbs_left = input_dict['left']['video']
    print("rgbs_left shape:", rgbs_left.shape)
    rgbs_right = input_dict['right']['video']
    print("rgbs_right shape:", rgbs_right.shape)
    depths = input_dict['depths']
    print("depths shape:", depths.shape)
    intr_normalized = input_dict['intr_normalized']
    print("intr_normalized:", intr_normalized)
    extrs_left = input_dict['left']['camera']
    print("extrs_left length:", len(extrs_left))
    extrs_right = input_dict['right']['camera']
    print("extrs_right length:", len(extrs_right))
    tracks3d = input_dict.get('tracks3d', None)
    if tracks3d is not None:
        print("tracks3d shape:", tracks3d.shape)
    instances_masks = input_dict.get('instances_masks', None)
    if instances_masks is not None:
        print("instances_masks shape:", instances_masks.shape)
        
    width, height = rgbs_left.shape[2], rgbs_left.shape[1]
        
    o3d_utils.run_open3d_viewer(
        rgbs_right,
        depths,
        intr_normalized,
        width, height,
        poses_c2w=(extrs_left, extrs_right),
        tracks3d=tracks3d,
        instances_masks=instances_masks
    )
    
    return input_dict



if __name__ == "__main__":
    
    root_dir = "/home/stefano/Codebase/stereo4d-code/data"
    split = "test"
    scene = "H5xOyNqJkPs"
    timestamp = "38738739"
    
    input_dict = load_data(root_dir, split, scene, timestamp)
    view_data(input_dict)