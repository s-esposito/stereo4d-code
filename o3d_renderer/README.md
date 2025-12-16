# O3D Renderer

Open3D-based point cloud visualization library for online and offline rendering.

## Features

- **Online Rendering**: Interactive GUI viewer with controls for frame navigation, rendering options, and camera views
- **Offline Rendering**: Headless rendering for batch video generation
- **Support for**:
  - Point cloud sequences
  - Camera trajectories and frustums
  - 3D track visualization with trails
  - Instance segmentation with bounding boxes
  - Time-based color coding
  - Stereo camera setups

## Installation

Install in editable mode (development):

```bash
cd /home/stefano/Codebase/stereo4d-code/o3d_renderer
pip install -e .
```

## Usage

### Online Viewer (Interactive)

```python
from o3d_renderer import run_open3d_viewer
import numpy as np

# Your data
rgbs = ...  # optional rgb frames (T, H, W, 3)
depths = ...  # optional depth maps (T, H, W)
intr_normalized = ...  # optional intrinsic matrix (normalized) (3, 3)
poses_c2w = ...  # optional camera-to-world poses (T, 4, 4)
tracks3d = ...  # optional 3D tracks (N, T, 3)
instances_masks = ...  # optional instance masks (T, H, W)

# Run viewer
run_open3d_viewer(
    rgbs=rgbs,
    depths=depths,
    intr_normalized=intr_normalized,
    width=1920,
    height=1080,
    poses_c2w=poses_c2w,
    tracks3d=tracks3d,
    instances_masks=instances_masks,
)
```

### Offline Renderer (Headless)

```python
from o3d_renderer import run_open3d_offline_renderer

# Returns list of rendered frames
frames = run_open3d_offline_renderer(
    rgbs=rgbs,
    depths=depths,
    intr_normalized=intr_normalized,
    width=512,
    height=512,
    poses_c2w=poses_c2w,
    tracks3d=tracks3d,
    instances_masks=instances_masks,
)
```

## Requirements

- Python >= 3.8
- open3d >= 0.17.0
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0

## License

See parent project license.
