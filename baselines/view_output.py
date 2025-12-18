import torch
import numpy as np

if __name__ == "__main__":
    
    # filepath = "/home/stefano/Codebase/stereo4d-code/baselines/trace_anything/outputs/H5xOyNqJkPs_38738739-right_rectified/output.pt"
    filepath = "/home/stefano/Codebase/stereo4d-code/baselines/trace_anything/outputs/elephant/output.pt"
    # filepath = "/home/stefano/Codebase/stereo4d-code/baselines/any4d/outputs/H5xOyNqJkPs_38738739-right_rectified"
    # filepath = "/home/stefano/Codebase/stereo4d-code/baselines/any4d/outputs/stroller"
    
    # check if path contrains "trace_anything"
    if "trace_anything" in filepath:
        print("Loading trace_anything output")
        from trace_anything.utils import load_output
        from trace_anything.utils import view_with_open3d_viewer

    elif "any4d" in filepath:
        print("Loading any4d output")
        from any4d.utils import load_output
        from any4d.utils import view_with_open3d_viewer
        
    else:
        raise ValueError("Unknown baseline in filepath")
    
    data = load_output(filepath)
    view_with_open3d_viewer(data)
    
    