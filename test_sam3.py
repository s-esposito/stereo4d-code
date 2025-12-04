from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    # load_frame,
    prepare_masks_for_visualization,
)
import numpy as np
import utils
import sam_utils
import matplotlib.pyplot as plt


def main(root_dir: str, split:str, scene:str, timestamp:str):

    # "video_path" needs to be either a JPEG folder or a MP4 video file
    video_path = f"{root_dir}/{split}/{scene}_{timestamp}/{scene}_{timestamp}-right_rectified.mp4"

    # Load video frames     
    video_frames_for_vis, _ = utils.load_video_frames(video_path)
    
    predictor = build_sam3_video_predictor()

    # Start a session
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    prompt_text_str = "person"
    frame_idx = 0 # Add a text prompt to frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx, # Arbitrary frame index
            text=prompt_text_str,
        )
    )

    # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    outputs_per_frame = sam_utils.propagate_in_video(predictor, session_id)
    
    # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    vis_frame_stride = 60
    plt.close("all")
    for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        sam_utils.visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )
        plt.savefig(f"sam3_text_prompt_output_frame{frame_idx:04d}.png", dpi=300)
        plt.close("all")
        
    # Convert integer keys to strings
    outputs_for_save = {
        str(key): value 
        for key, value in outputs_per_frame.items()
    }
    
    # Save outputs
    np.savez("sam3_text_prompt_output.npz", **outputs_for_save)


if __name__ == "__main__":
    
    root_dir = "/home/stefano/Codebase/stereo4d-code/data"
    # 
    split = "test"
    scene = "H5xOyNqJkPs"
    timestamp = "38738739"
    
    main(root_dir, split, scene, timestamp)