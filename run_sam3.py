from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    # load_frame,
    prepare_masks_for_visualization,
)
import numpy as np
import utils
import sam_utils
import matplotlib.pyplot as plt
import torch

# use bfloat16 for the entire notebook. If your card doesn't support it, try float16 instead
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

CLASSES = [
    "vehicle",
    "person",
    "animal",
]


def main(root_dir: str, split:str, scene:str, timestamp:str):

    video_path = f"{root_dir}/stereo4d-righteye-perspective/{split}_mp4s/{scene}_{timestamp}-right_rectified.mp4"

    # Load video frames     
    video_frames_for_vis, nr_frames = utils.load_video_frames(video_path)
    height, width = video_frames_for_vis[0].shape[:2]
    
    predictor = build_sam3_video_predictor()

    # Start a session
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    # prompt_text_str = ". ".join(CLASSES)
    frame_idx = 0 # Add a text prompt to frame 0
    cls_outputs = {}
    for cls in CLASSES:
        
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx, # Arbitrary frame index
                text=cls,
            )
        )

        # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
        outputs_per_frame = sam_utils.propagate_in_video(predictor, session_id)
        
        # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

        # vis_frame_stride = 60
        # plt.close("all")
        # for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        #     sam_utils.visualize_formatted_frame_output(
        #         frame_idx,
        #         video_frames_for_vis,
        #         outputs_list=[outputs_per_frame],
        #         titles=["SAM 3 Dense Tracking outputs"],
        #         figsize=(6, 4),
        #     )
        #     plt.savefig(f"sam3_{cls}_output_frame{frame_idx:04d}.png", dpi=300)
        #     plt.close("all")
        
        # Convert integer keys to strings
        outputs_for_save = {
            str(key): value 
            for key, value in outputs_per_frame.items()
        }
        
        has_output = any(
            len(objs_dict) > 0 
            for objs_dict in outputs_for_save.values()
        )
        if not has_output:
            print(f"No outputs for class {cls}, skip saving.")
            continue
        
        instances_masks = []
        for fid in range(nr_frames):
            objs_dict = outputs_for_save[str(fid)]
            instance_mask = np.zeros((height, width), dtype=np.int32)
            for oid in objs_dict:
                mask = objs_dict[oid]  # (N, H, W)
                instance_mask[mask] = int(oid) + 1  # start from 1
            instances_masks.append(instance_mask)
        instances_masks = np.stack(instances_masks, axis=0)  # (N, H, W)
        # convert to uint8 if possible
        if instances_masks.max() < 256:
            instances_masks = instances_masks.astype(np.uint8)
        outputs_for_save = instances_masks  # (N, H, W)
        
        cls_outputs[cls] = outputs_for_save
    
    # Save outputs
    sam_path = f"{root_dir}/stereo4d-sam3/{split}/{scene}_{timestamp}-sam3.npz"
    np.savez(sam_path, **cls_outputs)
    
    # finally, close the inference session to free its GPU resources
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    # after all inference is done, we can shutdown the predictor
    # to free up the multi-GPU process group
    predictor.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', type=str, help='data root directory')
    parser.add_argument('--split', default='test', type=str, help='data split')
    parser.add_argument('--scene', default='H5xOyNqJkPs', type=str, help='scene name')
    parser.add_argument('--timestamp', default='38738739', type=str, help='timestamp')
    args = parser.parse_args()
    
    root_dir = args.data_root
    split = args.split
    scene = args.scene
    timestamp = args.timestamp
    
    main(root_dir, split, scene, timestamp)