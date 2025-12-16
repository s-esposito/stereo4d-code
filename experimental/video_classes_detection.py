import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import utils


def main(
    model,
    processor,
    root_dir: str,
    split: str,
    scene: str,
    timestamp: str,
    objs_list: list,
):

    video_path = f"{root_dir}/stereo4d-righteye-perspective/{split}/{scene}_{timestamp}-right_rectified.mp4"

    # Load video frames
    video_frames, nr_frames = utils.load_video_frames(video_path)
    height, width = video_frames[0].shape[:2]

    # Create the detection prompt for specified objects
    objects_str = ", ".join(objs_list)
    detection_prompt = f"Are any of these objects visible in the image: {objects_str}? Answer with only the object names that are present, separated by commas. If none are present, answer 'none'."

    # # Load the image
    # # NOTE: Replace 'path/to/your/image.jpg' with a local file path or a URL
    # image_path_or_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    # if image_path_or_url.startswith(('http://', 'https://')):
    #     response = requests.get(image_path_or_url)
    #     image = Image.open(BytesIO(response.content))
    # else:
    #     image = Image.open(image_path_or_url)

    # Check for objects in video frames
    sample_interval = max(1, nr_frames // 5)  # Sample up to 5 frames
    sampled_indices = range(0, nr_frames, sample_interval)

    print(f"\nChecking for objects: {objs_list}")
    print(f"Sampling {len(sampled_indices)} frames from {nr_frames} total frames\n")

    # Track which objects are detected in which frames
    object_detections = {obj: [] for obj in objs_list}

    for frame_idx in sampled_indices:
        image_path_or_url = video_path
        image = video_frames[frame_idx]

        print(f"Frame {frame_idx}/{nr_frames}...", end=" ")

        # Prepare vision input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path_or_url},
                    {"type": "text", "text": detection_prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        response = processor.decode(outputs[0], skip_special_tokens=True)
        extracted = response.split(detection_prompt)[-1].strip()

        if extracted.startswith("assistant"):
            extracted = extracted[len("assistant") :].strip()

        # Parse detected objects
        detected = extracted.lower().strip()
        if detected != "none" and detected != "none.":
            detected_objs = [obj.strip() for obj in detected.split(",") if obj.strip()]
            # Match detected objects to our list
            for obj in objs_list:
                if any(obj.lower() in d for d in detected_objs):
                    object_detections[obj].append(frame_idx)
            print(f"Detected: {detected_objs}")
        else:
            print("None detected")

    # Print results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    for obj in objs_list:
        frames_detected = object_detections[obj]
        if frames_detected:
            percentage = (len(frames_detected) / len(sampled_indices)) * 100
            print(
                f"✓ {obj}: PRESENT (detected in {len(frames_detected)}/{len(sampled_indices)} frames - {percentage:.1f}%)"
            )
            print(f"  Frames: {frames_detected}")
        else:
            print(f"✗ {obj}: NOT DETECTED")

    print("\n" + "=" * 60)

    # return list of detected objects
    detected_objects = [obj for obj, frames in object_detections.items() if frames]

    print(f"Objects detected in video: {detected_objects}")

    return detected_objects


if __name__ == "__main__":

    root_dir = "/home/stefano/Codebase/stereo4d-code/data"
    #
    split = "test"

    scene = "0TT6XE7fGso"
    timestamp = "17520000"

    # scene = "1ycXgM8obsc"
    # timestamp = "101835169"

    # scene = "H5xOyNqJkPs"
    # timestamp = "38738739"

    # scene = "0D5nD9OfyM0"
    # timestamp = "108508509"

    # List of highest-level, non-overlapping object classes
    objs_list = [
        # Living Entities (Highest Abstraction)
        "person",
        "animal",
        "vehicle",
        # "device",
        # "plant",
        # "furniture",
        # "food",
        # "clothing",
        # "tool",
        # "book"
    ]

    # --- Configuration ---
    # You can change the model name if you have more VRAM (e.g., Qwen/Qwen2.5-VL-14B-Instruct)
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Use "cuda" if you have an NVIDIA GPU, otherwise use "cpu" (inference will be slow)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Model and Processor ---
    print(f"Loading model {MODEL_NAME} to device: {DEVICE}...")

    # Load the model with automatic device mapping for efficient memory usage
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",  # Automatically determines where to load model layers
    ).eval()

    # Load the processor (tokenizer and image pre-processor)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    main(model, processor, root_dir, split, scene, timestamp, objs_list)
