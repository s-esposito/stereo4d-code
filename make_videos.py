import os
import utils

if __name__ == "__main__":

    # list all files in data/stereo4d-disps
    
    for split in ["test"]: # ["train", "test"]:
        
        # check if split csv file already exists
        csv_filename = f"data/{split}_file_list.csv"
        if os.path.exists(csv_filename):
            
            print(f"CSV file {csv_filename} already exists. Skipping listing for {split} split.")
            
            with open(csv_filename, "r") as f:
                lines = f.readlines()[1:]  # skip header
                files = [line.split(",")[0] for line in lines]
                # remove newline characters
                files = [file.strip() for file in files]
        else:
            raise ValueError(f"CSV file {csv_filename} does not exist.")
        
        # print the number of files found
        print(f"Number of files found in {split} split: {len(files)}")
        
        # if args.unique_scenes:
        # filter files to only include unique scenes based on identifier
        files = utils.get_unique_scenes(files)
        print(f"Number of unique scene files in {split} split: {len(files)}")
        
        # check if corresponding two-eyes mp4 file exists in 
        # stereo4d-lefteye-perspective/{split}_mp4s
        # stereo4d-righteye-perspective/{split}_mp4s
        
        lefteye_dir = os.path.join("data", "stereo4d-lefteye-perspective", f"{split}_mp4s")
        righteye_dir = os.path.join("data", "stereo4d-righteye-perspective", f"{split}_mp4s")
        npz_dir = os.path.join("data", "stereo4d-npz", split)
        disps_dir = os.path.join("data", "stereo4d-disps", split)
        sam3_dir = os.path.join("data", "stereo4d-sam3", split)
        
        # get first 100 files only for testing
        # files = files[:100]

        for f in files:
            
            filename = f[:-4]  # remove .npz
            print(f"Generating video for file {filename}")
            
            # check if video already exists
            video_output_path = os.path.join("videogallery", "videos", f"{filename}.mp4")
            if os.path.exists(video_output_path):
                print(f"Video {video_output_path} already exists. Skipping.")
                continue
            
            # make sure that mp4 files are present
            left_mp4_path = os.path.join(lefteye_dir, f"{filename}-left_rectified.mp4")
            right_mp4_path = os.path.join(righteye_dir, f"{filename}-right_rectified.mp4")
            if not os.path.exists(left_mp4_path) or not os.path.exists(right_mp4_path):
                print(f"MP4 files for {filename} are missing. Please run download_mp4s_from_cluster.py first.")
                continue
            
            # make sure npz file is present
            npz_path = os.path.join(npz_dir, f"{filename}.npz")
            if not os.path.exists(npz_path):
                print(f"NPZ file for {filename} is missing. Please run download_data_from_cluster.py first.")
                continue
            
            # make sure sam3 file is present
            sam3_path = os.path.join(sam3_dir, f"{filename}-sam3.npz")
            if not os.path.exists(sam3_path):
                print(f"SAM3D file for {filename} is missing. Please run download_sam3d_from_cluster.py first.")
                continue
            
            # make sure disps file is present
            disps_path = os.path.join(disps_dir, f"{filename}-disps.npz")
            if not os.path.exists(disps_path):
                print(f"Disps file for {filename} is missing. Please run download_data_from_cluster.py first.")
                continue

            scene = filename.split('_')[0]
            timestamp = filename.split('_')[1]
            print(f"Scene: {scene}, Timestamp: {timestamp}")
            
            # run view_sample.py script to generate videos for each file
            cmd = f"python view_sample.py \
                --root_dir=data \
                    --split={split} \
                        --scene={filename.split('_')[0]} \
                            --timestamp={filename.split('_')[1]} \
                                --output-video-path=videogallery/videos"
            print(f"Running command: {cmd}")
            os.system(cmd)
            
            # break  # for testing, remove this line to process all files
            
