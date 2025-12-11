# script to download mp4 files from a remote cluster when missing locally
import os
import utils

SSH_NAME = "galvani"  # saved in ssh config
CLUSTER_DATA_ROOT = "/home/geiger/gwb987/work/data/stereo4d/stereo4d-code/data"

if __name__ == "__main__":
    
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
        
        for f in files:
            scene_timestamp = f[:-4]  # remove .npz
            
            print(f"Processing {scene_timestamp}")
            
            left_mp4_path = os.path.join(lefteye_dir, f"{scene_timestamp}-left_rectified.mp4")
            right_mp4_path = os.path.join(righteye_dir, f"{scene_timestamp}-right_rectified.mp4")
            npz_path = os.path.join(npz_dir, f"{scene_timestamp}.npz")
            disps_path = os.path.join(disps_dir, f"{scene_timestamp}-disps.npz")
            sam3_path = os.path.join(sam3_dir, f"{scene_timestamp}-sam3.npz")
            
            if not os.path.exists(left_mp4_path):
                # download from cluster
                remote_path = f"{SSH_NAME}:{os.path.join(CLUSTER_DATA_ROOT, 'stereo4d-lefteye-perspective', f'{split}_mp4s', f'{scene_timestamp}-left_rectified.mp4')}"
                os.system(f"scp {remote_path} {left_mp4_path}")
                # check for successful download
                if os.path.exists(left_mp4_path):
                    print(f"Downloaded {left_mp4_path}")
                else:
                    print(f"Failed to download {left_mp4_path}")
            else:
                print(f"{left_mp4_path} already exists.")
                
            if not os.path.exists(right_mp4_path):
                # download from cluster
                remote_path = f"{SSH_NAME}:{os.path.join(CLUSTER_DATA_ROOT, 'stereo4d-righteye-perspective', f'{split}_mp4s', f'{scene_timestamp}-right_rectified.mp4')}"
                os.system(f"scp {remote_path} {right_mp4_path}")
                # check for successful download
                if os.path.exists(right_mp4_path):
                    print(f"Downloaded {right_mp4_path}")
                else:
                    print(f"Failed to download {right_mp4_path}")
            else:
                print(f"{right_mp4_path} already exists.")
                
            if not os.path.exists(npz_path):
                # download from cluster
                remote_path = f"{SSH_NAME}:{os.path.join(CLUSTER_DATA_ROOT, 'stereo4d-npz', split, f'{scene_timestamp}.npz')}"
                os.system(f"scp {remote_path} {npz_path}")
                # check for successful download
                if os.path.exists(npz_path):
                    print(f"Downloaded {npz_path}")
                else:
                    print(f"Failed to download {npz_path}")
            else:
                print(f"{npz_path} already exists.")
                
            if not os.path.exists(disps_path):
                # download from cluster
                remote_path = f"{SSH_NAME}:{os.path.join(CLUSTER_DATA_ROOT, 'stereo4d-disps', split, f'{scene_timestamp}-disps.npz')}"
                os.system(f"scp {remote_path} {disps_path}")
                # check for successful download
                if os.path.exists(disps_path):
                    print(f"Downloaded {disps_path}")
                else:
                    print(f"Failed to download {disps_path}")
            else:
                print(f"{disps_path} already exists.")
                
            if not os.path.exists(sam3_path):
                # download from cluster
                remote_path = f"{SSH_NAME}:{os.path.join(CLUSTER_DATA_ROOT, 'stereo4d-sam3', split, f'{scene_timestamp}-sam3.npz')}"
                os.system(f"scp {remote_path} {sam3_path}")
                # check for successful download
                if os.path.exists(sam3_path):
                    print(f"Downloaded {sam3_path}")
                else:
                    print(f"Failed to download {sam3_path}")
            else:
                print(f"{sam3_path} already exists.")
                
            # break  # for testing, remove this line to process all files