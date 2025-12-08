#####################################################################
# Script to download data files from a Google Cloud Storage bucket.
# process_data.py
# Usage:
#   python process_data.py --source_dir <path_to_source_files> --destination_dir <path_to_save_files> --split <train/test/both> [--unique_scenes]
# Options:
#   --destination_dir: Directory to save processed files (default: "stereo4d-data").
#   --split: Data split to process ("train", "test", or "both"; default: "both").
#   --unique_scenes: If set, only process unique scenes based on identifier (first timestamp).
#####################################################################

import os
import argparse
from tqdm import tqdm
import utils


def process_file(filename, split, source_dir, destination_dir):
    return True



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process data files.")
    parser.add_argument("--source_dir", type=str, default="stereo4d-npz", help="Directory containing source files to process.")
    parser.add_argument("--destination_dir", type=str, default="stereo4d-data", help="Directory to save processed files.")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both", help="Data split to process.")
    parser.add_argument("--unique_scenes", action="store_true", help="If set, only process unique scenes based on identifier (first timestamp).")
    args = parser.parse_args()
    
    DESTINATION_DIR = args.destination_dir
    SOURCE_DIR = args.source_dir

    # The prefix (folder path) must end with a forward slash for accurate folder listing.
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    for split in splits:
        
        # # clean tmp files in destination directory
        # for root, dirs, files in os.walk(f"{DESTINATION_DIR}/{split}"):
        #     for file in files:
        #         if file.endswith(".tmp"):
        #             temp_file_path = os.path.join(root, file)
        #             cleanup_temp(temp_file_path)
        
        
        # check if split csv file already exists
        csv_filename = f"{SOURCE_DIR}/{split}_file_list.csv"
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"CSV file {csv_filename} does not exist. Please run download_data.py first to generate the file list.")
        
        with open(csv_filename, "r") as f:
            lines = f.readlines()[1:]  # skip header
            files = [line.split(",")[0] for line in lines]
            # remove newline characters
            files = [file.strip() for file in files]
            
        # print the number of files found
        print(f"Number of files found in {split} split: {len(files)}")
        
        if args.unique_scenes:
            # filter files to only include unique scenes based on identifier
            files = utils.get_unique_scenes(files)
            print(f"Number of unique scene files in {split} split: {len(files)}")
            
        # iterate over files and process each one if not already processed
        for i, filename in tqdm(enumerate(files), total=len(files)):
            
            # check if file has been processed (is in destination directory)
            file_path = f"{DESTINATION_DIR}/{split}/{filename}"
            
            if os.path.exists(f"{file_path}"):
                pass
            else:
                # process the file
                success = process_file(filename, split, SOURCE_DIR, DESTINATION_DIR)
                if success:
                    pass
                else:
                    raise Exception(f"Failed to process file: {filename}.")