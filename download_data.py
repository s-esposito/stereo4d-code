#####################################################################
# Script to download data files from a Google Cloud Storage bucket.
# download_data.py
# Usage:
#   python download_data.py --destination_dir <path_to_save_files> --split <train/test/both> [--unique_scenes]
# Options:
#   --destination_dir: Directory to save downloaded files (default: "stereo4d-npz").
#   --split: Data split to download ("train", "test", or "both"; default: "both").
#   --unique_scenes: If set, only download unique scenes based on identifier (first timestamp).
#####################################################################

import os
import argparse
from tqdm import tqdm
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import utils


    
def list_gcs_files(bucket_name, split) -> list[str]:
    """
    Lists all files (blobs) in a Google Cloud Storage bucket
    that match the specified prefix.

    Args:
        bucket_name (str): The name of the GCS bucket (e.g., 'stereo4d').
        split (str): The path/folder within the bucket to search (e.g., 'train').
    Returns:
        list[str]: A list of all object names (blobs) found under the given prefix.
                   Returns an empty list if access fails.
    """
    file_list = []
    
    prefix = split if split.endswith('/') else split + '/'
    
    try:
        # Create an anonymous client. This client does not attempt to authenticate.
        storage_client = storage.Client.create_anonymous_client()
        
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        print(f"--- Searching for files in gs://{bucket_name}/{prefix} ---")
        for blob in blobs:
            # Exclude the "folder" name itself if it's returned
            if blob.name != prefix:
                # remove the prefix from the blob name for cleaner output
                cleaned_name = blob.name[len(prefix):]
                file_list.append(cleaned_name)
        
        print(f"\n--- Anonymous listing complete. Total files found: {len(file_list)} ---")
        return file_list # Success, return the list

    except Exception as e:
        # If anonymous access fails (e.g., 401 Unauthorized or 403 Forbidden), 
        # we proceed to authenticated attempt.
        if '403 Forbidden' in str(e) or 'Unauthorized' in str(e):
            print("Anonymous access failed. Bucket requires authentication.")
        else:
            # Handle other errors like network issues or bucket not found
            print(f"Anonymous access failed due to an unexpected error: {type(e).__name__}: {e}")
        return file_list

def cleanup_temp(temp_file):
    """Helper to safely remove the temporary file."""
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"🗑️ Cleaned up interrupted temporary file: {temp_file}")
        except OSError as e:
            print(f"⚠️ Warning: Could not delete temporary file {temp_file}: {e}")

def download_gcs_file(bucket_name, split, filename, destination_dir="."):
    """
    Downloads a specific file from GCS anonymously.

    Args:
        bucket_name (str): The GCS bucket name.
        split (str): The path/folder within the bucket (e.g., 'train').
        filename (str): The base filename (e.g., 'RqV4YTPH5nM_879145812').
    """
    prefix = split if split.endswith('/') else split + '/'
    
    full_blob_name = f"{prefix}{filename}"
    local_destination = f"{destination_dir}/{full_blob_name}"
    # Use a temporary file name to ensure atomicity
    temp_destination = f"{local_destination}.tmp"
    
    # print(f"Preparing to download gs://{bucket_name}/{full_blob_name} to {local_destination}")
    # make sure destination directory exists
    os.makedirs(os.path.dirname(local_destination), exist_ok=True)
    
    try:
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(full_blob_name)
        
        try:
            # 1. Download to temporary file
            blob.download_to_filename(temp_destination)
            
            # 2. Atomic rename to final destination (only runs if download succeeded)
            os.rename(temp_destination, local_destination)
            return True

        except Exception as e:
            cleanup_temp(temp_destination) # Clean up partial file on any download error
            
            error_msg = str(e)
            if '403 Forbidden' in error_msg or 'Unauthorized' in error_msg or 'AccessDenied' in error_msg:
                print("Anonymous access failed due to GCS permissions. Proceeding to Attempt 2.")
                # Fall through to the next block
            else:
                # Other errors (network, interruption, etc.)
                print(f"Anonymous download failed due to an unexpected error: {type(e).__name__}: {e}")
            return False

    except Exception as e:
        print(f"Error during Anonymous client setup: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Download data from Google Cloud Storage.")
    parser.add_argument("--destination_dir", type=str, default="stereo4d-npz", help="Directory to save downloaded files.")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both", help="Data split to download.")
    parser.add_argument("--unique_scenes", action="store_true", help="If set, only download unique scenes based on identifier (first timestamp).")
    args = parser.parse_args()
    
    DESTINATION_DIR = args.destination_dir
    
    # Extracted from the provided URL:
    # https://console.cloud.google.com/storage/browser/stereo4d/train/...
    GCS_BUCKET_NAME = "stereo4d"
    
    # The prefix (folder path) must end with a forward slash for accurate folder listing.
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    for split in splits:
        
        # clean tmp files in destination directory
        for root, dirs, files in os.walk(f"{DESTINATION_DIR}/{split}"):
            for file in files:
                if file.endswith(".tmp"):
                    temp_file_path = os.path.join(root, file)
                    cleanup_temp(temp_file_path)
        
        # check if split csv file already exists
        csv_filename = f"{DESTINATION_DIR}/{split}_file_list.csv"
        if os.path.exists(csv_filename):
            
            print(f"CSV file {csv_filename} already exists. Skipping listing for {split} split.")
            
            with open(csv_filename, "r") as f:
                lines = f.readlines()[1:]  # skip header
                files = [line.split(",")[0] for line in lines]
                # remove newline characters
                files = [file.strip() for file in files]
        else:
            
            print(f"CSV file {csv_filename} does not exist. Proceeding to list files for {split} split.")
        
            files = list_gcs_files(GCS_BUCKET_NAME, split)
            # sort files alphabetically
            files.sort()

            # create a csv file to save the list of files (one per line)
            csv_filename = f"{DESTINATION_DIR}/{split}_file_list.csv"
            with open(csv_filename, "w") as f:
                f.write("filename\n")
                for file in files:
                    f.write(f"{file}\n")

        # print the number of files found
        print(f"Number of files found in {split} split: {len(files)}")
        
        if args.unique_scenes:
            # filter files to only include unique scenes based on identifier
            files = utils.get_unique_scenes(files)
            print(f"Number of unique scene files in {split} split: {len(files)}")
        
        # iterate over files and download each one if not already downloaded
        for i, filename in tqdm(enumerate(files), total=len(files)):

            # check if file has been downloaded (is in destination directory)
            file_path = f"{DESTINATION_DIR}/{split}/{filename}"

            if os.path.exists(f"{file_path}"):
                # print(f"File {filename} already exists in {file_path}. Skipping download.")
                pass
            else:
                # print(f"File {filename} does not exist in {file_path}. Proceeding to download.")
                success = download_gcs_file(GCS_BUCKET_NAME, split, filename, DESTINATION_DIR)
                if success:
                #     print(f"✅ Successfully downloaded {filename}.")
                    pass
                else:
                    raise Exception(f"Failed to download file: {filename}.")
                
                    # print(f"Failed to download file: {filename}.")
                    # return 1
                # else:
                    # print(f"Downloaded file {i+1}/{len(files)}: {filename}")
                    
            # break files loop
            # break