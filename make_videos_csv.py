import os
import utils
import numpy as np

if __name__ == "__main__":

    # list all files in data/stereo4d-disps
    
    # csv output path
    csv_output_path = "videogallery/videos.csv"
    
    csv_lines = []
    for split in ["test"]: # ["train", "test"]:
        
        # check if split csv file already exists
        csv_filename = f"data/{split}_file_list.csv"
        if os.path.exists(csv_filename):
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
        
        sam3_dir = os.path.join("data", "stereo4d-sam3", split)
        
        # get first 100 files only for testing
        files = files[:100]

        for f in files:
            
            
            filename = f[:-4]  # remove .npz
            
            line = f"{filename}"
            
            
            # make sure sam3 file is present
            sam_path = os.path.join(sam3_dir, f"{filename}-sam3.npz")
            if not os.path.exists(sam_path):
                print(f"SAM3D file for {filename} is missing.")
            else:
                # load sam3 npz file to check for classes
                sam_data = np.load(sam_path, allow_pickle=True)
                print("Loaded SAM3D data keys:", sam_data.keys())
                classes = sam_data.keys()
                if classes is not None and len(classes) > 0:
                    classes_str = ",".join(classes)
                    line += f",{classes_str}"
                else:
                    line += ","

            csv_lines.append(line)
            
    # write to csv file (overwrite if exists)
    with open(csv_output_path, "w") as f:
        for line in csv_lines:
            f.write(line + "\n")
    print(f"Wrote {len(csv_lines)} lines to {csv_output_path}")
