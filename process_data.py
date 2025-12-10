#####################################################################
# Script to process data files using SLURM job arrays.
# process_data.py
# Usage:
#   python process_data.py --source_dir <path_to_source_files> --destination_dir <path_to_save_files> --split <train/test/both> [--unique_scenes]
# Options:
#   --destination_dir: Directory to save processed files (default: "stereo4d-data").
#   --split: Data split to process ("train", "test", or "both"; default: "both").
#   --unique_scenes: If set, only process unique scenes based on identifier (first timestamp).
#   --max_concurrent_jobs: Maximum number of jobs to run concurrently (default: 50).
#   --dry_run: If set, create scripts but don't submit jobs.
#####################################################################

import math
import os
import argparse
import utils
# import subprocess
from datetime import datetime


def create_segmentation_slurm_script(files_to_process, split, source_dir, destination_dir, 
                               script_dir, max_concurrent_jobs, num_files_per_job):
    """Create a SLURM job array script for processing multiple files efficiently.
    
    Args:
        files_to_process: List of files to process
        num_files_per_job: Number of files each array job should process
    """
    
    # Replace <CONDA_ROOT> with your actual conda installation path
    CONDA_ROOT = "/home/geiger/gwb987/.conda"
    CONDA_ENV_NAME = 'sam3'
    PYTHON_EXEC = os.path.join(CONDA_ROOT, 'envs', CONDA_ENV_NAME, 'bin', 'python')
    
    # Create a file list for the job array
    file_list_path = os.path.join(script_dir, f"{split}_files_to_process.txt")
    with open(file_list_path, 'w') as f:
        for filename in files_to_process:
            f.write(f"{filename}\n")
    
    num_total_files = len(files_to_process)
    # Calculate number of array tasks needed
    num_array_tasks = math.ceil(num_total_files / num_files_per_job)
    
    print(f"Total files: {num_total_files}")
    print(f"Files per job: {num_files_per_job}")
    print(f"Number of array tasks: {num_array_tasks}")
    
    base_duration = 5  # minutes per file
    estimated_duration = base_duration * num_files_per_job
    # convert to hh:mm:ss format
    hours = estimated_duration // 60
    minutes = estimated_duration % 60
    time_limit = f"{hours:02}:{minutes:02}:00"
    
    # Create SLURM job array script with throttling
    # Format: --array=0-N%max_concurrent means run jobs 0 to N with max_concurrent running at once
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name="segmentation_{split}"
#SBATCH --partition="geiger"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --array=0-{num_array_tasks-1}%{max_concurrent_jobs}
#SBATCH --output={script_dir}/logs/dataprocessing_{split}_%A_%a.out
#SBATCH --error={script_dir}/logs/dataprocessing_{split}_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=stefano.esposito97@outlook.com

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Calculate the range of files this array task should process
FILES_PER_JOB={num_files_per_job}
START_IDX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB + 1))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * FILES_PER_JOB))
TOTAL_FILES={num_total_files}

# Don't exceed the total number of files
if [ $END_IDX -gt $TOTAL_FILES ]; then
    END_IDX=$TOTAL_FILES
fi

echo "Processing files from line $START_IDX to $END_IDX"

# Process each file assigned to this array task
for LINE_NUMBER in $(seq $START_IDX $END_IDX); do
    FILENAME=$(awk "NR==$LINE_NUMBER" {file_list_path})
    
    # Skip if filename is empty (shouldn't happen, but be safe)
    if [ -z "$FILENAME" ]; then
        echo "Warning: Empty filename at line $LINE_NUMBER"
        continue
    fi
    
    # Remove .npz extension if present
    FILENAME_CLEAN=${{FILENAME%.npz}}
    
    # Split filename into scene and timestamp
    SCENE=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f2- | rev)
    TIMESTAMP=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f1 | rev)
    
    echo ""
    echo "----------------------------------------"
    echo "Processing file $((LINE_NUMBER - START_IDX + 1))/$((END_IDX - START_IDX + 1)): $FILENAME"
    echo "Scene: $SCENE, Timestamp: $TIMESTAMP"
    
    # Check if output already exists (for robustness in case of reruns)
    OUTPUT_FILE="{destination_dir}/stereo4d-disps/{split}/${{FILENAME_CLEAN}}-disps.npz"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file already exists: $OUTPUT_FILE"
        echo "Skipping processing."
    else
        # Run the disparity processing
        {PYTHON_EXEC} run_sam3.py \\
            --data-root="{source_dir}" \\
            --split={split} \\
            --scene="$SCENE" \\
            --timestamp="$TIMESTAMP"
            
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "Successfully processed: $FILENAME"
        else
            echo "Error processing: $FILENAME"
            exit 1
        fi
    fi
done
"""

    slurm_script += """

echo ""
echo "----------------------------------------"
echo "All files processed for this array task"
echo "End time: $(date)"
"""
    
    # Write SLURM script to file
    script_path = os.path.join(script_dir, f"process_{split}.sh")
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created SLURM script: {script_path}")
    print(f"Created file list: {file_list_path}")
    
    return script_path, file_list_path


def create_disparities_slurm_script(files_to_process, split, source_dir, destination_dir, 
                               foundation_stereo_root, script_dir, max_concurrent_jobs, num_files_per_job):
    """Create a SLURM job array script for processing multiple files efficiently.
    
    Args:
        files_to_process: List of files to process
        num_files_per_job: Number of files each array job should process
    """
    
    # Replace <CONDA_ROOT> with your actual conda installation path
    CONDA_ROOT = "/home/geiger/gwb987/.conda"
    CONDA_ENV_NAME = 'foundation_stereo'
    PYTHON_EXEC = os.path.join(CONDA_ROOT, 'envs', CONDA_ENV_NAME, 'bin', 'python')
    
    # Create a file list for the job array
    file_list_path = os.path.join(script_dir, f"{split}_files_to_process.txt")
    with open(file_list_path, 'w') as f:
        for filename in files_to_process:
            f.write(f"{filename}\n")
    
    num_total_files = len(files_to_process)
    # Calculate number of array tasks needed
    num_array_tasks = math.ceil(num_total_files / num_files_per_job)
    
    print(f"Total files: {num_total_files}")
    print(f"Files per job: {num_files_per_job}")
    print(f"Number of array tasks: {num_array_tasks}")
    
    base_duration = 5  # minutes per file
    estimated_duration = base_duration * num_files_per_job
    # convert to hh:mm:ss format
    hours = estimated_duration // 60
    minutes = estimated_duration % 60
    time_limit = f"{hours:02}:{minutes:02}:00"
    
    # Create SLURM job array script with throttling
    # Format: --array=0-N%max_concurrent means run jobs 0 to N with max_concurrent running at once
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name="disparity_{split}"
#SBATCH --partition="geiger"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --array=0-{num_array_tasks-1}%{max_concurrent_jobs}
#SBATCH --output={script_dir}/logs/dataprocessing_{split}_%A_%a.out
#SBATCH --error={script_dir}/logs/dataprocessing_{split}_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=stefano.esposito97@outlook.com

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Calculate the range of files this array task should process
FILES_PER_JOB={num_files_per_job}
START_IDX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB + 1))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * FILES_PER_JOB))
TOTAL_FILES={num_total_files}

# Don't exceed the total number of files
if [ $END_IDX -gt $TOTAL_FILES ]; then
    END_IDX=$TOTAL_FILES
fi

echo "Processing files from line $START_IDX to $END_IDX"

# Process each file assigned to this array task
for LINE_NUMBER in $(seq $START_IDX $END_IDX); do
    FILENAME=$(awk "NR==$LINE_NUMBER" {file_list_path})
    
    # Skip if filename is empty (shouldn't happen, but be safe)
    if [ -z "$FILENAME" ]; then
        echo "Warning: Empty filename at line $LINE_NUMBER"
        continue
    fi
    
    # Remove .npz extension if present
    FILENAME_CLEAN=${{FILENAME%.npz}}
    
    # Split filename into scene and timestamp
    SCENE=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f2- | rev)
    TIMESTAMP=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f1 | rev)
    
    echo ""
    echo "----------------------------------------"
    echo "Processing file $((LINE_NUMBER - START_IDX + 1))/$((END_IDX - START_IDX + 1)): $FILENAME"
    echo "Scene: $SCENE, Timestamp: $TIMESTAMP"
    
    # Check if output already exists (for robustness in case of reruns)
    OUTPUT_FILE="{destination_dir}/stereo4d-disps/{split}/${{FILENAME_CLEAN}}-disps.npz"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file already exists: $OUTPUT_FILE"
        echo "Skipping processing."
    else
        # Run the disparity processing
        {PYTHON_EXEC} {foundation_stereo_root}/scripts/run_video.py \\
            --scene="$SCENE" \\
            --timestamp="$TIMESTAMP" \\
            --left_file={source_dir}/stereo4d-lefteye-perspective/{split}_mp4s/${{FILENAME_CLEAN}}-left_rectified.mp4 \\
            --right_file={source_dir}/stereo4d-righteye-perspective/{split}_mp4s/${{FILENAME_CLEAN}}-right_rectified.mp4 \\
            --ckpt_dir={foundation_stereo_root}/pretrained_models/23-51-11/model_best_bp2.pth \\
            --out_dir={destination_dir}/stereo4d-disps/{split}
            
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "Successfully processed: $FILENAME"
        else
            echo "Error processing: $FILENAME"
            exit 1
        fi
    fi
done
"""

    slurm_script += """

echo ""
echo "----------------------------------------"
echo "All files processed for this array task"
echo "End time: $(date)"
"""
    
    # Write SLURM script to file
    script_path = os.path.join(script_dir, f"process_{split}.sh")
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created SLURM script: {script_path}")
    print(f"Created file list: {file_list_path}")
    
    return script_path, file_list_path


def create_segmentation_slurm(files_to_process, split, source_dir, destination_dir, 
                           max_concurrent_jobs, max_array_size):
    if not files_to_process:
        print(f"No files to process for split: {split}")
        return None
    
    num_files = len(files_to_process)
    print(f"Setting up SLURM job array for {split} split")
    print(f"Total number of files to process: {num_files}")
    print(f"Max array size allowed: {max_array_size}")
    
    # Calculate how many files each job should process
    # This ensures we don't exceed the max_array_size limit
    num_files_per_job = math.ceil(num_files / max_array_size)
    num_array_tasks = math.ceil(num_files / num_files_per_job)
    
    print(f"Files per job: {num_files_per_job}")
    print(f"Number of array tasks: {num_array_tasks}")
    print(f"Max concurrent jobs: {max_concurrent_jobs}")
    
    # Check if we need batching
    if num_array_tasks > max_array_size:
        print(f"\n⚠️  WARNING: Calculated array tasks ({num_array_tasks}) exceeds max_array_size ({max_array_size})")
        print("This should not happen. Adjusting files_per_job...")
        num_files_per_job = math.ceil(num_files / max_array_size)
        num_array_tasks = math.ceil(num_files / num_files_per_job)
        print(f"Adjusted - Files per job: {num_files_per_job}, Array tasks: {num_array_tasks}")
        
    # Create script directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.join(os.getcwd(), "slurm_jobs", f"segmentation_{split}_{timestamp}")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Script directory: {script_dir}")
    
    # Create SLURM script
    script_path, file_list_path = create_segmentation_slurm_script(
        files_to_process, split, source_dir, destination_dir,
        script_dir, max_concurrent_jobs, num_files_per_job,
    )
    
    print(f"\n{'='*60}")
    print("✓ SLURM job array script created successfully!")
    print(f"{'='*60}")
    print("\nTo submit the job, run:")
    print(f"  sbatch {script_path}")
    print("\nTo monitor the job:")
    print("  squeue -u $USER")
    print("\nLog files will be in:")
    print(f"  {logs_dir}/")
    
    return script_path


def create_disparities_slurm(files_to_process, split, source_dir, destination_dir, 
                           foundation_stereo_root, max_concurrent_jobs, 
                           max_array_size):
    """Create a SLURM job array for processing multiple files.
    
    Args:
        max_array_size: Maximum number of array tasks allowed (default 40 for SLURM limits)
    """
    
    if not files_to_process:
        print(f"No files to process for split: {split}")
        return None
    
    num_files = len(files_to_process)
    print(f"Setting up SLURM job array for {split} split")
    print(f"Total number of files to process: {num_files}")
    print(f"Max array size allowed: {max_array_size}")
    
    # Calculate how many files each job should process
    # This ensures we don't exceed the max_array_size limit
    num_files_per_job = math.ceil(num_files / max_array_size)
    num_array_tasks = math.ceil(num_files / num_files_per_job)
    
    print(f"Files per job: {num_files_per_job}")
    print(f"Number of array tasks: {num_array_tasks}")
    print(f"Max concurrent jobs: {max_concurrent_jobs}")
    
    # Check if we need batching
    if num_array_tasks > max_array_size:
        print(f"\n⚠️  WARNING: Calculated array tasks ({num_array_tasks}) exceeds max_array_size ({max_array_size})")
        print("This should not happen. Adjusting files_per_job...")
        num_files_per_job = math.ceil(num_files / max_array_size)
        num_array_tasks = math.ceil(num_files / num_files_per_job)
        print(f"Adjusted - Files per job: {num_files_per_job}, Array tasks: {num_array_tasks}")
    
    # Create script directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.join(os.getcwd(), "slurm_jobs", f"disparity_{split}_{timestamp}")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Script directory: {script_dir}")
    
    # Create SLURM script
    script_path, file_list_path = create_disparities_slurm_script(
        files_to_process, split, source_dir, destination_dir,
        foundation_stereo_root, script_dir, max_concurrent_jobs, num_files_per_job,
    )
    
    print(f"\n{'='*60}")
    print("✓ SLURM job array script created successfully!")
    print(f"{'='*60}")
    print("\nTo submit the job, run:")
    print(f"  sbatch {script_path}")
    print("\nTo monitor the job:")
    print("  squeue -u $USER")
    print("\nLog files will be in:")
    print(f"  {logs_dir}/")
    
    return script_path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process data files using SLURM job arrays.")
    parser.add_argument("--source-dir", type=str, default="data", help="Directory containing source files to process.")
    parser.add_argument("--destination-dir", type=str, default="data", help="Directory to save processed files.")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both", help="Data split to process.")
    parser.add_argument("--unique-scenes", action="store_true", help="If set, only process unique scenes based on identifier (first timestamp).")
    parser.add_argument("--foundation-stereo-root", type=str, default="/home/geiger/gwb987/work/codebase/FoundationStereo", help="Directory for foundation stereo data.")
    parser.add_argument("--max-concurrent-jobs", type=int, default=10, help="Maximum number of jobs to run concurrently (default: 10).")
    parser.add_argument("--max-array-size", type=int, default=40, help="Maximum job array size to avoid SLURM limits (default: 40).")
    # parser.add_argument("--dry-run", action="store_true", help="If set, create scripts but don't submit jobs.")
    args = parser.parse_args()
    
    DESTINATION_DIR = args.destination_dir
    SOURCE_DIR = args.source_dir

    # Verify foundation stereo root exists
    if not os.path.exists(args.foundation_stereo_root):
        raise FileNotFoundError(f"Foundation Stereo root directory {args.foundation_stereo_root} does not exist.")

    # Determine which splits to process
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    # job_ids = []
    
    for split in splits:
        
        # Check if split csv file already exists
        csv_filename = f"{SOURCE_DIR}/{split}_file_list.csv"
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"CSV file {csv_filename} does not exist. Please run download_data.py first to generate the file list.")
        
        # Read file list
        with open(csv_filename, "r") as f:
            lines = f.readlines()[1:]  # skip header
            files = [line.split(",")[0] for line in lines]
            # remove newline characters
            files = [file.strip() for file in files]
            
        print(f"\nNumber of files found in {split} split: {len(files)}")
        
        if args.unique_scenes:
            # filter files to only include unique scenes based on identifier
            files = utils.get_unique_scenes(files)
            print(f"Number of unique scene files in {split} split: {len(files)}")

        # DISPARITY PROCESSING

        # Filter out already processed files
        files_to_process = []
        for filename in files:
            # check if file has been processed (is in destination directory)
            file_path = f"{DESTINATION_DIR}/stereo4d-disps/{split}/{filename}"
            # remove .npz extension for checking
            if file_path.endswith(".npz"):
                file_path = file_path[:-4]
            file_path += "-disps.npz"
            
            if not os.path.exists(file_path):
                files_to_process.append(filename)
        
        print(f"Number of files already processed: {len(files) - len(files_to_process)}")
        print(f"Number of files to process: {len(files_to_process)}")
        
        if not files_to_process:
            print(f"All files in {split} split are already disparity processed. Skipping.")
            continue
        
        # Submit SLURM job array (may return single job ID or list of IDs for batched submission)
        create_disparities_slurm(
            files_to_process=files_to_process,
            split=split,
            source_dir=SOURCE_DIR,
            destination_dir=DESTINATION_DIR,
            foundation_stereo_root=args.foundation_stereo_root,
            max_concurrent_jobs=args.max_concurrent_jobs,
            max_array_size=args.max_array_size
        )
    
        # SEGMENTATION PROCESSING
        
        # Filter out already processed files
        files_to_process = []
        for filename in files:
            # check if file has been processed (is in destination directory)
            file_path = f"{DESTINATION_DIR}/stereo4d-sam3/{split}/{filename}"
            # remove .npz extension for checking
            if file_path.endswith(".npz"):
                file_path = file_path[:-4]
            file_path += "-sam3.npz"
            
            if not os.path.exists(file_path):
                files_to_process.append(filename)
        
        print(f"Number of files already processed: {len(files) - len(files_to_process)}")
        print(f"Number of files to process: {len(files_to_process)}")
        
        if not files_to_process:
            print(f"All files in {split} split are already SAM3 processed. Skipping.")
            continue
        
        # Submit SLURM job array (may return single job ID or list of IDs for batched submission)
        create_segmentation_slurm(
            files_to_process=files_to_process,
            split=split,
            source_dir=SOURCE_DIR,
            destination_dir=DESTINATION_DIR,
            max_concurrent_jobs=args.max_concurrent_jobs,
            max_array_size=args.max_array_size
        )
        
        
    # TODO Stefano: process 2D tracks
    
    print("\nScripts created but not submitted.")