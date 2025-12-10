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

import os
import argparse
import utils
import subprocess
from datetime import datetime


def create_slurm_array_script(files_to_process, split, source_dir, destination_dir, 
                               foundation_stereo_root, script_dir, max_concurrent_jobs, 
                               actual_split=None):
    """Create a SLURM job array script for processing multiple files efficiently.
    
    Args:
        actual_split: The actual data split (train/test) for file paths. 
                     If None, uses 'split' (which may include batch suffix).
    """
    
    # Replace <CONDA_ROOT> with your actual conda installation path
    CONDA_ROOT = "/home/geiger/gwb987/.conda"
    CONDA_ENV_NAME = 'foundation_stereo'
    PYTHON_EXEC = os.path.join(CONDA_ROOT, 'envs', CONDA_ENV_NAME, 'bin', 'python')
    
    # Use actual_split for file paths if provided (handles batched submissions)
    data_split = actual_split if actual_split else split
    
    # Create a file list for the job array
    file_list_path = os.path.join(script_dir, f"{split}_files_to_process.txt")
    with open(file_list_path, 'w') as f:
        for filename in files_to_process:
            f.write(f"{filename}\n")
    
    num_jobs = len(files_to_process)
    print("Number of jobs in array:", num_jobs)
    
    # Create SLURM job array script with throttling
    # Format: --array=0-N%max_concurrent means run jobs 0 to N with max_concurrent running at once
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name="dataproc_{split}"
#SBATCH --partition="geiger"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --array=0-{num_jobs-1}%{max_concurrent_jobs}
#SBATCH --output={script_dir}/logs/dataprocessing_{split}_%A_%a.out
#SBATCH --error={script_dir}/logs/dataprocessing_{split}_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=stefano.esposito97@outlook.com

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Get the filename for this array task
LINE_NUMBER=$((${{SLURM_ARRAY_TASK_ID}} + 1))
FILENAME=$(awk "NR==${{LINE_NUMBER}}" {file_list_path})

# Remove .npz extension if present
FILENAME_CLEAN=${{FILENAME%.npz}}

# Split filename into scene and timestamp
SCENE=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f2- | rev)
TIMESTAMP=$(echo "$FILENAME_CLEAN" | rev | cut -d'_' -f1 | rev)

echo "Processing file: $FILENAME"
echo "Scene: $SCENE, Timestamp: $TIMESTAMP"

# Check if output already exists (for robustness in case of reruns)
OUTPUT_FILE="{destination_dir}/stereo4d-disps/{data_split}/${{FILENAME_CLEAN}}-disps.npz"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file already exists: $OUTPUT_FILE"
    echo "Skipping processing."
    exit 0
fi

# Run the disparity processing
{PYTHON_EXEC} {foundation_stereo_root}/scripts/run_video.py \\
    --scene="$SCENE" \\
    --timestamp="$TIMESTAMP" \\
    --left_file={source_dir}/stereo4d-lefteye-perspective/{data_split}_mp4s/${{FILENAME_CLEAN}}-left_rectified.mp4 \\
    --right_file={source_dir}/stereo4d-righteye-perspective/{data_split}_mp4s/${{FILENAME_CLEAN}}-right_rectified.mp4 \\
    --ckpt_dir={foundation_stereo_root}/pretrained_models/23-51-11/model_best_bp2.pth \\
    --out_dir={destination_dir}/stereo4d-disps/{data_split}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Successfully processed: $FILENAME"
else
    echo "Error processing: $FILENAME"
    exit 1
fi

echo "End time: $(date)"
"""
    
    return slurm_script, file_list_path


def submit_slurm_array_job(files_to_process, split, source_dir, destination_dir, 
                           foundation_stereo_root, max_concurrent_jobs, dry_run=False, 
                           max_array_size=300):
    """Submit a SLURM job array for processing multiple files.
    
    Args:
        max_array_size: Maximum number of array tasks allowed (default 300 for typical SLURM limits)
    """
    
    if not files_to_process:
        print(f"No files to process for split: {split}")
        return None
    
    num_files = len(files_to_process)
    
    # Check if we exceed SLURM array size limits
    if num_files > max_array_size:
        print(f"\n{'='*80}")
        print(f"⚠️  WARNING: Too many files ({num_files}) for a single job array")
        print(f"{'='*80}")
        print(f"SLURM typically limits job arrays to {max_array_size} tasks.")
        print(f"You have {num_files} files to process.")
        print(f"\nOptions to proceed:")
        print(f"  1. Process in batches (recommended)")
        print(f"  2. Process already-completed files won't be reprocessed")
        print(f"\nThis script will create multiple job arrays in batches of {max_array_size}.")
        print(f"{'='*80}\n")
        
        # Split into batches
        job_ids = []
        for batch_num, i in enumerate(range(0, num_files, max_array_size), 1):
            batch_files = files_to_process[i:i+max_array_size]
            print(f"Processing batch {batch_num}/{(num_files + max_array_size - 1) // max_array_size}")
            print(f"  Files in this batch: {len(batch_files)}")
            
            job_id = _submit_single_array(
                batch_files, f"{split}_batch{batch_num}", source_dir, destination_dir,
                foundation_stereo_root, max_concurrent_jobs, dry_run, actual_split=split
            )
            if job_id:
                job_ids.append(job_id)
        
        return job_ids if job_ids else None
    
    # Single array submission
    return _submit_single_array(
        files_to_process, split, source_dir, destination_dir,
        foundation_stereo_root, max_concurrent_jobs, dry_run
    )


def _submit_single_array(files_to_process, split, source_dir, destination_dir,
                         foundation_stereo_root, max_concurrent_jobs, dry_run, actual_split=None):
    """Internal function to submit a single SLURM job array.
    
    Args:
        actual_split: The actual data split (train/test) for file paths. 
                     Needed when split includes batch suffix (e.g., 'test_batch1').
    """
    
    if not files_to_process:
        return None
    
    # Create script directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.join(os.getcwd(), "slurm_jobs", f"{split}_{timestamp}")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Preparing SLURM job array for {split} split")
    print(f"Number of files to process: {len(files_to_process)}")
    print(f"Max concurrent jobs: {max_concurrent_jobs}")
    print(f"Script directory: {script_dir}")
    print(f"{'='*80}\n")
    
    # Create SLURM script
    slurm_script, file_list_path = create_slurm_array_script(
        files_to_process, split, source_dir, destination_dir,
        foundation_stereo_root, script_dir, max_concurrent_jobs, actual_split
    )
    
    # Write SLURM script to file
    script_path = os.path.join(script_dir, f"process_{split}.sh")
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created SLURM script: {script_path}")
    print(f"Created file list: {file_list_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would submit the following command:")
        print(f"  sbatch {script_path}")
        print("\nYou can manually submit the job later with:")
        print(f"  sbatch {script_path}")
        return script_path
    
    # Submit the job array
    command = ['sbatch', script_path]
    print(f"\nSubmitting job array: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"\n{'='*80}")
        print("ERROR: Failed to submit SLURM job array")
        print(f"{'='*80}")
        print(f"Exit code: {result.returncode}")
        print(f"Error message:\n{result.stderr}")
        
        # Provide helpful suggestions based on error type
        if "QOSMaxSubmitJobPerUserLimit" in result.stderr or "job submit limit" in result.stderr:
            print(f"\n⚠️  You've hit the SLURM job submission limit!")
            print("\nPossible solutions:")
            print("  1. Check your current job queue: squeue -u $USER")
            print("  2. Wait for existing jobs to complete or cancel them: scancel -u $USER")
            print("  3. Reduce concurrent jobs: --max_concurrent_jobs 10")
            print(f"  4. The script has been saved and can be submitted later:")
            print(f"     sbatch {script_path}")
            print("\nThe job array is ready but not submitted. You can submit it manually when ready.")
            return None
        elif "No partition specified" in result.stderr or "Invalid partition" in result.stderr:
            print(f"\n⚠️  Partition issue detected!")
            print("\nPossible solutions:")
            print("  1. Check available partitions: sinfo")
            print("  2. Verify 'a100-galvani' is correct for your cluster")
            print("  3. Edit the script if needed and submit manually:")
            print(f"     sbatch {script_path}")
            return None
        else:
            print(f"\n⚠️  Script created but submission failed.")
            print(f"You can try submitting manually:")
            print(f"  sbatch {script_path}")
            return None
    
    # Extract job ID from sbatch output
    if "Submitted batch job" in result.stdout:
        job_id = result.stdout.strip().split()[-1]
        print("\n✓ Job array submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Cancel with: scancel {job_id}")
        print(f"  View logs in: {logs_dir}/")
        return job_id
    else:
        print(f"Warning: Unexpected sbatch output: {result.stdout}")
        return None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process data files using SLURM job arrays.")
    parser.add_argument("--source-dir", type=str, default="data", help="Directory containing source files to process.")
    parser.add_argument("--destination-dir", type=str, default="data", help="Directory to save processed files.")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both", help="Data split to process.")
    parser.add_argument("--unique-scenes", action="store_true", help="If set, only process unique scenes based on identifier (first timestamp).")
    parser.add_argument("--foundation-stereo-root", type=str, default="/home/geiger/gwb987/work/codebase/FoundationStereo", help="Directory for foundation stereo data.")
    parser.add_argument("--max-concurrent-jobs", type=int, default=10, help="Maximum number of jobs to run concurrently (default: 10).")
    parser.add_argument("--max-array-size", type=int, default=40, help="Maximum job array size to avoid SLURM limits (default: 40).")
    parser.add_argument("--dry-run", action="store_true", help="If set, create scripts but don't submit jobs.")
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

    job_ids = []
    
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
            print(f"All files in {split} split are already processed. Skipping.")
            continue
        
        # Submit SLURM job array (may return single job ID or list of IDs for batched submission)
        result = submit_slurm_array_job(
            files_to_process=files_to_process,
            split=split,
            source_dir=SOURCE_DIR,
            destination_dir=DESTINATION_DIR,
            foundation_stereo_root=args.foundation_stereo_root,
            max_concurrent_jobs=args.max_concurrent_jobs,
            dry_run=args.dry_run,
            max_array_size=args.max_array_size
        )
        
        if result:
            # Handle both single job ID and list of job IDs (for batched submissions)
            if isinstance(result, list):
                for idx, jid in enumerate(result, 1):
                    job_ids.append((f"{split}_batch{idx}", jid))
            else:
                job_ids.append((split, result))
        else:
            print(f"\n⚠️  Job for {split} split was not submitted (see error above).")
    
    # Print summary
    if job_ids:
        print(f"\n{'='*80}")
        print("SUBMISSION SUMMARY")
        print(f"{'='*80}")
        for split_name, job_id in job_ids:
            print(f"  {split_name}: Job ID {job_id}")
        print("\nMonitor all jobs: squeue -u $USER")
        print("Cancel all jobs: scancel -u $USER")
        print(f"{'='*80}\n")
    elif args.dry_run:
        print("\n[DRY RUN COMPLETE] Scripts created but not submitted.")
    else:
        print("\n⚠️  No jobs were submitted.")
        print("Check the errors above or verify all files are already processed.")

    # TODO Stefano: process segmentations
    
    # TODO Stefano: process 2D tracks