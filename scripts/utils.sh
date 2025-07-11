#!/bin/bash

# Returns the current PST timestamp in month day year, hour:minute:second AM/PM
# format.
get_timestamp() {
    TZ="America/Los_Angeles" date +"%b %d, %Y %I:%M:%S%p"
}

# Prints timestamped info messages to stdout
log_info() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*"
}

# Prints timestamped error messages to stderr
log_error() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*" >&2
}

# Deletes lightning logs from previous runs
delete_lightning_logs() {
    if [ -d "lightning_logs" ]; then
        rm -rf lightning_logs
    fi
}

# Deletes wandb logs from previous runs
delete_wandb_logs() {
    if [ -d "wandb" ]; then
        rm -rf wandb
    fi
}

# Logs the following slurm job information:
# - `SLURM_JOB_NAME`: The job's name.
# - `SLURM_JOB_ID`: The unique job ID assigned by SLURM if the job is part of an
#   array. E.g. 123456_0 for task 0, 123456_1 for task 1, etc.
# - `SLURM_ARRAY_JOB_ID`: The array job's ID if the job's part of an array. Same
#   for all tasks in the array. E.g. 123456.
# - `SLURM_ARRAY_TASK_ID`: The specific task index within the array job (if
#   applicable). E.g. 0 for task 0, 1 for task 1, etc.
# - `SLURM_GPUS_ON_NODE`: Number of devices (e.g. GPUs) used per each node.
# - `SLURM_JOB_NUM_NODES`: Number of nodes (i.e. machines) used for the job.
log_job_info() {
    # Check if SLURM variables are set. Otherwise, use "N/A".
    {
        echo -e "Field\t|\tValue"
        echo -e "-----\t|\t-----"
        echo -e "Job name\t|\t${SLURM_JOB_NAME:-N/A}"
        echo -e "Job ID\t|\t${SLURM_JOB_ID:-N/A}"
        echo -e "Array job ID\t|\t${SLURM_ARRAY_JOB_ID:-N/A}"
        echo -e "Array task ID\t|\t${SLURM_ARRAY_TASK_ID:-N/A}"
        echo -e "Devices per node\t|\t${SLURM_GPUS_ON_NODE:-N/A}"
        echo -e "Number of nodes\t|\t${SLURM_JOB_NUM_NODES:-N/A}"
    } | column -t -s $'\t'
}

# Returns a "done" directory path for marking job completion and ensures the
# directory exists using `mkdir -p`.
#
# The directory path depends on whether the job is part of a SLURM array:
# - If part of an array, the path is:
#     ./logs/<SLURM_JOB_NAME>/done/<SLURM_ARRAY_JOB_ID>
# - Else, the path is:
#     ./logs/<SLURM_JOB_NAME>/done
#
# - `SLURM_JOB_NAME`: The name of the SLURM job.
# - `SLURM_ARRAY_JOB_ID`: The array job ID, if applicable.
get_done_dir() {
    local base_dir="./logs/${SLURM_JOB_NAME}/done"

    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        done_dir="${base_dir}/${SLURM_ARRAY_JOB_ID}"
    else
        done_dir="$base_dir"
    fi

    mkdir -p "$done_dir"
    echo "$done_dir"
}

# Returns a file path to a "done" file to mark completion of a SLURM job or
# array task and ensures the file exists using `touch`.
#
# The file name depends on whether the job is part of a SLURM array:
# - If part of an array, the file is named:
#     <SLURM_ARRAY_TASK_ID>.done
# - Else, the file is named:
#     <SLURM_JOB_ID>.done
#
# The file is placed inside the "done" directory created by `create_done_dir`.
#
# - `SLURM_JOB_ID`: The unique SLURM job ID.
# - `SLURM_ARRAY_JOB_ID`: The ID shared across all array tasks, if applicable.
# - `SLURM_ARRAY_TASK_ID`: The index of the array task, if applicable.
# - `SLURM_JOB_NAME`: The name of the SLURM job (used to determine directory
# path).
get_done_file() {
    local done_dir
    done_dir=$(get_done_dir)

    local done_file
    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        done_file="${SLURM_ARRAY_TASK_ID}.done"
    else
        done_file="${SLURM_JOB_ID}.done"
    fi

    local done_path="${done_dir}/${done_file}"
    touch "$done_path"
    echo "$done_path"
}
