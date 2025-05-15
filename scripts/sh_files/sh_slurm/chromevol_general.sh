#!/bin/bash

#SBATCH --job-name=<your_job_name>
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1 <change according to your number of tasks>
#SBATCH --cpus-per-task=1 <change according to your cpu per task>
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/groups/itay_mayrose/noybandel/ChromEvol_project/<task_directory>/output.OU
#SBATCH --error=/groups/itay_mayrose/noybandel/ChromEvol_project/<task_directory>/err.ER

# Load environment and set paths
source ~/.bashrc
hostname
conda activate chromevol
export PATH=$CONDA_PREFIX/bin:$PATH

# Navigate to the working directory
cd /groups/itay_mayrose/noybandel/ChromEvol_project/<task_directory>

# Run ChromEvol with the parameter file, and add Enter at the end
/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_program/chromevol/ChromEvol/chromEvol param="/groups/itay_mayrose/noybandel/ChromEvol_project/<param_file_directory>/paramFile" > /groups/itay_mayrose/noybandel/ChromEvol_project/<task_directory>/log.txt 2> /groups/itay_mayrose/noybandel/ChromEvol_project/<task_directory>/err.txt
