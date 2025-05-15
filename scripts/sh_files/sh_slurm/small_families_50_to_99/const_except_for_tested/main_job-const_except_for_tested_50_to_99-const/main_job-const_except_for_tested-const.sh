#!/bin/bash

#SBATCH --job-name=small_fam_const
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested_50_to_99/main_job-const_except_for_tested_50_to_99-const/output.OU
#SBATCH --error=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested_50_to_99/main_job-const_except_for_tested_50_to_99-const/err.ER

source ~/.bashrc
hostname
conda activate chromevol
export PATH=$CONDA_PREFIX/bin:$PATH

cd /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested_50_to_99/main_job-const_except_for_tested_50_to_99-const/

python /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/create_pfiles_and_run_jobs.py \
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_50_to_99/" \
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results_const_except_for_tested_50_to_99/" \
    --function_type_dir allConst \
    --transitions_dir gain loss dupl demiDupl baseNum allConst \
    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc allConst \
    --function_types linear exponential log-normal reverse-sigmoid linear-bd ignore \
    --functions_param LINEAR EXP LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE \
    --transitions_abbr "{'gain': 'g', 'loss': 'l', 'dupl': 'dup', 'demiDupl': 'dem', 'baseNum': 'bN', 'allConst': 'const'}" \
    --functions_abbr "{'linear': 'lin', 'exponential': 'exp', 'log-normal': 'logN', 'reverse-sigmoid': 'revS', 'linear-bd': 'BD', 'ignore': 'ign'}" \
    > /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested_50_to_99/main_job-const_except_for_tested_50_to_99-const/log.txt \
    2> /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested_50_to_99/main_job-const_except_for_tested_50_to_99-const/err.txt
