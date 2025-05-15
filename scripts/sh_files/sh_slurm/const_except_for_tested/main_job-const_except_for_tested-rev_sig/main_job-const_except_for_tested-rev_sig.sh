#!/bin/bash

#SBATCH --job-name=all_fam_sig
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested/main_job-const_except_for_tested-rev_sig/output.OU
#SBATCH --error=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested/main_job-const_except_for_tested-rev_sig/err.ER

source ~/.bashrc
hostname
conda activate chromevol
export PATH=$CONDA_PREFIX/bin:$PATH

cd /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/const_except_for_tested/main_job-const_except_for_tested-rev_sig/

python /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/create_pfiles_and_run_jobs.py \
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/all_families_over_100/" \
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/const_except_for_tested/" \
    --function_type_dir reverse-sigmoid \
    --transitions_dir gain loss dupl demiDupl baseNum allConst \
    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc allConst \
    --function_types linear exponential log-normal reverse-sigmoid linear-bd ignore \
    --functions_param LINEAR EXP LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE \
    --transitions_abbr "{'gain': 'g', 'loss': 'l', 'dupl': 'dup', 'demiDupl': 'dem', 'baseNum': 'bN', 'allConst': 'const'}" \
    --functions_abbr "{'linear': 'lin', 'exponential': 'exp', 'log-normal': 'logN', 'reverse-sigmoid': 'revS', 'linear-bd': 'BD', 'ignore': 'ign'}" \
    > /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/main_job-1_family_for_test-rev_sig/log.txt \
    2> /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/main_job-1_family_for_test-rev_sig/err.txt
