#!/bin/bash

#SBATCH --job-name=small_families_50_to_99_chosen_except_for_tested_linear-bd
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/small_families_50_to_99/chosen_except_for_tested/main_job-chosen_except_for_tested-linear-bd/out.OU
#SBATCH --error=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/small_families_50_to_99/chosen_except_for_tested/main_job-chosen_except_for_tested-linear-bd/err.ER

source ~/.bashrc
hostname
conda activate chromevol
export PATH=$CONDA_PREFIX/bin:$PATH

cd /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/small_families_50_to_99/chosen_except_for_tested/main_job-chosen_except_for_tested-linear-bd
python /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/create_pfiles_and_run_jobs_for_chosen.py \
    /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_50_to_99/ \
    /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results_50_to_99/chromevol_raw_results_chosen_except_for_tested_50_to_99/ \
    --function_type_dir linear-bd \
    --transitions_dir gain loss dupl demiDupl baseNum \
    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc \
    --function_types exponential linear log-normal reverse-sigmoid linear-bd ignore const \
    --functions_param EXP LINEAR LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE CONST \
    --transitions_abbr '{"gain": "g", "loss": "l", "dupl": "dup", "demiDupl": "dem", "baseNum": "bN"}' \
    --functions_abbr '{"linear": "lin", "exponential": "exp", "log-normal": "logN", "reverse-sigmoid": "revS", "linear-bd": "linBD", "ignore": "ign", "const": "con"}' \
    --all_chosen_models_path /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis_50_to_99/const_except_for_tested/all_chosen_models.csv \
    --transitions_csv_to_param_dict '{"gain": "_gainFunc", "loss": "_lossFunc", "dupl": "_duplFunc", "demi": "_demiDuplFunc", "baseNum": "_baseNumRFunc"}' \
    > /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/small_families_50_to_99/chosen_except_for_tested/main_job-chosen_except_for_tested-linear-bd/log.txt \
    2> /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/small_families_50_to_99/chosen_except_for_tested/main_job-chosen_except_for_tested-linear-bd/err.txt


