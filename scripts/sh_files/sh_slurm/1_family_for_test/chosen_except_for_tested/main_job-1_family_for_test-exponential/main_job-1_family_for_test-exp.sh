#!/bin/bash

#SBATCH --job-name=1_fam_chosen_exp
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/chosen_except_for_tested/main_job-1_family_for_test-exponential/output.OU
#SBATCH --error=/groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/chosen_except_for_tested/main_job-1_family_for_test-exponential/err.ER

source ~/.bashrc
hostname
conda activate chromevol
export PATH=$CONDA_PREFIX/bin:$PATH

cd /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/chosen_except_for_tested/main_job-1_family_for_test-exponential/

python /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/create_pfiles_and_run_jobs_for_chosen.py \
    /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_under_k/1_family_for_test/ \
    /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results_test_1_family/chosen_except_for_tested/ \
    --function_type_dir exponential \
    --transitions_dir gain loss dupl demiDupl baseNum \
    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc \
    --function_types linear exponential log-normal reverse-sigmoid linear-bd ignore constant \
    --functions_param LINEAR EXP LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE CONST \
    --transitions_abbr '{"gain": "g", "loss": "l", "dupl": "dup", "demiDupl": "dem", "baseNum": "bN"}' \
    --functions_abbr '{"linear": "lin", "exponential": "exp", "log-normal": "logN", "reverse-sigmoid": "revS", "linear-bd": "linBD", "ignore": "ign", "constant": "con"}' \
    --all_chosen_models_path /groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis_1_family/all_chosen_models.csv \
    --transitions_csv_to_param_dict '{"gain": "_gainFunc", "loss": "_lossFunc", "dupl": "_duplFunc", "demi": "_demiDuplFunc", "baseNum": "_baseNumRFunc"}' \
    > /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/chosen_except_for_tested/main_job-1_family_for_test-exponential/log.txt \
    2> /groups/itay_mayrose/noybandel/ChromEvol_project/scripts/sh_files/sh_slurm/1_family_for_test/chosen_except_for_tested/main_job-1_family_for_test-exponential/err.txt
