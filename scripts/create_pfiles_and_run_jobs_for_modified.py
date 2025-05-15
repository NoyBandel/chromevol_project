import argparse
import time
import ast
from subprocess import Popen
from create_job_files import create_job_format_for_all_jobs
from pathlib import Path
import random
from typing import List, Dict, Tuple
import pandas as pd

CHROMEVOL_EXE = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_program/chromevol/ChromEvol/chromEvol"
CONDA_ENV = "source /groups/itay_mayrose/noybandel/miniconda3/etc/profile.d/conda.sh; conda activate chromevol"
CONDA_EXPORT = "export LD_LIBRARY_PATH=/groups/itay_mayrose/noybandel/miniconda3/envs/chromevol/lib:$LD_LIBRARY_PATH\n"

def create_chrom_dependence_param_file(dataFile_path: str, treeFile_path: str, param_file_path: str, resultsPathDir_path: str, transitions_names_for_csv_and_dir: List[str], family_chosen_models_dict: dict, transitions_names_to_pfile_form: Dict[str,Tuple]) -> None:
    with open(param_file_path, 'w') as file:
        file.write(f"_dataFile = {dataFile_path}\n")
        file.write(f"_treeFile = {treeFile_path}\n")
        file.write(f"_resultsPathDir = {resultsPathDir_path}\n")

        for transition in transitions_names_for_csv_and_dir:
            file.write(f"{transitions_names_to_pfile_form[transition][0]} = {family_chosen_models_dict[f"{transition}_chosen_model"]}\n")

        for i, transition in enumerate(transitions_names_for_csv_and_dir, start=1):
            file.write(f"{transitions_names_to_pfile_form[transition][1]} = {str(i)};{family_chosen_models_dict[f"{transition}_parameters"]}\n")

        file.write(f"_baseNum_1 = 6;{family_chosen_models_dict["chrom_base_number"]}\n")
        file.write("_optimizationMethod = Brent\n")
        file.write("_baseNumOptimizationMethod = Ranges\n")
        file.write("_minChrNum = -1\n")
        file.write("_optimizePointsNum = 10,3,1\n")
        file.write("_optimizeIterNum = 0,2,5\n")
        file.write("_maxParsimonyBound = true\n")
        file.write("_tolParamOptimization = 0.1\n")
        file.write(f"_seed = {random.randint(1, 8000)}\n")
        file.write("_heterogeneousModel = false\n")
        file.write("_backwardPhase = false\n")


def create_and_run_single_job(run_path: Path, param_file_path: str, family_name: str, transition: str, function: str) -> None:
    job_name = f'{family_name}_{transition}_{function}'
    job_path = run_path / "chrom_dependence.sh"
    param_arg = f"\"param={param_file_path}\""
    log_file = run_path / "log.txt"
    err_file = run_path / "ERR.txt"
    cmd = f'{CONDA_ENV}\n{CONDA_EXPORT}{CHROMEVOL_EXE} {param_arg} > {log_file} 2> {err_file}\n'
    memory_str = f"4G"
    job_content = create_job_format_for_all_jobs(str(run_path), job_name, memory_str, "itaym", 1, cmd, 1)
    with open(job_path, 'w') as file:
        file.write(job_content)
    print(f"Submitting job: {job_name}")
    Popen(["sbatch", str(job_path)])

def create_param_files_and_run_jobs(families_source_dir: Path, chromevol_raw_results_dir: Path, tested_function_dir_and_csv_name: str, transitions_names_for_csv_and_dir: List[str], transitions_names_to_pfile_form: Dict[str, Tuple], function_dir_to_pfile_dict: Dict[str, str], all_chosen_models_path: str) -> None:
    family_dirs: list[Path] = [folder for folder in families_source_dir.rglob('*') if folder.is_dir()]
    chosen_models_df = pd.read_csv(all_chosen_models_path, index_col=0)

    for family_source_folder in family_dirs:
        dataFile_path = str(family_source_folder / "counts.fasta")
        treeFile_path = str(family_source_folder / "tree.nwk")
        family_name = str(family_source_folder.name)
        row = chosen_models_df.loc[family_name]
        family_chosen_models_dict = row.to_dict()
        print(f"family_chosen_models_dict: {family_chosen_models_dict}")

        for transition in transitions_names_for_csv_and_dir:
            family_result_folder = chromevol_raw_results_dir / family_name / transition / tested_function_dir_and_csv_name
            resultsPathDir_path = str(family_result_folder / "Results/")
            param_file_path = str(family_result_folder / "paramFile")

            ####### access raw results for the family and transition - can change according to the family size
            transition_raw_results_df = pd.read_csv(f"{transition}_raw_results.csv", index_col=0)
            #######
            tested_function_init_parameters = transition_raw_results_df.loc[family_name, tested_function_dir_and_csv_name]
            tested_function_init_parameters = [str(float(value)) for value in tested_function_init_parameters]
            tested_function_pfile_name = function_dir_to_pfile_dict[tested_function_dir_and_csv_name]
            family_chosen_models_dict[f"{transition}_chosen_model"] = tested_function_pfile_name
            family_chosen_models_dict[f"{transition}_parameters"] = tested_function_init_parameters

            create_chrom_dependence_param_file(dataFile_path, treeFile_path, tested_function_pfile_name, param_file_path, resultsPathDir_path, transitions_names_for_csv_and_dir, family_chosen_models_dict, transitions_names_to_pfile_form)
            #create_and_run_single_job(family_result_folder, param_file_path, family_name, transition, tested_function_dir_and_csv_name)

        time.sleep(3)

def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter files for ChromEvol and submit jobs for execution.")

    parser.add_argument("--families_source_dir", type=Path,
                        help="Path to the directory containing source files for different families.")
    parser.add_argument("--chromevol_raw_results_dir", type=Path,
                        help="Path to the directory where raw ChromEvol results will be stored.")
    parser.add_argument("--tested_function_dir_and_csv_name", type=str,
                        help="Name of the function type directory to be processed.")
    parser.add_argument("--transitions_names_for_csv_and_dir", type=str, nargs="+", required=True,
                        help="List of transition types used in CSV files and directory naming.")
    parser.add_argument("--transitions_names_to_pfile_form", type=str, required=True,
                        help="String representation of a dictionary mapping transition types to their corresponding parameter file format tuples.")
    parser.add_argument("--function_dir_to_pfile_dict", type=str, nargs="+", required=True,
                        help="String representation of a dictionary mapping function types to their corresponding parameter file names.")
    parser.add_argument("--all_chosen_models_path", type=str, required=True,
                        help="Path to the CSV file containing all chosen models and their configurations.")
    ##################
    parser.add_argument("--const_except_for_tested_results_path_list", type=str, required=True,
                        help="Path to the CSV file containing const except for tested results")
    ##################

    args = parser.parse_args()

    transitions_names_to_pfile_form = ast.literal_eval(args.transitions_names_to_pfile_form)

    create_param_files_and_run_jobs(
        families_source_dir=args.families_source_dir,
        chromevol_raw_results_dir=args.chromevol_raw_results_dir,
        tested_function_dir_and_csv_name=args.tested_function_dir_and_csv_name,
        transitions_names_for_csv_and_dir=args.transitions_names_for_csv_and_dir,
        transitions_names_to_pfile_form=transitions_names_to_pfile_form,
        function_dir_to_pfile_dict=args.function_dir_to_pfile_dict,
        all_chosen_models_path=args.all_chosen_models_path
    )

if __name__ == "__main__":
    main()
