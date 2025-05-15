import argparse
import time
import ast
from subprocess import Popen

from create_job_files import create_job_format_for_all_jobs
from pathlib import Path
import random
from typing import List, Dict
import pandas as pd

CHROMEVOL_EXE = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_program/chromevol/ChromEvol/chromEvol"
CONDA_ENV = "source /groups/itay_mayrose/noybandel/miniconda3/etc/profile.d/conda.sh; conda activate chromevol"
CONDA_EXPORT = "export LD_LIBRARY_PATH=/groups/itay_mayrose/noybandel/miniconda3/envs/chromevol/lib:$LD_LIBRARY_PATH\n"

def func_to_value(func_type: str, transition_param: str, index: int) -> str:
    transition_to_param_name = {"_gainFunc": "_gain_1",
                                "_lossFunc": "_loss_1",
                                "_duplFunc": "_dupl_1",
                                "_demiDuplFunc": "_demiPloidyR_1",
                                "_baseNumRFunc": "_baseNumR_1"}
    func_to_values = {"CONST": "1.0",
                      "LINEAR": "1.0,0.1",
                      "EXP": "1.0,0.01",
                      "LOGNORMAL": "1.0,0.1,0.1",
                      "REVERSE_SIGMOID": "1.0,0.1,-0.1",
                      "LINEAR_BD": "0.1",
                      "IGNORE": ""}
    param_name = transition_to_param_name[transition_param]
    return f"{param_name} = {str(index)};{func_to_values[func_type]}"

def create_chrom_dependence_param_file(dataFile_path: str, treeFile_path: str, function_type: str, transition_param: str, param_file_path: str, resultsPathDir_path: str, transition_params: List[str], family_chosen_models_dict: dict) -> None:
    with open(param_file_path, 'w') as file:
        file.write(f"_dataFile = {dataFile_path}\n")
        file.write(f"_treeFile = {treeFile_path}\n")
        file.write(f"_resultsPathDir = {resultsPathDir_path}\n")

        for transition in transition_params:
            if transition == transition_param:
                file.write(f"{transition} = {function_type}\n")
            else:
                file.write(f"{transition} = {family_chosen_models_dict[transition]}\n")

        for i, transition in enumerate(transition_params, start=1):
            if transition == transition_param:
                file.write(f"{func_to_value(function_type, transition, i)}\n")
            else:
                file.write(f"{func_to_value(f"{family_chosen_models_dict[transition]}", transition, i)}\n")

        if (function_type == "IGNORE" and transition_param == "_baseNumRFunc") or (family_chosen_models_dict["_baseNumRFunc"] == "IGNORE" and transition_param != "_baseNumRFunc"):
            file.write("_baseNum_1 = 6;\n")
        else:
            file.write("_baseNum_1 = 6;6\n")
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


def create_and_run_single_job(run_path: Path, param_file_path: str, family_name: str, transition: str, function_type: str, transitions_abbr: dict, functions_abbr: dict) -> None:
    transition_abbr = transitions_abbr.get(transition, transition[:3])
    function_abbr = functions_abbr.get(function_type, function_type[:3])
    job_name = f'{family_name}_{transition_abbr}_{function_abbr}'

    job_path = run_path / "chrom_dependence.sh"
    param_arg = f"\"param={param_file_path}\""
    log_file = run_path / "log.txt"
    err_file = run_path / "ERR.txt"
    cmd = f'{CONDA_ENV}\n{CONDA_EXPORT}{CHROMEVOL_EXE} {param_arg} > {log_file} 2> {err_file}\n'
    memory = 4
    memory_str = f"{memory}G"
    job_content = create_job_format_for_all_jobs(str(run_path), job_name, memory_str, "itaym", 1, cmd, 1)
    with open(job_path, 'w') as file:
        file.write(job_content)
    print(f"Submitting job: {job_name}")
    Popen(["sbatch", str(job_path)])

def create_param_files_and_run_jobs(families_source_dir: Path, chromevol_raw_results_dir: Path, function_type_dir: str, transitions_dir: List[str], transitions_param: List[str], dict_transition_dir_to_param: Dict[str, str], dict_function_type_to_param: Dict[str, str], transitions_abbr: Dict[str, str], functions_abbr: Dict[str, str], all_chosen_models_path: str, transitions_csv_to_param_dict: dict) -> None:
    family_dirs: list[Path] = [folder for folder in families_source_dir.rglob('*') if folder.is_dir()]
    chosen_models_df = pd.read_csv(all_chosen_models_path, index_col=0)

    for family_source_folder in family_dirs:
        dataFile_path: str = str(family_source_folder / "counts.fasta")
        treeFile_path: str = str(family_source_folder / "tree.nwk")
        family_name: str = str(family_source_folder.name)
        row = chosen_models_df.loc[family_name]
        family_chosen_models_dict: dict = row.to_dict()
        family_chosen_models_dict = {
            transitions_csv_to_param_dict[key]: family_chosen_models_dict[key]
            for key in family_chosen_models_dict
        }
        print(f"family_chosen_models_dict: {family_chosen_models_dict}")

        for transition in transitions_dir:
            transition_param: str = dict_transition_dir_to_param.get(transition)
            family_result_folder = chromevol_raw_results_dir / family_name / transition / function_type_dir
            resultsPathDir_path = str(family_result_folder / "Results/")
            param_file_path = str(family_result_folder / "paramFile")
            function_type_param = dict_function_type_to_param.get(function_type_dir)
            create_chrom_dependence_param_file(dataFile_path, treeFile_path, function_type_param, transition_param, param_file_path, resultsPathDir_path, transitions_param, family_chosen_models_dict)
            create_and_run_single_job(family_result_folder, param_file_path, family_name, transition, function_type_dir, transitions_abbr, functions_abbr)

        time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="Generate ChromEvol parameter files and execute jobs.")
    parser.add_argument("families_source_dir", type=Path, help="Directory containing family source files.")
    parser.add_argument("chromevol_raw_results_dir", type=Path, help="Directory for storing raw ChromEvol results.")
    parser.add_argument("--function_type_dir", type=str, help="Function type directory.")
    parser.add_argument("--transitions_dir", type=str, nargs="+", required=True, help="List of transition types.")
    parser.add_argument("--transitions_param", type=str, nargs="+", required=True, help="List of transition parameters.")
    parser.add_argument("--function_types", type=str, nargs="+", required=True, help="List of function types.")
    parser.add_argument("--functions_param", type=str, nargs="+", required=True, help="List of function parameters.")
    parser.add_argument("--transitions_abbr", type=str, required=True, help="Dictionary for transition abbreviations (as a string).")
    parser.add_argument("--functions_abbr", type=str, required=True, help="Dictionary for function abbreviations (as a string).")
    parser.add_argument("--all_chosen_models_path", type=str, required=True, help="Path to the all chosen models CSV file.")
    parser.add_argument("--transitions_csv_to_param_dict", type=str, required=True, help="Dictionary mapping transition types to parameters (as a string).")

    args = parser.parse_args()

    transitions_abbr = ast.literal_eval(args.transitions_abbr)
    functions_abbr = ast.literal_eval(args.functions_abbr)
    transitions_csv_to_param_dict = ast.literal_eval(args.transitions_csv_to_param_dict)

    dict_transition_dir_to_param = dict(zip(args.transitions_dir, args.transitions_param))
    dict_function_type_to_param = dict(zip(args.function_types, args.functions_param))

    create_param_files_and_run_jobs(
        families_source_dir=args.families_source_dir,
        chromevol_raw_results_dir=args.chromevol_raw_results_dir,
        function_type_dir=args.function_type_dir,
        transitions_dir=args.transitions_dir,
        transitions_param=args.transitions_param,
        dict_transition_dir_to_param=dict_transition_dir_to_param,
        dict_function_type_to_param=dict_function_type_to_param,
        transitions_abbr=transitions_abbr,
        functions_abbr=functions_abbr,
        all_chosen_models_path=args.all_chosen_models_path,
        transitions_csv_to_param_dict=transitions_csv_to_param_dict
    )

if __name__ == "__main__":
    main()

