import time
from subprocess import Popen
from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame
from create_job_files import create_job_format_for_all_jobs
from pathlib import Path
import random
from typing import Dict, Tuple
import pandas as pd
from constants import *

CHROMEVOL_EXE = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_program/chromevol/ChromEvol/chromEvol"
CONDA_ENV = "source /groups/itay_mayrose/noybandel/miniconda3/etc/profile.d/conda.sh; conda activate chromevol"
CONDA_EXPORT = "export LD_LIBRARY_PATH=/groups/itay_mayrose/noybandel/miniconda3/envs/chromevol/lib:$LD_LIBRARY_PATH\n"

OPT_FILE_RECOGNIZER = "optimization_use_case"

def create_family_output_folders(run_type_folder: str, family_name: str) -> Path:
    family_folder = Path(run_type_folder) / family_name
    results_folder = family_folder / "Results"
    results_folder.mkdir(parents=True, exist_ok=True)
    return family_folder


def get_init_parameter_dict(tested_transition: str, opt_problem_type: str) -> Tuple[Dict[str, str], str]:
    param_dict = LABEL_TRANSITIONS_LST_TO_CE_TRANSITIONS_INIT.copy()
    if opt_problem_type == "run_lin":
         param_dict["best_constant"] = param_dict[tested_transition]
    elif opt_problem_type == "run_const":
        param_dict["lin_intersection"] = param_dict[tested_transition]
    chromevol_pfile_tested_transition_init = param_dict[tested_transition]
    del param_dict[tested_transition]
    return param_dict, chromevol_pfile_tested_transition_init


def build_init_parameter_string(param_dict: Dict[str, str], family_row: Series, opt_problem_type: str, chromevol_pfile_tested_transition_init: str) -> str:
    output_str = ""
    for i, (column_name, chromevol_pfile_init) in enumerate(param_dict.items(), start=1):
        if opt_problem_type == "run_lin" and chromevol_pfile_init == chromevol_pfile_tested_transition_init:
            output_str += f"{chromevol_pfile_init} = {str(i)};{family_row[column_name]},0\n"
            continue
        output_str += f"{chromevol_pfile_init} = {str(i)};{family_row[column_name]}\n"
    print(output_str)
    return output_str


def build_param_file_content(
    dataFile_path: str,
    treeFile_path: str,
    resultsPathDir_path: str,
    opt_problem_type: str,
    tested_transition: str,
    init_parameters: str
) -> str:
    lines = [
        f"_dataFile = {dataFile_path}",
        f"_treeFile = {treeFile_path}",
        f"_resultsPathDir = {resultsPathDir_path}"
    ]

    for transition in CE_TRANSITIONS_FUNCS:
        transition_label = CE_TRANSITIONS_FUNCS_TO_LABEL_TRANSITIONS[transition]
        model = CE_LINEAR if opt_problem_type == "run_lin" and transition_label == tested_transition else CE_CONSTANT
        lines.append(f"{transition} = {model}")

    lines.append(init_parameters.strip())
    lines.extend([
        "_optimizationMethod = Brent",
        "_baseNumOptimizationMethod = Ranges",
        "_minChrNum = -1",
        "_optimizePointsNum = 1",
        "_optimizeIterNum = 7",
        "_maxParsimonyBound = true",
        "_tolParamOptimization = 0.1",
        f"_seed = {random.randint(1, 8000)}",
        "_heterogeneousModel = false",
        "_backwardPhase = false"
    ])
    return "\n".join(lines)


def write_param_file(param_file_path: str, content: str) -> None:
    with open(param_file_path, 'w') as f:
        f.write(content)
    print(f"[✓] Param file written: {param_file_path}")


def build_job_script(run_path: Path, job_name: str, param_file_path: str) -> str:
    log_file = run_path / "log.txt"
    err_file = run_path / "ERR.txt"
    cmd = f'{CONDA_ENV}\n{CONDA_EXPORT}{CHROMEVOL_EXE} "param={param_file_path}" > {log_file} 2> {err_file}\n'
    return create_job_format_for_all_jobs(str(run_path), job_name, "4G", "itaym", 1, cmd, 1)


def submit_job(run_path: Path, job_name: str, script_content: str) -> None:
    job_path = run_path / "chrom_dependence.sh"
    with open(job_path, 'w') as f:
        f.write(script_content)
    Popen(["sbatch", str(job_path)])
    print(f"[→] Submitting job: {job_name}")


def create_param_files_and_run_jobs(
    families_source_dir: Path,
    chromevol_raw_results_dir: Path,
    linear_analysis_main_folder: Path
) -> None:
    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_DEMI, LABEL_BASE_NUM, LABEL_LOSS, LABEL_GAIN]:
            continue

        print(f"\n====== Transition: {transition} ======")
        transition_folder = linear_analysis_main_folder / transition

        if not transition_folder.exists():
            print(f"[!] Skipping missing folder: {transition_folder}")
            continue

        for file in transition_folder.iterdir():
            # if file.name.startswith(f"{transition}_{OPT_FILE_RECOGNIZER}_{LABEL_CONSTANT}"):
            #     opt_problem_type = "run_lin"
            #     run_folder_name = "const_was_chosen_run_lin_with_const_params"
            # elif file.name.startswith(f"{transition}_{OPT_FILE_RECOGNIZER}_{LABEL_LINEAR}"):
            #     opt_problem_type = "run_const"
            #     run_folder_name = "lin_was_chosen_run_const_with_lin_params"
            # else:
            #     continue

            if file.name.startswith(f"{transition}_all_families_data_for_run_const"):
                opt_problem_type = "run_const"
                run_folder_name = "run_const_with_lin_params"
            # if file.name.startswith(f"{transition}_all_families_data_for_run_lin"):
            #     opt_problem_type = "run_lin"
            #     run_folder_name = "run_lin_with_const_params"
            else:
                continue

            print(f"\n ------ opt_problem_type = {opt_problem_type} ------")
            print(f"[→] Reading file: {file.name}")

            df = pd.read_csv(file, index_col=0)

            for family_name, family_row in df.iterrows():
                family_name = str(family_name)
                print(f"\n  ↳ Family: {family_name}")

                result_folder = chromevol_raw_results_dir / transition / run_folder_name
                family_results_folder = create_family_output_folders(result_folder, family_name)

                family_source = families_source_dir / family_name
                data_path = str(family_source / "counts.fasta")
                tree_path = str(family_source / "tree.nwk")
                param_path = str(family_results_folder / "paramFile")
                results_dir_path = str(family_results_folder / "Results/")

                param_dict, chromevol_pfile_tested_transition_init = get_init_parameter_dict(transition, opt_problem_type)
                init_str = build_init_parameter_string(param_dict, family_row, opt_problem_type, chromevol_pfile_tested_transition_init)
                param_content = build_param_file_content(
                    data_path, tree_path, results_dir_path, opt_problem_type, transition, init_str
                )

                write_param_file(param_path, param_content)

                job_name = f"{family_name}_{transition}_{opt_problem_type}"
                script = build_job_script(family_results_folder, job_name, param_path)
                submit_job(family_results_folder, job_name, script)

                time.sleep(3)

### run
# families_source_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_chrom_input/")
# chromevol_raw_results_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/")
# linear_analysis_main_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/")
#
# create_param_files_and_run_jobs(families_source_dir, chromevol_raw_results_dir, linear_analysis_main_folder)


families_source_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_chrom_input/")
chromevol_raw_results_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/")
linear_analysis_main_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/")

create_param_files_and_run_jobs(families_source_dir, chromevol_raw_results_dir, linear_analysis_main_folder)



