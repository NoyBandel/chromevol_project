import argparse
from pathlib import Path
from constants import *

def create_results_directories(families_source_dir: Path, chromevol_raw_results_dir: Path, transitions: list,
                               functions: list) -> None:
    for folder in families_source_dir.rglob('*'):
        if folder.is_dir():
            family_name = folder.name
            family_folder_path = chromevol_raw_results_dir / family_name
            family_folder_path.mkdir(exist_ok=True)

            for transition in transitions:
                transition_path = family_folder_path / transition
                transition_path.mkdir(exist_ok=True)

                if transition == "allConst":
                    results_path = transition_path / "Results"
                    results_path.mkdir(exist_ok=True)
                else:
                    for function in functions:
                        function_path = transition_path / function
                        function_path.mkdir(exist_ok=True)
                        results_path = function_path / "Results"
                        results_path.mkdir(exist_ok=True)

#####
def add_ignore_directories(chromevol_raw_results_dir: Path) -> None:
    for family_folder in chromevol_raw_results_dir.rglob('*'):
        if family_folder.is_dir():
            for transition_folder in family_folder.iterdir():
                if transition_folder.is_dir():
                    transition_name = transition_folder.name
                    if transition_name == "allConst":
                        continue
                    function_path = transition_folder / "ignore"
                    function_path.mkdir(exist_ok=True)
                    results_path = function_path / "results"
                    results_path.mkdir(exist_ok=True)
#####


def create_results_directories_linear_vs_constant(families_source_dir: Path, chromevol_linear_vs_constant_raw_results_dir: Path, run_types: list) -> None:
    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_DEMI, LABEL_BASE_NUM]:
            continue
        print(f" ======== transition: {transition} ======== ")
        for opt_problem_type in run_types:
            print(f" ==== opt_problem_type: {opt_problem_type} ==== ")
            for folder in families_source_dir.rglob('*'):
                if folder.is_dir() and not folder.name.startswith('.'):
                    family_name = folder.name
                    print(f"family_name: {family_name}")
                    family_folder_path = chromevol_linear_vs_constant_raw_results_dir / transition / opt_problem_type / family_name
                    family_folder_path.mkdir(exist_ok=True)
                    results_path = family_folder_path / "Results"
                    results_path.mkdir(exist_ok=True)


#
# def main():
#     parser = argparse.ArgumentParser(description="Create result directories for ChromEvol project.")
#     parser.add_argument('families_source_dir', type=Path, help="Source directory for family data.")
#     parser.add_argument('chromevol_raw_results_dir', type=Path, help="Directory to store the results.")
#     parser.add_argument('--transitions', type=str, nargs='+', help="List of transition types (e.g., gain loss dupl).")
#     parser.add_argument('--functions', type=str, nargs='+', help="List of function types (e.g., linear exponential).")
#
#     args = parser.parse_args()
#
#     create_results_directories(args.families_source_dir, args.chromevol_raw_results_dir, args.transitions, args.functions)
#
#
# if __name__ == "__main__":
#     main()

# families_source_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/all_families_over_50/")
# chromevol_linear_vs_constant_raw_results_dir = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/")
# run_types = ["run_lin_with_const_params", "run_const_with_lin_params"]
#
# create_results_directories_linear_vs_constant(families_source_dir, chromevol_linear_vs_constant_raw_results_dir, run_types)
#
#
#
#

