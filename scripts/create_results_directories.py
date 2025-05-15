import argparse
from pathlib import Path
import shutil


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

def main():
    parser = argparse.ArgumentParser(description="Create result directories for ChromEvol project.")
    parser.add_argument('families_source_dir', type=Path, help="Source directory for family data.")
    parser.add_argument('chromevol_raw_results_dir', type=Path, help="Directory to store the results.")
    parser.add_argument('--transitions', type=str, nargs='+', help="List of transition types (e.g., gain loss dupl).")
    parser.add_argument('--functions', type=str, nargs='+', help="List of function types (e.g., linear exponential).")

    args = parser.parse_args()

    create_results_directories(args.families_source_dir, args.chromevol_raw_results_dir, args.transitions, args.functions)


if __name__ == "__main__":
    main()




