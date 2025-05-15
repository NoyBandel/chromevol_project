from pathlib import Path
import argparse
import pandas as pd

def look_for_err(raw_results_path: Path, func_dir_name: str) -> None:
    err_lst = []
    for family_folder in raw_results_path.iterdir():
        family_name = family_folder.name
        if func_dir_name == "allConst":
            err_file_path = family_folder / "allConst/ERR.txt"
            try:
                with open(err_file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) != 0:
                        err_lst.append((family_name, "allConst - err"))
            except:
                err_lst.append((family_name, "allConst - did not run"))
                continue
            res_file_path = family_folder / "allConst/Results/chromEvol.res"
            if not res_file_path.exists():
                err_lst.append((family_name, "allConst - no res"))
            else:
                with open(res_file_path, 'r') as file:
                    last_line = file.readlines()[-1].strip()
                    if "AICc of the best model" not in last_line:
                        err_lst.append((family_name, "allConst - no valid res"))
        else:
            for transition_folder in family_folder.iterdir():
                transition_name = transition_folder.name
                if transition_name == "allConst":
                    continue
                err_file_path = transition_folder / func_dir_name / "ERR.txt"
                try:
                    with open(err_file_path, 'r') as file:
                        lines = file.readlines()
                        if len(lines) != 0:
                            err_lst.append((family_name, f"{transition_name} - err"))
                except:
                    err_lst.append((family_name, f"{transition_name} - did not run"))
                    continue
                res_file_path = transition_folder / func_dir_name / "Results/chromEvol.res"
                if not res_file_path.exists():
                    err_lst.append((family_name, f"{transition_name} - no res"))
                else:
                    with open(res_file_path, 'r') as file:
                        last_line = file.readlines()[-1].strip()
                        if "AICc of the best model" not in last_line:
                            err_lst.append((family_name, f"{transition_name} - no valid res"))
    if len(err_lst) == 0:
        print(f"No error-files for func '{func_dir_name}'")
    else:
        print(f"Errors for func '{func_dir_name}' in: {err_lst}")

# def err_to_csv(raw_results_path: Path) -> None:



def main():
    parser = argparse.ArgumentParser(description="Check for ERR.txt files with content in a directory structure.")
    parser.add_argument("raw_results_path", type=str, help="Path to the root directory containing family folders.")
    parser.add_argument("func_dir_name", type=str, help="The name of the function directory to check.")
    args = parser.parse_args()
    raw_results_path = Path(args.raw_results_path)
    func_dir_name = args.func_dir_name
    look_for_err(raw_results_path, func_dir_name)

if __name__ == "__main__":
    main()


