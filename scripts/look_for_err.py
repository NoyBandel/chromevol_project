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
        print(f"[v][v][v] func '{func_dir_name}'")
    else:
        print(f"[x] Errors for func '{func_dir_name}' !!! \n{err_lst}")
        output_file_path = str(raw_results_path / f"error_summary_{func_dir_name}.csv")
        df = pd.DataFrame(err_lst, columns=["Family", "Error"])
        df.to_csv(output_file_path, index=False)


def look_for_err_one_func_type(raw_results_path: Path) -> None:
    err_lst = []
    for family_folder in raw_results_path.iterdir():
        if not family_folder.is_dir():
            continue
        family_name = family_folder.name
        err_file_path = raw_results_path / family_folder / "ERR.txt"
        try:
            with open(err_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) != 0:
                    err_lst.append((family_name, f"ERR.txt not empty"))
        except:
            err_lst.append((family_name, f"job did not run"))
            continue
        res_file_path = raw_results_path / family_folder / "Results/chromEvol.res"
        if not res_file_path.exists():
            err_lst.append((family_name, f"no chromEvol.res file"))
        else:
            with open(res_file_path, 'r') as file:
                last_line = file.readlines()[-1].strip()
                if "AICc of the best model" not in last_line:
                    err_lst.append((family_name, f"chromEvol.res file not valid"))
    if len(err_lst) == 0:
        print(f"[v][v][v] ")
    else:
        print(f"[x] Errors !!! \n{err_lst}")
        output_file_path = str(raw_results_path / "error_summary.csv")
        df = pd.DataFrame(err_lst, columns=["Family", "Error"])
        df.to_csv(output_file_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Check chromEvol result directories for errors.")
    parser.add_argument("raw_results_path", type=str, help="Path to the root directory containing family folders.")
    parser.add_argument("func_dir_name", type=str, nargs="?", default=None,
                        help="(Optional) Name of function directory (only for nested structure).")

    args = parser.parse_args()
    raw_results_path = Path(args.raw_results_path)

    if args.func_dir_name is None:
        look_for_err_one_func_type(raw_results_path)
    else:
        look_for_err(raw_results_path, args.func_dir_name)

if __name__ == "__main__":
    main()


