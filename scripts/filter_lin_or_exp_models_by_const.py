from pathlib import Path
import ast
import pandas as pd
import argparse

def vs_const_func_param_to_csv(analysis_output_folder: str, raw_results_folder: Path, func_type: str) -> None:
    raw_results_files = list(raw_results_folder.glob("*_raw_results.csv"))
    for transition_res_file in raw_results_files:
        transition_name = str(transition_res_file.name).replace("_raw_results.csv", "")
        p0_s: dict[str, str] = {}
        p1_rates: dict[str, str] = {}
        behavior_class: dict[str, str] = {}

        df = pd.read_csv(str(transition_res_file), index_col=0)
        filtered_values = df.loc[df.index.str.endswith("_param"), func_type]
        filtered_df = pd.DataFrame(filtered_values)
        for family in filtered_df.itertuples():
            family_name = family.Index
            AICc_row = family_name.replace("param", "AICc")
            if float(df.loc[AICc_row, func_type]) > float(df.loc[AICc_row, "constant"]):
                continue
            family_name = family.Index.replace("_param", "")
            parameters_str = getattr(family, func_type)
            parameters = ast.literal_eval(parameters_str)
            p0 = str(parameters[0])
            p1_rate = str(parameters[1])
            p0_s[family_name] = p0
            p1_rates[family_name] = p1_rate
            behavior_class[family_name] = str(int(float(p1_rate) > 0))

        data = {
            "p0": p0_s,
            "p1_rate": p1_rates,
            "behavior_class": behavior_class
        }
        output_df = pd.DataFrame(data)

        output_file_path = f"{analysis_output_folder}/{transition_name}_vs_const_{func_type}_parameters.csv"
        output_df.to_csv(output_file_path, index=True)


def main():
    parser = argparse.ArgumentParser(description="Process raw transition result files.")
    parser.add_argument("csv_data_folder", type=Path, help="Directory containing CSV result files.")
    parser.add_argument("analysis_output_folder", type=Path, help="Directory of the analysis output folder.")
    parser.add_argument("func_type", type=str, help="The function type (e.g., 'linear', 'exponential').")

    args = parser.parse_args()
    vs_const_func_param_to_csv(args.analysis_output_folder, args.csv_data_folder, args.func_type)

if __name__ == "__main__":
    main()
