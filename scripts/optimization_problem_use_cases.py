import pandas as pd
import argparse
import sys

# for each transition type:
#   from the families for which the linear function was chosen over const.
#   search for 5 families with linear slope values, a, closest to 0


# for each such family:
#   from linear - extract the intersection value, b
#   from const - extract the intersection value, c
#   compare them:
#       if b,c are very close - no optimization problem
#       else - may be optimization problem
#           rerun with const func value set to b

def optimization_problem_use_cases(transitions: list[str], linear_analysis_input_folder: str, output_folder: str,
                                   num_of_families_to_test: int, analysis_root_folder: str) -> None:
    for transition in transitions:
        print(f"-------{transition}-------")
        df = pd.read_csv(f"{linear_analysis_input_folder}{transition}_vs_const_linear_parameters.csv", index_col=0)
        closest_to_zero_df = df.loc[df['p1_rate'].abs().sort_values().index[:num_of_families_to_test]]

        closest_to_zero_df["const_AICc"] = None
        closest_to_zero_df["lin_AICc"] = None
        closest_to_zero_df["constant_value"] = None
        closest_to_zero_df[f"{transition}_relative_diff"] = None

        for family_row in closest_to_zero_df.itertuples():
            family_name = family_row.Index
            print(f"family {family_name}")
            a = family_row.p1_rate
            print(f"a = {a}")
            b = family_row.p0
            print(f"b = {b}")

            raw_results_df = pd.read_csv(f"{analysis_root_folder}{transition}_raw_results.csv", index_col=0)
            c = raw_results_df.loc[f"{family_name}_param", "constant"]
            c = float(c.strip("[]").strip("'"))
            print(f"c = {c}")
            const_AICc = raw_results_df.loc[f"{family_name}_AICc", "constant"]
            lin_AICc = raw_results_df.loc[f"{family_name}_AICc", "linear"]

            closest_to_zero_df.at[family_name, "const_AICc"] = const_AICc
            closest_to_zero_df.at[family_name, "lin_AICc"] = lin_AICc
            closest_to_zero_df.at[family_name, "constant_value"] = c
            closest_to_zero_df.at[family_name, f"{transition}_relative_diff"] = abs(c - b) / c
            print(f"relative_diff = {abs(c - b) / c}\n")

        closest_to_zero_df.to_csv(f"{output_folder}{transition}_optimization_problem_use_cases.csv")


def main():
    parser = argparse.ArgumentParser(description="Identify optimization problem use cases.")
    parser.add_argument("--transitions", type=str, required=True, nargs="+", help="List of transition types.")
    parser.add_argument("--linear_analysis_input_folder", type=str, required=True,
                        help="Path to linear vs constant parameter analysis files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save output files.")
    parser.add_argument("--num_of_families_to_test", type=int, default=5,
                        help="Number of families with slope values closest to 0 to test.")
    parser.add_argument("--analysis_root_folder", type=str, required=True, help="Path to raw results files.")
    args = parser.parse_args()

    log_file = f"{args.output_folder}optimization_log.log"

    with open(log_file, "w") as log:
        sys.stdout = log
        sys.stderr = log
        optimization_problem_use_cases(
            args.transitions,
            args.linear_analysis_input_folder,
            args.output_folder,
            args.num_of_families_to_test,
            args.analysis_root_folder
        )


if __name__ == "__main__":
    main()