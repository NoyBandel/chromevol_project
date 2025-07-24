import ast
import pandas as pd
import argparse
from pathlib import Path

def linear_param_to_csv(csv_data_folder: str, analysis_output_folder: str, transitions_res_file_name: list[str]) -> None:
    for transition_res_file_name in transitions_res_file_name:
        p0_intersections: dict[str:str] = {}
        p1_slopes: dict[str:str] = {}
        behavior_class: dict[str:str] = {}
        
        transition_res_path: str = f"{csv_data_folder}/{transition_res_file_name}"
        df = pd.read_csv(transition_res_path, index_col=0)
        filtered_values = df.loc[df.index.str.endswith("_param"), "linear"]
        filtered_df = pd.DataFrame(filtered_values)
        for family in filtered_df.itertuples():
            family_name = str(str(family.Index).replace("_param", ""))
            linear_parameters_str = family.linear
            linear_parameters = ast.literal_eval(linear_parameters_str)
            p0_intersection = str(linear_parameters[0])
            p1_slope = str(linear_parameters[1])
            p0_intersections[family_name] = p0_intersection
            p1_slopes[family_name] = p1_slope
            behavior_class[family_name] = str(int(float(p1_slope) > 0))
        
       
        ascending_percentage = (list(behavior_class.values()).count("1") / len(behavior_class)) * 100
        descending_percentage = 100 - ascending_percentage
        behavior_distribution = {
            "ascending_percentage": ascending_percentage,
            "descending_percentage": descending_percentage
        }
        behavior_distribution_df = pd.DataFrame([behavior_distribution])
        behavior_distribution_df.to_csv(f"{analysis_output_folder}/{transition_res_file_name.replace('_raw_results.csv', '')}_behavior_distribution.csv", index=False)
        
        data = {
        "p0_intersection": p0_intersections,
        "p1_slope": p1_slopes,
        "behavior_class": behavior_class
        }
        output_df = pd.DataFrame(data)
        output_df.to_csv(f"{analysis_output_folder}/{transition_res_file_name.replace("_raw_results.csv", "")}_linear_parameters.csv", index=True)
        
    
# def main():
#     parser = argparse.ArgumentParser(description="?.")
#     parser.add_argument("csv_data_folder", type=str, help="Directory containing csv result file.")
#     parser.add_argument("analysis_output_folder", type=str, help="Directory of the analysis folder.")
#     parser.add_argument("transitions_res_file_name", nargs='+', type=str, help="List of transitions result file names.")
#
#     args = parser.parse_args()
#     linear_param_to_csv(args.csv_data_folder, args.analysis_output_folder, args.transitions_res_file_name)
#
# if __name__ == "__main__":
#     main()

    
# transitions_res_file_name = ["baseNum_raw_results.csv", "demi_raw_results.csv", "dupl_raw_results.csv", "gain_raw_results.csv", "loss_raw_results.csv"]

from constants import *
import os

def linear_analysis_file(main_output_folder: str, raw_results_csvs_folder: str) -> None:
    """
    Create linear model analysis csv files.

    For each transition type:
    - Loads the raw results CSV
    - Extracts linear model parameters (slope, intercept)
    - Computes the behavior class (ascending or descending slope)
    - Extracts AICc scores for model comparison (linear, constant, ignore)
    - Saves:
        - Full linear model analysis
        - A subset where linear model AICc is best (lowest)
        - A subset where ignore model is not the best (i.e., is worse than at least one alternative)

    Args:
        main_output_folder (str): The main output directory, containing folder for each transition, where result files will be saved.
        raw_results_csvs_folder (str): The folder containing raw CSV files with all model results.
    """
    linear_analysis_columns = [LABEL_LINEAR, LABEL_CONSTANT, LABEL_IGNORE]
    for transition in LABEL_TRANSITIONS_LST:
        raw_results_path = os.path.join(
            raw_results_csvs_folder, f"{transition}_all_families_over_50_modified_{RAW_RESULTS}"
        )
        df = pd.read_csv(raw_results_path, index_col=0)
        df = df[[col for col in linear_analysis_columns if col in df.columns]]

        params_df = df.loc[df.index.str.endswith(LABEL_PARAMS_SHORT), [LABEL_LINEAR]]
        AICc_df = df.loc[df.index.str.endswith(LABEL_AICc)]

        # Extract slope, intercept, and behavior class
        p0_intersections, p1_slopes, behavior_class = {}, {}, {}
        for family in params_df.itertuples():
            family_name = str(family.Index).replace(f"_{LABEL_PARAMS_SHORT}", "")
            linear_parameters = ast.literal_eval(family.linear)
            p0, p1 = str(linear_parameters[0]), str(linear_parameters[1])
            p0_intersections[family_name] = p0
            p1_slopes[family_name] = p1
            behavior_class[family_name] = str(int(float(p1) > 0))

        # Print behavior distribution
        total = len(behavior_class)
        ascending = sum(1 for val in behavior_class.values() if val == "1")
        ascending_pct = ascending / total * 100
        descending_pct = 100 - ascending_pct
        print(f"{transition} behavior_distribution")
        print({
            "ascending_percentage": ascending_pct,
            "descending_percentage": descending_pct
        })

        # Extract AICc values
        linear_AICc, constant_AICc, ignore_AICc = {}, {}, {}
        for family, row in AICc_df.iterrows():
            family_name = str(family).replace(f"_{LABEL_AICc}", "")
            linear_AICc[family_name] = row[LABEL_LINEAR]
            constant_AICc[family_name] = row[LABEL_CONSTANT]
            ignore_AICc[family_name] = row[LABEL_IGNORE]

        data = {
            "lin_intersection": p0_intersections,
            "lin_slope": p1_slopes,
            "lin_behavior_class": behavior_class,
            "linear_AICc": linear_AICc,
            "constant_AICc": constant_AICc,
            "ignore_AICc": ignore_AICc,
        }

        transition_folder = os.path.join(main_output_folder, transition)
        os.makedirs(transition_folder, exist_ok=True)

        # Save full analysis
        output_df = pd.DataFrame(data)
        full_analysis_path = os.path.join(transition_folder, f"{transition}_linear_analysis.csv")
        output_df.to_csv(full_analysis_path)

        # Save families where linear AICc is best
        best_lin_df = output_df[
            (output_df["linear_AICc"].astype(float) < output_df["constant_AICc"].astype(float)) &
            (output_df["linear_AICc"].astype(float) < output_df["ignore_AICc"].astype(float))
        ]
        best_lin_path = os.path.join(transition_folder, f"{transition}_best_linear_AICc.csv")
        best_lin_df.to_csv(best_lin_path)

        # Save families where ignore model is NOT the best
        not_ignore_best_df = output_df[
            (output_df["ignore_AICc"].astype(float) > output_df["linear_AICc"].astype(float)) |
            (output_df["ignore_AICc"].astype(float) > output_df["constant_AICc"].astype(float))
        ]
        not_ignore_best_path = os.path.join(transition_folder, f"{transition}_ignore_not_best.csv")
        not_ignore_best_df.to_csv(not_ignore_best_path)

#
# main_output_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/"
# raw_results_csvs_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/raw_results/modified_raw_results/"

# linear_analysis_file(main_output_folder, raw_results_csvs_folder)


def prepare_reciprocal_runs_data_for_graphs(main_linear_analysis_folder: Path) -> None:
    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_BASE_NUM, LABEL_DEMI]:
            continue

        transition_folder = main_linear_analysis_folder / transition
        reciprocal_runs_summary_path = transition_folder / f"{transition}_all_families_final_reciprocal_runs_summary.csv"
        reciprocal_runs_summary_df = pd.read_csv(reciprocal_runs_summary_path, index_col=0)

        # Start building new DataFrame with the same index
        new_df = pd.DataFrame(index=reciprocal_runs_summary_df.index)

        # ---- Constant model ----
        df_const = reciprocal_runs_summary_df[["new_constant_AICc", "old_constant_AICc"]]
        new_df["constant_AICc"] = df_const.min(axis=1)
        new_df["constant_new_or_old"] = df_const.apply(
            lambda row: "new" if row["new_constant_AICc"] < row["old_constant_AICc"] else "old", axis=1
        )

        # ---- Ignore model ----
        new_df["ignore_AICc"] = reciprocal_runs_summary_df["old_ignore_AICc"]

        # ---- Linear model ----
        df_linear = reciprocal_runs_summary_df[["new_linear_AICc", "old_linear_AICc"]]
        new_df["linear_AICc"] = df_linear.min(axis=1)
        new_df["linear_new_or_old"] = df_linear.apply(
            lambda row: "new" if row["new_linear_AICc"] < row["old_linear_AICc"] else "old", axis=1
        )

        # Assign linear model parameters based on which AICc was chosen
        def extract_linear_params(row):
            if row["linear_new_or_old"] == "old":
                slope = reciprocal_runs_summary_df.at[row.name, "old_linear_lin_slope"]
                intercept = reciprocal_runs_summary_df.at[row.name, "old_linear_lin_intersection"]
            else:
                slope = reciprocal_runs_summary_df.at[row.name, "new_linear_lin_slope"]
                intercept = reciprocal_runs_summary_df.at[row.name, "new_linear_lin_intersection"]
            return pd.Series({
                "lin_slope": slope,
                "lin_intersection": intercept,
                "lin_behavior_class": str(int(float(slope) > 0))
            })

        new_df[["lin_slope", "lin_intersection", "lin_behavior_class"]] = new_df.apply(extract_linear_params, axis=1)

        # ---- Save full analysis ----
        full_analysis_path = transition_folder / f"{transition}_reciprocal_runs_linear_analysis.csv"
        new_df.to_csv(full_analysis_path)

        # ---- Save families where linear AICc is best ----
        best_lin_df = new_df[
            (new_df["linear_AICc"].astype(float) < new_df["constant_AICc"].astype(float)) &
            (new_df["linear_AICc"].astype(float) < new_df["ignore_AICc"].astype(float))
        ]
        best_lin_path = transition_folder / f"{transition}_reciprocal_runs_best_linear_AICc.csv"
        best_lin_df.to_csv(best_lin_path)

        # ---- Save families where ignore model is NOT the best ----
        not_ignore_best_df = new_df[
            (new_df["ignore_AICc"].astype(float) > new_df["linear_AICc"].astype(float)) |
            (new_df["ignore_AICc"].astype(float) > new_df["constant_AICc"].astype(float))
        ]
        not_ignore_best_path = transition_folder / f"{transition}_reciprocal_runs_ignore_not_best.csv"
        not_ignore_best_df.to_csv(not_ignore_best_path)

main_linear_analysis_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/")
prepare_reciprocal_runs_data_for_graphs(main_linear_analysis_folder)