import pandas as pd
import ast
import os
from constants import *

LIN_FILE_SUFFIX = "_linear_analysis.csv"
RAW_RESULTS_SUFFIX = "_all_families_over_50_modified_raw_results.csv"


def enrich_with_likelihoods(df, families, raw_results_df, label_prefixes):
    for fam in families:
        row = f"{fam}_{LABEL_LIKELIHOOD}"
        if row in raw_results_df.index:
            for label in label_prefixes:
                df.loc[fam, f"{label}_{LABEL_LIKELIHOOD}"] = raw_results_df.loc[row, label]
        else:
            print(f"Warning: {row} not found in raw results")
    return df


def enrich_with_params(df, families, raw_results_df, param_keys, model_column):
    for key in param_keys:
        df[key] = None

    for fam in families:
        param_row = f"{fam}_{LABEL_CONSTS_PARAMS}"
        if param_row in raw_results_df.index:
            param_str = raw_results_df.loc[param_row, model_column]
            try:
                param_dict = ast.literal_eval(param_str)
                if isinstance(param_dict, dict):
                    for key in param_keys:
                        df.loc[fam, key] = param_dict.get(key)
                else:
                    print(f"Parsed params for {fam} are not a dictionary.")
            except Exception as e:
                print(f"Failed to parse params for {fam}: {e}")
        else:
            print(f"{param_row} not found in raw results")
    return df


def extract_best_constant_value(df, families, raw_results_df):
    df['best_constant'] = None
    for fam in families:
        param_row = f"{fam}_{LABEL_PARAM}"
        if param_row in raw_results_df.index:
            try:
                raw_val = raw_results_df.loc[param_row, LABEL_CONSTANT]
                val = float(ast.literal_eval(raw_val)[0])
                df.loc[fam, 'best_constant'] = val
            except Exception as e:
                print(f"Failed to extract best constant value for {fam}: {e}")
        else:
            print(f"{param_row} not found in raw results")
    return df


def optimization_problem_use_cases(linear_analysis_folder: str,
                                   num_of_families_to_test: int,
                                   raw_results_folder: str) -> None:

    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_DEMI, LABEL_BASE_NUM]:
            continue

        print(f"------- {transition} -------")

        lin_csv_path = os.path.join(linear_analysis_folder, transition, f"{transition}{LIN_FILE_SUFFIX}")
        raw_results_path = os.path.join(raw_results_folder, f"{transition}{RAW_RESULTS_SUFFIX}")

        df = pd.read_csv(lin_csv_path, index_col=0)
        raw_results_df = pd.read_csv(raw_results_path, index_col=0)

        # Identify best model
        df['best_model'] = df[[f"{LABEL_IGNORE}_{LABEL_AICc}",
                               f"{LABEL_LINEAR}_{LABEL_AICc}",
                               f"{LABEL_CONSTANT}_{LABEL_AICc}"]].idxmin(axis=1)

        param_keys = [label for label in LABEL_TRANSITIONS_LST if label != transition]
        param_keys.append(LABEL_BASE_NUMR)

        # --- LINEAR best but |slope| ≈ 0 ---
        linear_best_df = df[df['best_model'] == f"{LABEL_LINEAR}_{LABEL_AICc}"]
        linear_best_minimal_slope = linear_best_df.reindex(
            linear_best_df['lin_slope'].abs().sort_values().head(num_of_families_to_test).index
        ).copy()

        linear_families = list(linear_best_minimal_slope.index)

        linear_best_minimal_slope = enrich_with_likelihoods(
            linear_best_minimal_slope,
            linear_families,
            raw_results_df,
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )

        linear_best_minimal_slope = enrich_with_params(
            linear_best_minimal_slope,
            linear_families,
            raw_results_df,
            param_keys,
            LABEL_LINEAR
        )

        output_path = os.path.join(
            linear_analysis_folder, transition,
            f"{transition}_optimization_use_case_linear_best_minimal_slope.csv"
        )
        linear_best_minimal_slope.to_csv(output_path)
        print(f"Saved: {output_path}")

        # --- CONSTANT best but |slope| ≫ 0 ---
        constant_best_df = df[df['best_model'] == f"{LABEL_CONSTANT}_{LABEL_AICc}"]
        constant_best_extreme_slope = constant_best_df.reindex(
            constant_best_df['lin_slope'].abs().sort_values(ascending=False).head(num_of_families_to_test).index
        ).copy()

        constant_families = list(constant_best_extreme_slope.index)

        constant_best_extreme_slope = enrich_with_likelihoods(
            constant_best_extreme_slope,
            constant_families,
            raw_results_df,
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )

        constant_best_extreme_slope = enrich_with_params(
            constant_best_extreme_slope,
            constant_families,
            raw_results_df,
            param_keys,
            LABEL_CONSTANT
        )

        constant_best_extreme_slope = extract_best_constant_value(
            constant_best_extreme_slope,
            constant_families,
            raw_results_df
        )

        output_path = os.path.join(
            linear_analysis_folder, transition,
            f"{transition}_optimization_use_case_constant_best_extreme_slope.csv"
        )
        constant_best_extreme_slope.to_csv(output_path)
        print(f"Saved: {output_path}")


def prepare_data_for_reciprocal_runs(linear_analysis_folder: str,
                                   raw_results_folder: str) -> None:

    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_DEMI, LABEL_BASE_NUM]:
            continue

        print(f"------- {transition} -------")

        lin_csv_path = os.path.join(linear_analysis_folder, transition, f"{transition}{LIN_FILE_SUFFIX}")
        raw_results_path = os.path.join(raw_results_folder, f"{transition}{RAW_RESULTS_SUFFIX}")

        df = pd.read_csv(lin_csv_path, index_col=0)
        raw_results_df = pd.read_csv(raw_results_path, index_col=0)

        param_keys = [label for label in LABEL_TRANSITIONS_LST if label != transition]
        param_keys.append(LABEL_BASE_NUMR)

        # --- prepare for run CONST with LINEAR parameters ---
        linear_df = df.copy()
        linear_families = list(linear_df.index)

        linear_df = enrich_with_likelihoods(
            linear_df,
            linear_families,
            raw_results_df,
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )

        linear_df = enrich_with_params(
            linear_df,
            linear_families,
            raw_results_df,
            param_keys,
            LABEL_LINEAR
        )

        output_path = os.path.join(
            linear_analysis_folder, transition,
            f"{transition}_all_families_data_for_run_const_with_lin_params.csv"
        )
        linear_df.to_csv(output_path)
        print(f"Saved: {output_path}")

        # --- prepare for run LINEAR with CONST parameters ---
        constant_df = df.copy()
        constant_families = list(constant_df.index)

        constant_df = enrich_with_likelihoods(
            constant_df,
            constant_families,
            raw_results_df,
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )

        constant_df = enrich_with_params(
            constant_df,
            constant_families,
            raw_results_df,
            param_keys,
            LABEL_CONSTANT
        )

        constant_df = extract_best_constant_value(
            constant_df,
            constant_families,
            raw_results_df
        )

        output_path = os.path.join(
            linear_analysis_folder, transition,
            f"{transition}_all_families_data_for_run_lin_with_const_params.csv"
        )
        constant_df.to_csv(output_path)
        print(f"Saved: {output_path}")



### run
# linear_analysis_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/"
# num_of_families_to_test = 5
# raw_results_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/raw_results/modified_raw_results/"
# optimization_problem_use_cases(linear_analysis_folder, num_of_families_to_test, raw_results_folder)

#
# linear_analysis_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/"
# raw_results_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/raw_results/modified_raw_results/"
# prepare_data_for_reciprocal_runs(linear_analysis_folder, raw_results_folder)