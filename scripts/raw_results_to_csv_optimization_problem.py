from typing import Dict, Tuple
import pandas as pd
from pathlib import Path
from constants import *
import re
import ast

main_dict = {
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/dupl/lin_was_chosen_run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
     LABEL_DUPL,
     LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/dupl/const_was_chosen_run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
    LABEL_DUPL,
    LABEL_LINEAR],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/gain/lin_was_chosen_run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
    LABEL_GAIN,
    LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/gain/const_was_chosen_run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
    LABEL_GAIN,
    LABEL_LINEAR],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/loss/lin_was_chosen_run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
    LABEL_LOSS,
    LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/optimization_problem_use_cases/loss/const_was_chosen_run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
    LABEL_LOSS,
    LABEL_LINEAR]
}

def raw_results_to_csv(main_dict: Dict[Path, Tuple]) -> None:
    for raw_results_path, output_data in main_dict.items():
        output_folder, tested_transition_type, tested_transition_func = output_data

        print(f"===== {tested_transition_type}, {tested_transition_func} =====")

        indices = [LABEL_PARAM, LABEL_AICc, LABEL_LIKELIHOOD, LABEL_CONSTS_PARAMS]
        label_transitions_lst = LABEL_TRANSITIONS_LST.copy()
        label_transitions_lst.append(LABEL_BASE_NUMR)
        transition_types = label_transitions_lst

        df = pd.DataFrame(index=indices)

        for family_folder in raw_results_path.iterdir():
            if not family_folder.is_dir():
                continue
            family_name = family_folder.name

            print(f"    Processing family: {family_name}")

            if tested_transition_func == LABEL_LINEAR:
                param_list = [None] * 2
            elif tested_transition_func == LABEL_CONSTANT:
                param_list = [None]

            consts_param_dict = {
                transition: ""
                for transition in label_transitions_lst
                if transition != tested_transition_type
            }

            results_file_path = family_folder / "Results/chromEvol.res"
            if not results_file_path.exists():
                print(f"Warning: {results_file_path} does not exist.")
                continue

            with open(results_file_path, 'r') as file:
                lines = file.readlines()
                for line in reversed(lines):
                    print(f"        {line}")
                    line = line.strip()
                    if "AICc of the best model" in line:
                        AICc = line.split("=")[-1].strip()
                        print(f"        {AICc}")
                        df.at[LABEL_AICc, family_name] = AICc
                    elif "Final optimized likelihood" in line:
                        likelihood = line.split(":")[-1].strip()
                        print(f"        {likelihood}")
                        df.at[LABEL_LIKELIHOOD, family_name] = likelihood
                        break
                    elif line.startswith("Chromosome."):
                        match = re.match(r"Chromosome\.([a-zA-Z]+)(\d*)_1\s*=\s*(.+)", line)
                        if match:
                            print(f"        {match.groups()}")
                            trans_prefix, index_str, value = match.groups()
                            if trans_prefix in transition_types:
                                if trans_prefix == tested_transition_type:
                                    index = int(index_str) if index_str else 0
                                    param_list[index] = value
                                else:
                                    consts_param_dict[trans_prefix] = value

            df.at[LABEL_PARAM, family_name] = param_list
            df.at[LABEL_CONSTS_PARAMS, family_name] = consts_param_dict

        df.to_csv(output_folder / f"{tested_transition_type}_run_{tested_transition_func}_{RAW_RESULTS}")


summarize_dict = {
    # dupl
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_optimization_use_case_constant_best_extreme_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_run_linear_raw_results.csv"),
        LABEL_DUPL,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_optimization_use_case_linear_best_minimal_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_run_constant_raw_results.csv"),
        LABEL_DUPL,
        LABEL_CONSTANT
    ],

    # gain
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_optimization_use_case_constant_best_extreme_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_run_linear_raw_results.csv"),
        LABEL_GAIN,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_optimization_use_case_linear_best_minimal_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_run_constant_raw_results.csv"),
        LABEL_GAIN,
        LABEL_CONSTANT
    ],

    # loss
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_optimization_use_case_constant_best_extreme_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_run_linear_raw_results.csv"),
        LABEL_LOSS,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_optimization_use_case_linear_best_minimal_slope.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_run_constant_raw_results.csv"),
        LABEL_LOSS,
        LABEL_CONSTANT
    ]
}

def summarize_optimization_problem_use_cases(summarize_dict) -> None:
    for initial_linear_analysis_file, output_folder_and_optimization_test_data in summarize_dict.items():
        output_folder, optimization_test_raw_results, tested_transition_type, tested_transition_func = output_folder_and_optimization_test_data
        df_old = pd.read_csv(initial_linear_analysis_file, index_col=0)
        df_new = pd.read_csv(optimization_test_raw_results, index_col=0)

        output_excel_path = output_folder / f"{tested_transition_type}_{tested_transition_func}_summarize_optimization_problem_use_cases.xlsx"

        with pd.ExcelWriter(output_excel_path) as writer:
            for family in df_new.columns:
                # Define columns depending on tested transition
                if tested_transition_func == LABEL_LINEAR:
                    columns = ["new_linear", "old_constant", "old_linear", "summarize"]
                elif tested_transition_func == LABEL_CONSTANT:
                    columns = ["new_constant", "old_linear", "old_constant", "summarize"]

                output_family_df = pd.DataFrame(
                    index=[LABEL_AICc, LABEL_LIKELIHOOD, "lin_intersection", "lin_slope", "constant_val"],
                    columns=columns
                )

                # OLD CONSTANT
                output_family_df.at[LABEL_AICc, "old_constant"] = df_old.at[family, f"{LABEL_CONSTANT}_{LABEL_AICc}"]
                output_family_df.at[LABEL_LIKELIHOOD, "old_constant"] = df_old.at[
                    family, f"{LABEL_CONSTANT}_{LABEL_LIKELIHOOD}"]
                if tested_transition_func == LABEL_LINEAR:
                    # best_constant exists only here
                    output_family_df.at["constant_val", "old_constant"] = df_old.at[family, "best_constant"]
                else:
                    # If best_constant doesn't exist, put NaN or None or skip
                    output_family_df.at["constant_val", "old_constant"] = None

                # OLD LINEAR
                output_family_df.at[LABEL_AICc, "old_linear"] = df_old.at[family, f"{LABEL_LINEAR}_{LABEL_AICc}"]
                output_family_df.at[LABEL_LIKELIHOOD, "old_linear"] = df_old.at[family, f"{LABEL_LINEAR}_{LABEL_LIKELIHOOD}"]
                output_family_df.at["lin_intersection", "old_linear"] = df_old.at[family, "lin_intersection"]
                output_family_df.at["lin_slope", "old_linear"] = df_old.at[family, "lin_slope"]

                # NEW MODEL (linear or constant)
                new_col = columns[0]
                output_family_df.at[LABEL_AICc, new_col] = df_new.at["AICc", family]
                output_family_df.at[LABEL_LIKELIHOOD, new_col] = df_new.at["likelihood", family]
                parameters = ast.literal_eval(df_new.at[LABEL_PARAM, family])
                if tested_transition_func == LABEL_LINEAR:
                    output_family_df.at["lin_intersection", new_col] = parameters[0]
                    output_family_df.at["lin_slope", new_col] = parameters[1]
                else:  # tested_transition_func == LABEL_CONSTANT
                    output_family_df.at["constant_val", new_col] = parameters[0]

                # ---- FILL summarize column ----
                # Likelihood: best (max)
                likelihood_vals = output_family_df.loc[LABEL_LIKELIHOOD, columns[:-1]]
                output_family_df.at[LABEL_LIKELIHOOD, "summarize"] = likelihood_vals.astype(float).idxmin()

                # AICc: best (min)
                aicc_vals = output_family_df.loc[LABEL_AICc, columns[:-1]]
                output_family_df.at[LABEL_AICc, "summarize"] = aicc_vals.astype(float).idxmin()

                # constant_val: abs diff
                if tested_transition_func == LABEL_CONSTANT:
                    const_val = float(output_family_df.at["constant_val", "new_constant"])
                    lin_inter = float(output_family_df.at["lin_intersection", "old_linear"])
                else:  # tested_transition_func == LABEL_LINEAR
                    const_val = float(output_family_df.at["constant_val", "old_constant"])
                    lin_inter = float(output_family_df.at["lin_intersection", "new_linear"])

                output_family_df.at["constant_val", "summarize"] = abs(const_val - lin_inter)

                # Save to sheet
                output_family_df.to_excel(writer, sheet_name=family[:31])






### run
#raw_results_to_csv(main_dict)
summarize_optimization_problem_use_cases(summarize_dict)
