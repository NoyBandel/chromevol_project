from typing import Dict, List, Union, Any
import pandas as pd
from pathlib import Path
from constants import *
import re
import ast
import matplotlib.pyplot as plt


main_dict = {
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/dupl/run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
     LABEL_DUPL,
     LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/dupl/run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
    LABEL_DUPL,
    LABEL_LINEAR],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/gain/run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
    LABEL_GAIN,
    LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/gain/run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
    LABEL_GAIN,
    LABEL_LINEAR],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/loss/run_const_with_lin_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
    LABEL_LOSS,
    LABEL_CONSTANT],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/loss/run_lin_with_const_params/"):
    [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
    LABEL_LOSS,
    LABEL_LINEAR]
}

def raw_results_to_csv(main_dict: Dict[Path, List[Union[Path, Any]]]) -> None:
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

        df.to_csv(output_folder / f"{tested_transition_type}_all_families_run_{tested_transition_func}_{RAW_RESULTS}")


summarize_dict = {
    # dupl
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_data_for_run_lin_with_const_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_run_linear_raw_results.csv"),
        LABEL_DUPL,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_data_for_run_const_with_lin_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_run_constant_raw_results.csv"),
        LABEL_DUPL,
        LABEL_CONSTANT
    ],

    # gain
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_data_for_run_lin_with_const_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_run_linear_raw_results.csv"),
        LABEL_GAIN,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_data_for_run_const_with_lin_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_run_constant_raw_results.csv"),
        LABEL_GAIN,
        LABEL_CONSTANT
    ],

    # loss
    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_data_for_run_lin_with_const_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_run_linear_raw_results.csv"),
        LABEL_LOSS,
        LABEL_LINEAR
    ],

    Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_data_for_run_const_with_lin_params.csv"): [
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"),
        Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_run_constant_raw_results.csv"),
        LABEL_LOSS,
        LABEL_CONSTANT
    ]
}

def summarize_optimization_problem_use_cases_sheet_per_family(summarize_dict) -> None:
    for initial_linear_analysis_file, output_folder_and_optimization_test_data in summarize_dict.items():
        output_folder, optimization_test_raw_results, tested_transition_type, tested_transition_func = output_folder_and_optimization_test_data
        df_old = pd.read_csv(initial_linear_analysis_file, index_col=0)
        df_new = pd.read_csv(optimization_test_raw_results, index_col=0)

        output_excel_path = output_folder / f"{tested_transition_type}_all_families_{tested_transition_func}_summarize_optimization_problem.xlsx"

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



def summarize_optimization_problem_across_families(summarize_to_one_file_dict, family_data_with_chrom_path) -> None:
    new_main_raw_results_path = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/linear_vs_constant/")
    old_main_raw_results_path = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/const_except_for_tested/")

    family_data_with_chrom_df = pd.read_csv(family_data_with_chrom_path, index_col=0)

    for transition in LABEL_TRANSITIONS_LST:
        if transition in [LABEL_BASE_NUM, LABEL_DEMI]:
            continue

        sheet_per_family_run_const_path, sheet_per_family_run_linear_path, data_for_run_lin, output_folder = summarize_to_one_file_dict[transition]

        result_dict = {
            "family": [],
            f"new_constant_{LABEL_AICc}": [],
            f"new_linear_{LABEL_AICc}": [],
            f"old_constant_{LABEL_AICc}": [],
            f"old_linear_{LABEL_AICc}": [],
            f"old_ignore_{LABEL_AICc}": [],
            f"best_model_by_{LABEL_AICc}": [],

            f"new_constant_{LABEL_LIKELIHOOD}": [],
            f"new_linear_{LABEL_LIKELIHOOD}": [],
            f"old_constant_{LABEL_LIKELIHOOD}": [],
            f"old_linear_{LABEL_LIKELIHOOD}": [],
            f"old_ignore_{LABEL_LIKELIHOOD}": [],

            f"new_constant_num_of_events": [],
            f"new_linear_num_of_events": [],
            f"old_constant_num_of_events": [],
            f"old_linear_num_of_events": [],
            f"old_ignore_num_of_events": [],

            f"new_linear_lin_intersection": [],
            f"old_linear_lin_intersection": [],
            f"new_linear_lin_slope": [],
            f"old_linear_lin_slope": [],

            f"new_constant_val": [],
            f"old_constant_val": [],

            "family_size": [],
            "max_chrom": [],
            "min_chrom": [],
            "dif_chrom": [],
        }

        sheet_per_family_run_const_df = pd.ExcelFile(sheet_per_family_run_const_path)
        sheet_per_family_run_linear_df = pd.ExcelFile(sheet_per_family_run_linear_path)
        old_runs_data_df = pd.read_csv(data_for_run_lin)

        families_names = sheet_per_family_run_const_df.sheet_names

        for family_name in families_names:
            family_run_const_df = pd.read_excel(sheet_per_family_run_const_df, sheet_name=family_name, index_col=0)
            family_run_linear_df = pd.read_excel(sheet_per_family_run_linear_df, sheet_name=family_name, index_col=0)

            new_run_const_num_of_events_path =  new_main_raw_results_path / transition / "run_const_with_lin_params" / family_name / "Results" / "expectations_second_round.txt"
            new_run_linear_num_of_events_path =  new_main_raw_results_path / transition / "run_linear_with_lin_params" / family_name / "Results" / "expectations_second_round.txt"

            old_linear_num_of_events_path = old_main_raw_results_path / family_name / transition / LABEL_LINEAR / "Results" / "expectations_second_round.txt"
            old_ignore_num_of_events_path = old_main_raw_results_path / family_name / transition / LABEL_IGNORE / "Results" / "expectations_second_round.txt"
            old_const_num_of_events_path = old_main_raw_results_path / "allConst" / "Results" / "expectations_second_round.txt"

            # family name
            result_dict["family"].append(family_name)

            # old runs values
            result_dict[f"old_constant_{LABEL_AICc}"].append(old_runs_data_df.at[family_name, "constant_AICc"])
            result_dict[f"old_linear_{LABEL_AICc}"].append(old_runs_data_df.at[family_name, "linear_AICc"])
            result_dict[f"old_ignore_{LABEL_AICc}"].append(old_runs_data_df.at[family_name, "ignore_AICc"])

            result_dict[f"old_constant_{LABEL_LIKELIHOOD}"].append(old_runs_data_df.at[family_name, f"constant_{LABEL_LIKELIHOOD}"])
            result_dict[f"old_linear_{LABEL_LIKELIHOOD}"].append(old_runs_data_df.at[family_name, f"linear_{LABEL_LIKELIHOOD}"])
            result_dict[f"old_ignore_{LABEL_LIKELIHOOD}"].append(old_runs_data_df.at[family_name, f"ignore_{LABEL_LIKELIHOOD}"])

            result_dict["old_linear_lin_intersection"].append(old_runs_data_df.at[family_name, "lin_intersection"])
            result_dict[f"old_linear_lin_slope"].append(old_runs_data_df.at[family_name, f"lin_slope"])
            result_dict[f"old_constant_val"].append(old_runs_data_df.at[family_name, f"best_constant"])

            result_dict[f"old_constant_num_of_events"].append(extract_num_of_events(old_const_num_of_events_path, transition))
            result_dict[f"old_linear_num_of_events"].append(extract_num_of_events(old_linear_num_of_events_path, transition))
            result_dict[f"old_ignore_num_of_events"].append(extract_num_of_events(old_ignore_num_of_events_path, transition))

            # new runs values
            result_dict[f"new_constant_{LABEL_AICc}"].append(family_run_const_df.at[LABEL_AICc, "new_constant"])
            result_dict[f"new_linear_{LABEL_AICc}"].append(family_run_linear_df.at[LABEL_AICc, "new_linear"])
            result_dict[f"new_constant_{LABEL_LIKELIHOOD}"].append(family_run_const_df.at[LABEL_LIKELIHOOD, "new_constant"])
            result_dict[f"new_linear_{LABEL_LIKELIHOOD}"].append(family_run_linear_df.at[LABEL_LIKELIHOOD, "new_linear"])

            result_dict[f"new_constant_num_of_events"].append(extract_num_of_events(new_run_const_num_of_events_path, transition))
            result_dict[f"new_linear_num_of_events"].append(extract_num_of_events(new_run_linear_num_of_events_path, transition))

            result_dict[f"new_linear_lin_intersection"].append(family_run_linear_df.at["lin_intersection", "new_linear"])
            result_dict[f"new_linear_lin_slope"].append(family_run_linear_df.at["lin_slope", "new_linear"])
            result_dict[f"new_constant_val"].append(family_run_const_df.at["constant_val", "new_constant"])

            # best model, lowest AICc
            aicc_keys = [
                f"new_constant_{LABEL_AICc}",
                f"new_linear_{LABEL_AICc}",
                f"old_constant_{LABEL_AICc}",
                f"old_linear_{LABEL_AICc}",
                f"old_ignore_{LABEL_AICc}"
            ]

            aicc_values = [result_dict[k][-1] for k in aicc_keys]
            best_model = aicc_keys[aicc_values.index(min(aicc_values))]
            result_dict[f"best_model_by_{LABEL_AICc}"].append(best_model)

            # family data
            result_dict["family_size"].append(family_data_with_chrom_df.at[family_name, "family_size"])
            result_dict["max_chrom"].append(family_data_with_chrom_df.at[family_name, "max_chrom"])
            result_dict["min_chrom"].append(family_data_with_chrom_df.at[family_name, "min_chrom"])
            result_dict["diff_chrom"].append(family_data_with_chrom_df.at[family_name, "diff"])

        # Create and save dataframe
        df_summary = pd.DataFrame(result_dict)
        output_file = output_folder / f"{transition}_all_families_final_reciprocal_runs_summary.csv"
        df_summary.to_csv(output_file, index=False)

def extract_num_of_events(raw_results_expectations_path, label_transition) -> str:
    label_transition_to_expectation_format_dict = {
        LABEL_DUPL: "DUPLICATION",
        LABEL_GAIN: "GAIN",
        LABEL_LOSS: "LOSS"
    }
    with open(raw_results_expectations_path, 'r') as file:
        lines = file.readlines()
        ce_transition = label_transition_to_expectation_format_dict[label_transition]
        for line in reversed(lines):
            line = line.strip()
            if ce_transition in line:
                return line.split(": ")[-1].strip()


summarize_to_one_file_dict = {
    # transition: [
    #   sheet_per_family_run_const_path
    #   sheet_per_family_run_linear_path
    #   data_for_run_lin
    #   output_folder
    # ]

    LABEL_DUPL: [
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_constant_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_linear_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/dupl_all_families_data_for_run_lin_with_const_params.csv",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/dupl/"
    ],
    LABEL_GAIN: [
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_constant_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_linear_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analy   sis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/gain_all_families_data_for_run_lin_with_const_params.csv",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/gain/"
    ],
    LABEL_LOSS: [
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_constant_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_linear_summarize_optimization_problem.xlsx",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/loss_all_families_data_for_run_lin_with_const_params.csv",
        "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/loss/"
    ]
}
family_data_with_chrom_path = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data_with_chrom.csv"



### run
# raw_results_to_csv(main_dict)
# summarize_optimization_problem_use_cases_sheet_per_family(summarize_dict)
summarize_optimization_problem_across_families(summarize_to_one_file_dict,family_data_with_chrom_path)
