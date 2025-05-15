import pandas as pd
import argparse
import ast

def create_chosen_model_file_for_each_transition(analysis_output_folder: str, transitions_res_file_name: list[str], chrom_param_mapping: dict) -> None:
    for transition_file_name in transitions_res_file_name:
        output_data = {"AICc_Value": [],
                       "Chosen_Model": [],
                       "Chrom_Param": []
                    }
        df = pd.read_csv(f"{analysis_output_folder}/{transition_file_name}", index_col=0)
        filtered_values = df.loc[df.index.str.endswith("_AICc")]
        filtered_df = pd.DataFrame(filtered_values)
        for family_name, row in filtered_df.iterrows():
            output_data["AICc_Value"].append(row.min())
            output_data["Chosen_Model"].append(row.idxmin())
            output_data["Chrom_Param"].append(chrom_param_mapping[row.idxmin()])
        output_df = pd.DataFrame.from_dict(output_data)
        output_df.index = [family_name.replace("_AICc", "") for family_name in filtered_values.index]
        transition_name = transition_file_name.replace("_raw_results.csv", "")
        output_df.to_csv(f"{analysis_output_folder}/{transition_name}_chosen_model.csv", header=True, index=True)


def create_combined_chosen_model_file(analysis_output_folder: str, transitions_chosen_file_name: list[str]) -> None:
    combined_df = None
    for transition_chosen_file_name in transitions_chosen_file_name:
        transition_df = pd.read_csv(f"{analysis_output_folder}/{transition_chosen_file_name}", index_col=0)[['Chrom_Param']]
        column_name = transition_chosen_file_name.replace("_chosen_model.csv", "")
        transition_df.rename(columns={'Chrom_Param': column_name}, inplace=True)
        if combined_df is None:
            combined_df = transition_df
        else:
            combined_df = combined_df.join(transition_df, how='inner')
    output_file_path = f"{analysis_output_folder}/all_chosen_models.csv"
    combined_df.to_csv(output_file_path, header=True, index=True, index_label="")


def add_params_column_to_chosen(analysis_folder: str, transitions: list[str]) -> None:
    for transition in transitions:
        chosen_model_file = f"{analysis_folder}/{transition}_chosen_model.csv"
        chosen_model_df = pd.read_csv(chosen_model_file, index_col=0)

        chosen_model_df["func_parameters"] = None

        raw_results_file = f"{analysis_folder}/{transition}_raw_results.csv"
        raw_results_df = pd.read_csv(raw_results_file, index_col=0)

        for family in chosen_model_df.index:
            chosen_func = chosen_model_df.loc[family, "Chosen_Model"]
            chosen_parameters = raw_results_df.loc[f"{family}_param", chosen_func]
            chosen_model_df.loc[family, "func_parameters"] = chosen_parameters

        chosen_model_df.to_csv(f"{analysis_folder}/{transition}_chosen_model_with_params.csv")


def create_chosen_model_file_for_each_transition_modified(analysis_output_folder: str, transitions_res_file_name: list[str], chromevol_param_mapping: dict, raw_results_files_folders_list: list[str]) -> None:
    for transition_file_name in transitions_res_file_name:
        output_data = {"AICc_Value": [],
                       "Chosen_Model": [],
                       "ChromEvol_Param": [],
                       "Chosen_Parameters": [],
                       "IGNORE_is_Chosen": [],
                       "Best_Model_After_IGNORE": [],
                       "CONST_Parameters": []
                       }
        output_df = pd.DataFrame.from_dict(output_data)
        for raw_results_folder in raw_results_files_folders_list:
                df = pd.read_csv(f"{raw_results_folder}/{transition_file_name}", index_col=0)
                AICc_rows = df.loc[df.index.str.endswith("_AICc")]
                param_rows = df.loc[df.index.str.endswith("_param")]
                columns_to_keep = ["linear", "exponential", "log-normal", "ignore", "constant"]
                AICc_df = AICc_rows[columns_to_keep]
                param_df = param_rows[columns_to_keep]
                output_df["Chosen_Model"] = output_df["Chosen_Model"].astype("object")
                output_df["ChromEvol_Param"] = output_df["ChromEvol_Param"].astype("object")
                output_df["Chosen_Parameters"] = output_df["Chosen_Parameters"].astype("object")

                for row_index, row in AICc_df.iterrows():
                    family_name = row_index.replace("_AICc", "")
                    output_df.loc[family_name, "AICc_Value"] = float(row.min())
                    output_df.loc[family_name, "Chosen_Model"] = str(row.idxmin())
                    output_df.loc[family_name, "ChromEvol_Param"] = str(chromevol_param_mapping[row.idxmin()])
                    chosen_model_column_name = output_df.loc[family_name, "Chosen_Model"]
                    if chosen_model_column_name == "ignore":
                        sorted_row = row.sort_values()
                        second_min_idx = sorted_row.index[1]
                        best_model_after_IGNORE = str(second_min_idx)
                        output_df.loc[family_name, "IGNORE_is_Chosen"] = '1'
                        output_df.loc[family_name, "Best_Model_After_IGNORE"] = best_model_after_IGNORE
                    else:
                        output_df.loc[family_name, "IGNORE_is_Chosen"] = '0'

                for row_index, row in param_df.iterrows():
                    family_name = row_index.replace("_param", "")
                    chosen_model_column_name = output_df.loc[family_name, "Chosen_Model"]
                    parameters = row[chosen_model_column_name]
                    if isinstance(parameters, list):
                        output_df.loc[family_name, "Chosen_Parameters"] = ', '.join(map(str, parameters))
                    else:
                        output_df.loc[family_name, "Chosen_Parameters"] = str(parameters)
                    if chosen_model_column_name == "ignore":
                        CONST_parameters = row["constant"]
                        output_df.loc[family_name, "CONST_Parameters"] = CONST_parameters

                transition_name = transition_file_name.replace("_raw_results.csv", "")
                output_df.to_csv(f"{analysis_output_folder}/{transition_name}_from_const_run_to_modified_chosen_model.csv", header=True, index=True)


def create_combined_chosen_model_file_modified(analysis_folder: str, transitions: list[str]) -> None:
    combined_df = None
    for transition in transitions:
        transition_df = pd.read_csv(f"{analysis_folder}/{transition}_from_const_run_to_modified_chosen_model.csv", index_col=0)
        if combined_df is None:
            combined_df = pd.DataFrame(index=transition_df.index)

        chosen_model_column_name = f"{transition}_chosen_model"
        parameters_column_name = f"{transition}_parameters"
        combined_df[chosen_model_column_name] = transition_df.loc[:, "ChromEvol_Param"]
        combined_df[parameters_column_name] = transition_df.loc[:, "Chosen_Parameters"]

        if transition != "baseNum" and transition != "demi":
            ignore_rows = combined_df[chosen_model_column_name] == "IGNORE"
            combined_df.loc[ignore_rows, chosen_model_column_name] = "CONST"
            combined_df.loc[ignore_rows, parameters_column_name] = transition_df.loc[ignore_rows, "CONST_Parameters"]

        if transition == "baseNum":
            combined_df["chrom_base_number"] = transition_df.loc[:, "chrom_base_number"]

    output_file_path = f"{analysis_folder}/all_chosen_models.csv"
    combined_df.to_csv(output_file_path, index=True, index_label="")


#####
def add_params_column_and_chrom_bn_to_chosen_for_baseNum(analysis_folder: str) -> None:
    chosen_model_file = f"{analysis_folder}/baseNum_chosen_model.csv"
    chosen_model_df = pd.read_csv(chosen_model_file, index_col=0)

    chosen_model_df["func_parameters"] = None
    chosen_model_df["chrom_base_number"] = None

    raw_results_file = f"{analysis_folder}/baseNum_raw_results.csv"
    raw_results_df = pd.read_csv(raw_results_file, index_col=0)

    for family in chosen_model_df.index:
        chosen_func = chosen_model_df.loc[family, "Chosen_Model"]
        chosen_parameters = raw_results_df.loc[f"{family}_param", chosen_func]
        consts_params_dict = ast.literal_eval(raw_results_df.loc[f"{family}_consts_params", chosen_func])
        chrom_base_number = consts_params_dict['baseNum']
        chosen_model_df.loc[family, "func_parameters"] = chosen_parameters
        chosen_model_df.loc[family, "chrom_base_number"] = chrom_base_number

    chosen_model_df.to_csv(f"{analysis_folder}/baseNum_chosen_model_with_params.csv")
#####


def main():
    parser = argparse.ArgumentParser(description="Run all analysis steps sequentially for chosen model files.")
    parser.add_argument("analysis_output_folder", type=str, help="Directory where the analysis files are stored.")
    parser.add_argument("--transitions_res_file_name", nargs='*', type=str, required=True,
                        help="List of transitions result file names (e.g., baseNum_raw_results.csv).")
    parser.add_argument("--chrom_param_mapping", type=str, required=True,
                        help="Dictionary for chrom_param mapping (e.g., \"{'linear': 'LINEAR', 'linear-bd': 'LINEAR_BD'}\").")
    parser.add_argument("--transitions", nargs='*', type=str, required=False,
                        help="List of transition names (e.g., baseNum, demi, dupl).")
    parser.add_argument("--raw_results_files_folders_list", nargs='*', type=str, required=False,
                        help="List of folders containing raw result files (required for 'modified' version).")
    parser.add_argument("--version", choices=['regular', 'modified'], default='regular',
                        help="Choose between 'regular' (default) or 'modified' version of model file creation.")

    args = parser.parse_args()
    chrom_param_mapping = ast.literal_eval(args.chrom_param_mapping)

    if args.version == 'modified':
        create_chosen_model_file_for_each_transition_modified(
            args.analysis_output_folder,
            args.transitions_res_file_name,
            chrom_param_mapping,
            args.raw_results_files_folders_list
        )
        create_combined_chosen_model_file_modified(args.analysis_output_folder, args.transitions)
    else:
        create_chosen_model_file_for_each_transition(
            args.analysis_output_folder,
            args.transitions_res_file_name,
            chrom_param_mapping
        )
        transitions_chosen_file_name = [f"{transition}_chosen_model.csv" for transition in args.transitions]
        create_combined_chosen_model_file(args.analysis_output_folder, transitions_chosen_file_name)
        add_params_column_to_chosen(args.analysis_output_folder, args.transitions)

if __name__ == "__main__":
    main()


