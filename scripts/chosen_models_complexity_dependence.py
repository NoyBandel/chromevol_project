import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import stats

def chosen_models_complexity_dependence_on_family_size_csv(transitions: list[str], families_data_file: str, min_family_size: int, families_folders: list[str], output_folder: str) -> None:
    family_data_df = pd.read_csv(families_data_file)
    filtered_df = family_data_df[family_data_df['family_size'] > min_family_size-1][['family_name', 'family_size']]
    filtered_df["chosen_model"] = None
    filtered_df["num_of_params"] = None

    combined_df = pd.DataFrame(columns=['family_name', 'family_size', 'chosen_model', 'num_of_params', 'transition'])

    for transition in transitions:
        transition_df = filtered_df.copy()
        for families_folder in families_folders:
            curr_families_chosen_model = pd.read_csv(f"{families_folder}{transition}_chosen_model_with_params.csv", index_col=0)
            for family_row in curr_families_chosen_model.itertuples():
                family_name = family_row.Index
                if family_name not in transition_df['family_name'].values:
                    continue
                chosen_model = family_row.Chrom_Param
                func_parameters = family_row.func_parameters.strip("[]")
                if func_parameters == "":
                    num_of_params = 0
                else:
                    num_of_params = len(func_parameters.split(","))
                print(f"Processing family: {family_name}, chosen model: {chosen_model}, num of params: {num_of_params}")
                transition_df.loc[transition_df['family_name'] == family_name, "chosen_model"] = chosen_model
                transition_df.loc[transition_df['family_name'] == family_name, "num_of_params"] = num_of_params
                new_row = pd.DataFrame({
                    'family_name': [family_name],
                    'family_size': [filtered_df[filtered_df['family_name'] == family_name]['family_size'].values[0]],
                    'chosen_model': [chosen_model],
                    'num_of_params': [num_of_params],
                    'transition': [transition]
                })
                new_row['family_name'] = new_row['family_name'] + "_" + new_row['transition']
                combined_df = pd.concat([combined_df, new_row], ignore_index=True)

        transition_df.to_csv(f"{output_folder}{transition}_family_size_complexity_dependence.csv", index=False)
    combined_df.to_csv(f"{output_folder}all_transitions_family_size_complexity_dependence.csv", index=False)


def chosen_models_complexity_dependence_on_family_size_plot(dependence_folder_path: str, output_folder: str, prefix_list: list[str]) -> None:
    for prefix in prefix_list:
        dependence_df = pd.read_csv(f"{dependence_folder_path}{prefix}_family_size_complexity_dependence.csv")
        family_size = dependence_df['family_size']
        num_of_params = dependence_df['num_of_params']
        plt.figure(figsize=(10, 6))
        plt.scatter(family_size, num_of_params, alpha=0.7)
        plt.title(f"Dependence of Number of Parameters on Family Size ({prefix})")
        plt.xlabel("Family Size (Number of Species)")
        plt.ylabel("Number of Parameters (Model Complexity)")
        plt.grid(True)
        y_min, y_max = plt.ylim()
        plt.yticks(range(int(y_min), int(y_max) + 1))
        plt.savefig(f"{output_folder}{prefix}_family_size_complexity_dependence_plot.png")
        plt.close()


def anova_test(dependence_folder_path: str, prefix_list: list[str], output_folder: str, suffix: str, tested_variable: str) -> None:
    log = f"ANOVA test {tested_variable}\n"
    for prefix in prefix_list:
        log += f"----{prefix}----\n"
        dependence_df = pd.read_csv(f"{dependence_folder_path}{prefix}_{suffix}")
        grouped_by_num_of_params = dependence_df.groupby('num_of_params')[tested_variable]
        for num_of_params, group in grouped_by_num_of_params:
            log += f"Group for num_of_params = {num_of_params}:\n"
            log += f"{', '.join(map(str, group))}\n"
        groups_list = [group.tolist() for num_of_params, group in grouped_by_num_of_params]
        f_stat, p_value = stats.f_oneway(*groups_list)
        log += f"F-statistic: {f_stat}\n"
        log += f"P-value: {p_value}\n"
        if p_value < 0.05:
            log += "Significant difference between the groups.\n"
        else:
            log += "No significant difference between the groups.\n"
    with open(f"{output_folder}/anova_test_{tested_variable}.txt", "w") as file:
        file.write(log)

def chosen_models_complexity_dependence_on_transition_events_count_csv(transitions: list[str], raw_results_folders_list: list[Path], output_folder: str, input_folder: str) -> None:
    transition_name_to_df_dict = {
        transition: pd.read_csv(f"{input_folder}/{transition}_family_size_complexity_dependence.csv")
        for transition in transitions
    }
    for df in transition_name_to_df_dict.values():
        df["num_of_events"] = None

    for raw_results_folder_path in raw_results_folders_list:
        for family_folder in raw_results_folder_path.iterdir():
            if not family_folder.is_dir():
                continue
            family_name = family_folder.name
            expectations_file_path = family_folder / "all_Const/Results/expectations_second_round.txt"

            if expectations_file_path.exists():
                with expectations_file_path.open("r") as file:
                    lines = file.readlines()
                    for line in reversed(lines):
                        line = line.strip()
                        if "BASE-NUMBER" in line:
                            num_events = line.split(":")[-1].strip()
                            transition_name_to_df_dict["baseNum"].loc[transition_name_to_df_dict["baseNum"]["family_name"] == family_name,"num_of_events"] = num_events
                        elif "DEMI-DUPLICATION" in line:
                            num_events = line.split(":")[-1].strip()
                            transition_name_to_df_dict["demi"].loc[transition_name_to_df_dict["demi"]["family_name"] == family_name, "num_of_events"] = num_events
                        elif "DUPLICATION" in line:
                            num_events = line.split(":")[-1].strip()
                            transition_name_to_df_dict["dup"].loc[transition_name_to_df_dict["dup"]["family_name"] == family_name, "num_of_events"] = num_events
                        elif "LOSS" in line:
                            num_events = line.split(":")[-1].strip()
                            transition_name_to_df_dict["loss"].loc[transition_name_to_df_dict["loss"]["family_name"] == family_name, "num_of_events"] = num_events
                        elif "GAIN" in line:
                            num_events = line.split(":")[-1].strip()
                            transition_name_to_df_dict["gain"].loc[transition_name_to_df_dict["gain"]["family_name"] == family_name, "num_of_events"] = num_events

    combined_df = pd.DataFrame(columns=['family_name', 'family_size', 'chosen_model', 'num_of_params', 'transition', 'num_of_events'])
    for transition_name, transition_df in transition_name_to_df_dict.items():
        output_file = f"{output_folder}/{transition_name}_num_of_events_complexity_dependence.csv"
        transition_df.to_csv(output_file, index=False)
        combined_df = pd.concat([combined_df, transition_df], ignore_index=True)
    combined_df.to_csv(f"{output_folder}/all_transitions_num_of_events_complexity_dependence.csv", index=False)
#
#
# input_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/chosen_models_complexity_dependence/family_size/"
# output_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/chosen_models_complexity_dependence/num_of_events/"
# raw_results_folders_list = [Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results_50_to_99/chromevol_raw_results_const_except_for_tested_50_to_99/"), Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_raw_results/const_except_for_tested/") ]
# transitions = [baseNum demi dupl gain loss]


# transition_capital_to_name = {"GAIN": "gain",
#                               "LOSS": "loss",
#                               "DEMI-DUPLICATION": "demi",
#                               "DUPLICATION": "dupl",
#                               "BASE-NUMBER": "baseNum"
#                             }
#
#
#
#
# def chosen_models_complexity_dependence_on_transition_events_count_plot(dependence_folder_path: str, output_folder: str, prefix_list: list[str]) -> None:
#     for prefix in prefix_list:
#         dependence_df = pd.read_csv(f"{dependence_folder_path}{prefix}_family_size_complexity_dependence.csv")
#         family_size = dependence_df['family_size']
#         num_of_params = dependence_df['num_of_params']
#         plt.figure(figsize=(10, 6))
#         plt.scatter(family_size, num_of_params, alpha=0.7)
#         plt.title(f"Dependence of Number of Parameters on Family Size ({prefix})")
#         plt.xlabel("Family Size (Number of Species)")
#         plt.ylabel("Number of Parameters (Model Complexity)")
#         plt.grid(True)
#         y_min, y_max = plt.ylim()
#         plt.yticks(range(int(y_min), int(y_max) + 1))
#         plt.savefig(f"{output_folder}{prefix}_family_size_complexity_dependence_plot.png")
#         plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot family size vs. model complexity dependence.")
    parser.add_argument('function',
                        choices=['csv_family_size', 'plot_family_size', 'anova_test', 'csv_transition_events'],
                        help="Function to execute: 'csv_family_size' (generate CSVs), 'plot_family_size' (generate plots), 'anova_test' (run ANOVA tests), 'csv_transition_events' (generate transition events CSVs).")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Folder to save the output results.")
    parser.add_argument('--input_folder', type=str, required=False,
                        help="Folder to load the input data (if needed).")

    # Arguments for CSV generation based on family size
    parser.add_argument('--transitions', type=str, nargs='+',
                        help="List of transitions.")
    parser.add_argument('--families_data_file', type=str,
                        help="CSV file with family data.")
    parser.add_argument('--min_family_size', type=int,
                        help="Minimum family size to include.")
    parser.add_argument('--families_folders', type=str, nargs='+',
                        help="Folders containing family-specific data.")

    # Arguments for plotting
    parser.add_argument('--dependence_folder_path', type=str,
                        help="Folder containing dependence CSV files for plotting.")
    parser.add_argument('--prefix_list', type=str, nargs='+',
                        help="List of prefixes for file output.")

    # Arguments for ANOVA test
    parser.add_argument('--tested_variable', type=str,
                        help="Variable to test in the ANOVA analysis.")
    parser.add_argument('--suffix', type=str,
                        help="Suffix of dependence files for ANOVA.")

    # Arguments for transition events CSV generation
    parser.add_argument('--raw_results_folder_path', type=str,
                        help="Folder path to raw results for transition events.")

    args = parser.parse_args()

    if args.function == 'csv_family_size':
        if not all([args.transitions, args.families_data_file, args.min_family_size, args.families_folders]):
            parser.error("Missing arguments for 'csv_family_size'.")
        chosen_models_complexity_dependence_on_family_size_csv(
            transitions=args.transitions,
            families_data_file=args.families_data_file,
            min_family_size=args.min_family_size,
            families_folders=args.families_folders,
            output_folder=args.output_folder
        )

    elif args.function == 'plot_family_size':
        if not all([args.dependence_folder_path, args.prefix_list]):
            parser.error("Missing arguments for 'plot_family_size'.")
        chosen_models_complexity_dependence_on_family_size_plot(
            dependence_folder_path=args.dependence_folder_path,
            output_folder=args.output_folder,
            prefix_list=args.prefix_list
        )

    elif args.function == 'anova_test':
        if not all([args.dependence_folder_path, args.prefix_list, args.tested_variable, args.suffix]):
            parser.error("Missing arguments for 'anova_test'.")
        anova_test(
            dependence_folder_path=args.dependence_folder_path,
            prefix_list=args.prefix_list,
            output_folder=args.output_folder,
            suffix=args.suffix,
            tested_variable=args.tested_variable
        )

    elif args.function == 'csv_transition_events':
        if not all([args.transitions, args.raw_results_folder_path, args.input_folder]):
            parser.error("Missing arguments for 'csv_transition_events'.")
        chosen_models_complexity_dependence_on_transition_events_count_csv(
            transitions=args.transitions,
            raw_results_folders_list=[Path(folder) for folder in args.raw_results_folder_path.split(",")],
            output_folder=args.output_folder,
            input_folder=args.input_folder
        )


if __name__ == "__main__":
    main()

# prefix_list = [all_transitions ,baseNum, demi, dupl, gain loss]
# output_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/chosen_models_complexity_dependence/"
# dependence_folder_path = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/chosen_models_complexity_dependence/"
# suffix = family_size_complexity_dependence.csv
# tested_variable = "family_size"

