import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_histogram(data: list[float], title: str, filename: str, folder: Path, subgroup: list, x_range: float, use_fixed_x: bool) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, alpha=0.7, label="Full Data", color="skyblue")
    plt.hist(subgroup, bins=10, alpha=0.7, label="Better Than CONST", color="orange")
    plt.title(title)
    plt.xlabel('Transition Rate')
    plt.ylabel('Frequency')

    if use_fixed_x:
        plt.xlim(left=-x_range, right=x_range)  # Center 0 in the middle for fixed x-range
    else:
        min_value = min(data + subgroup)
        max_value = max(data + subgroup)
        max_abs_value = max(abs(min_value), abs(max_value))
        plt.xlim(left=-max_abs_value, right=max_abs_value)  # Center 0 in the middle for proportional x-range

    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label="0")
    median_value = np.median(data)
    plt.axvline(median_value, color='blue', linestyle='-', linewidth=1.5, label=f"Median (Full Data) = {median_value:.2f}")
    median_subgroup_value = np.median(subgroup)
    plt.axvline(median_subgroup_value, color='green', linestyle='-', linewidth=1.5, label=f"Median (Better Than CONST) = {median_subgroup_value:.2f}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(folder / filename)
    plt.close()

def analyze_data_separately(input_file_path: str, transition_name: str, linear_folder_path: str, exp_folder_path: str, linear_chosen_vs_const_path: str, exp_chosen_vs_const_path: str, x_range: float) -> None:
    data = pd.read_csv(input_file_path, index_col=0)
    lin_rate: list[float] = data['lin_rate'].tolist()
    exp_rate: list[float] = data['exp_rate'].tolist()
    lin_subgroup_indices = pd.read_csv(linear_chosen_vs_const_path, index_col=0).index.tolist()
    exp_subgroup_indices = pd.read_csv(exp_chosen_vs_const_path, index_col=0).index.tolist()
    chosen_lin_subgroup = [lin_rate[data.index.get_loc(i)] for i in lin_subgroup_indices]
    chosen_exp_subgroup = [exp_rate[data.index.get_loc(i)] for i in exp_subgroup_indices]
    linear_folder = Path(linear_folder_path)
    exp_folder = Path(exp_folder_path)

    plot_histogram(lin_rate, f"Histogram of {transition_name} linear rate (Fixed x Range)", f"{transition_name}_lin_rate_histogram_fixed_x.png", linear_folder, chosen_lin_subgroup, x_range, use_fixed_x=True)
    plot_histogram(exp_rate, f"Histogram of {transition_name} exponential rate (Fixed x Range)", f"{transition_name}_exp_rate_histogram_fixed_x.png", exp_folder, chosen_exp_subgroup, x_range, use_fixed_x=True)

    plot_histogram(lin_rate, f"Histogram of {transition_name} linear rate (Proportional x Range)", f"{transition_name}_lin_rate_histogram_proportional_x.png", linear_folder, chosen_lin_subgroup, x_range, use_fixed_x=False)
    plot_histogram(exp_rate, f"Histogram of {transition_name} exponential rate (Proportional x Range)", f"{transition_name}_exp_rate_histogram_proportional_x.png", exp_folder, chosen_exp_subgroup, x_range, use_fixed_x=False)

def main():
    parser = argparse.ArgumentParser(description="Analyze transition data and generate plots.")
    parser.add_argument("input_folder_path", type=str, help="Path to the folder of _lin_exp_data file.")
    parser.add_argument("linear_folder_path", type=str, help="Path to the folder where linear analysis results will be saved.")
    parser.add_argument("exp_folder_path", type=str, help="Path to the folder where exponential analysis results will be saved.")
    args = parser.parse_args()
    transition_names = ["gain", "loss", "dupl", "demi", "baseNum"]
    global_x_range = None
    for transition in transition_names:
        lin_exp_file_path = f"{args.input_folder_path}/{transition}_lin_exp_data.csv"
        linear_chosen_vs_const_path = f"{args.linear_folder_path}/{transition}_vs_const_linear_parameters.csv"
        exp_chosen_vs_const_path = f"{args.exp_folder_path}/{transition}_vs_const_exponential_parameters.csv"
        data = pd.read_csv(lin_exp_file_path, index_col=0)
        lin_rate = data['lin_rate'].tolist()
        exp_rate = data['exp_rate'].tolist()
        lin_subgroup_indices = pd.read_csv(linear_chosen_vs_const_path, index_col=0).index.tolist()
        exp_subgroup_indices = pd.read_csv(exp_chosen_vs_const_path, index_col=0).index.tolist()
        chosen_lin_subgroup = [lin_rate[data.index.get_loc(i)] for i in lin_subgroup_indices]
        chosen_exp_subgroup = [exp_rate[data.index.get_loc(i)] for i in exp_subgroup_indices]
        x_min = min(np.min(lin_rate), np.min(exp_rate), np.min(chosen_lin_subgroup), np.min(chosen_exp_subgroup))
        x_max = max(np.max(lin_rate), np.max(exp_rate), np.max(chosen_lin_subgroup), np.max(chosen_exp_subgroup))
        x_range = max(abs(x_min), abs(x_max))
        if global_x_range is None or x_range > global_x_range:
            global_x_range = x_range
    for transition in transition_names:
        lin_exp_file_path = f"{args.input_folder_path}/{transition}_lin_exp_data.csv"
        linear_chosen_vs_const_path = f"{args.linear_folder_path}/{transition}_vs_const_linear_parameters.csv"
        exp_chosen_vs_const_path = f"{args.exp_folder_path}/{transition}_vs_const_exponential_parameters.csv"
        analyze_data_separately(lin_exp_file_path, transition, args.linear_folder_path, args.exp_folder_path, linear_chosen_vs_const_path, exp_chosen_vs_const_path, global_x_range)

if __name__ == "__main__":
    main()
