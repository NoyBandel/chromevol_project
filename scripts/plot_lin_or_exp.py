import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal

def determine_global_ranges(analysis_folder: Path, function_type: Literal["linear", "exponential"] = "linear") -> tuple:
    all_x_values, all_y_values = [], []

    for file in analysis_folder.glob("*_parameters_all.csv"):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            a, b = row['p1_slope'], row['p0_intersection']
            x_min, x_max = row['min_chrom'], row['max_chrom']
            x_full = np.linspace(x_min, x_max, 1000)
            if function_type == "linear":
                y_full = a * x_full + b
            elif function_type == "exponential":
                y_full = b * np.exp(a * x_full)
            all_x_values.extend(x_full)
            all_y_values.extend(y_full)

    global_x_min = min(all_x_values)
    global_x_max = max(all_x_values)
    global_y_min = min(all_y_values)
    global_y_max = max(all_y_values)

    y_margin = (global_y_max - global_y_min) * 0.05
    global_y_min -= y_margin
    global_y_max += y_margin

    return global_x_min, global_x_max, global_y_min, global_y_max

def all_function_plot(
    transition_parameters_file: str,
    output_file_path: str,
    file_name: str,
    global_x_min: float,
    global_x_max: float,
    global_y_min: float,
    global_y_max: float,
    function_type: Literal["linear", "exponential"] = "linear"
) -> None:
    df = pd.read_csv(transition_parameters_file)
    functions_parameters = [
        (row['p1_slope'], row['p0_intersection'], (row['min_chrom'], row['max_chrom']))
        for _, row in df.iterrows()
    ]

    plt.figure(figsize=(10, 6))

    for a, b, x_range in functions_parameters:
        x_min, x_max = x_range
        x_full = np.linspace(global_x_min, global_x_max, 1000)
        if function_type == "linear":
            y_full = a * x_full + b
            y_in_range = a * np.linspace(x_min, x_max, 500) + b
        elif function_type == "exponential":
            y_full = b * np.exp(a * x_full)
            y_in_range = b * np.exp(a * np.linspace(x_min, x_max, 500))
        #plt.plot(x_full, y_full, linestyle='dashed', color='grey', alpha=0.6)
        x_in_range = np.linspace(x_min, x_max, 500)
        plt.plot(x_in_range, y_in_range, linestyle='solid')

    plt.xlim(global_x_min, global_x_max)
    plt.ylim(global_y_min, global_y_max)
    plt.xlabel('x (chromosome number)')
    plt.ylabel('y (event rate)')
    plt.title(f"{function_type.capitalize()} Dependency of {file_name}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True)
    plt.savefig(output_file_path)
    plt.close()

def function_plot_for_all_transitions(analysis_folder: Path, function_type: Literal["linear", "exponential"] = "linear") -> None:
    global_x_min, global_x_max, global_y_min, global_y_max = determine_global_ranges(analysis_folder, function_type)
    analysis_files = list(analysis_folder.glob("*_parameters_all.csv"))
    for file in analysis_files:
        file_name = str(file.name).replace(f"_{function_type}_parameters_all.csv", "")
        output_file_path = analysis_folder / f"no_dashed_{file_name}_all_{function_type}_plot.png"
        all_function_plot(
            str(file),
            str(output_file_path),
            file_name,
            global_x_min,
            global_x_max,
            global_y_min,
            global_y_max,
            function_type,
        )


### run
folder_path = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/")
function_plot_for_all_transitions(folder_path, function_type="linear")

def determine_global_ranges_vs_const(analysis_folder: Path, function_type: Literal["linear", "exponential"] = "linear") -> tuple:
    all_x_values, all_y_values = [], []

    for file in analysis_folder.glob("*_vs_const_exponential_parameters.csv"):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            a, b = row['p1_rate'], row['p0']
            x_min, x_max = row['min_chrom'], row['max_chrom']
            x_full = np.linspace(x_min, x_max, 1000)
            if function_type == "linear":
                y_full = a * x_full + b
            elif function_type == "exponential":
                y_full = b * np.exp(a * x_full)
            all_x_values.extend(x_full)
            all_y_values.extend(y_full)

    global_x_min = min(all_x_values)
    global_x_max = max(all_x_values)
    global_y_min = min(all_y_values)
    global_y_max = max(all_y_values)

    y_margin = (global_y_max - global_y_min) * 0.05
    global_y_min -= y_margin
    global_y_max += y_margin

    return global_x_min, global_x_max, global_y_min, global_y_max

def all_function_plot_vs_const(
    transition_parameters_file: str,
    output_file_path: str,
    file_name: str,
    global_x_min: float,
    global_x_max: float,
    global_y_min: float,
    global_y_max: float,
    function_type: Literal["linear", "exponential"] = "linear"
) -> None:
    df = pd.read_csv(transition_parameters_file)
    functions_parameters = [
        (row['p1_rate'], row['p0'], (row['min_chrom'], row['max_chrom']))
        for _, row in df.iterrows()
    ]

    plt.figure(figsize=(10, 6))

    for a, b, x_range in functions_parameters:
        x_min, x_max = x_range
        x_full = np.linspace(global_x_min, global_x_max, 1000)
        if function_type == "linear":
            y_full = a * x_full + b
            y_in_range = a * np.linspace(x_min, x_max, 500) + b
        elif function_type == "exponential":
            y_full = b * np.exp(a * x_full)
            y_in_range = b * np.exp(a * np.linspace(x_min, x_max, 500))
        #plt.plot(x_full, y_full, linestyle='dashed', color='grey', alpha=0.6)
        x_in_range = np.linspace(x_min, x_max, 500)
        plt.plot(x_in_range, y_in_range, linestyle='solid')

    plt.xlim(global_x_min, global_x_max)
    plt.ylim(global_y_min, global_y_max)
    plt.xlabel('x (chromosome number)')
    plt.ylabel('y (event rate)')
    plt.title(f"{function_type.capitalize()} Dependency of {file_name}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True)
    plt.savefig(output_file_path)
    plt.close()

def function_plot_for_all_transitions_vs_const(analysis_folder: Path, function_type: Literal["linear", "exponential"] = "linear") -> None:
    global_x_min, global_x_max, global_y_min, global_y_max = determine_global_ranges_vs_const(analysis_folder, function_type)
    analysis_files = list(analysis_folder.glob("*_vs_const_exponential_parameters.csv"))
    for file in analysis_files:
        file_name = str(file.name).replace(f"_vs_const_{function_type}_parameters.csv", "")
        output_file_path = analysis_folder / f"no_dashed_{file_name}_all_{function_type}_plot_vs_const.png"
        all_function_plot_vs_const(
            str(file),
            str(output_file_path),
            file_name,
            global_x_min,
            global_x_max,
            global_y_min,
            global_y_max,
            function_type,
        )


def plot_binary_pie_chart(analysis_folder: Path, save_fig_folder: str, transitions: list[str], input_file_recognizer: str) -> None:
    for transition in transitions:
        distribution_file = f"{analysis_folder}/{transition}{input_file_recognizer}_behavior_distribution.csv"
        distribution_df = pd.read_csv(distribution_file, index_col=0)
        distribution_df.reset_index(inplace=True)
        distribution_df.columns = ["ascending_percentage", "descending_percentage"]
        ascending_percentage = distribution_df["ascending_percentage"].iloc[0]
        descending_percentage = distribution_df["descending_percentage"].iloc[0]

        labels = ["Descending", "Ascending"]
        sizes = [descending_percentage, ascending_percentage]
        colors = ["#ff9999", "#66b3ff"]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 14})
        plt.title(f"Binary Pie Chart: {transition}", fontsize=16)

        output_file_path = f"{save_fig_folder}/{transition}{input_file_recognizer}_binary_pie_chart.png"
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()


# ### run
# folder_path = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/")
# function_plot_for_all_transitions_vs_const(folder_path, function_type="exponential")

transitions = ["baseNum", "demi", "dupl", "gain", "loss"]
analysis_folder_exp = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/")
save_fig_folder_exp = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/graphs/"

analysis_folder_lin = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/")
save_fig_folder_lin = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/graphs/"

input_file_recognizer = "_vs_const"
plot_binary_pie_chart(analysis_folder_exp, save_fig_folder_exp, transitions, input_file_recognizer)


def behavior_distribution_to_csv(parameters_folder: str, analysis_output_folder: str, transitions_prefix: list[str], input_file_suffix: str, output_file_suffix: str) -> None:
    for transition in transitions_prefix:
        parameters_file_path = f"{parameters_folder}/{transition}{input_file_suffix}"
        df = pd.read_csv(parameters_file_path)
        if "behavior_class" in df.columns:
            behavior_class_series = df["behavior_class"]
            ascending_count = (behavior_class_series == 1).sum()
            total_count = len(behavior_class_series)
            if total_count > 0:
                ascending_percentage = (ascending_count / total_count) * 100
                descending_percentage = 100 - ascending_percentage
                behavior_distribution = {
                    "ascending_percentage": ascending_percentage,
                    "descending_percentage": descending_percentage
                }
                behavior_distribution_df = pd.DataFrame([behavior_distribution])
                output_file_path = f"{analysis_output_folder}/{transition}{output_file_suffix}_behavior_distribution.csv"
                behavior_distribution_df.to_csv(output_file_path, index=False)

lin_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/"
exp_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/"

input_file_suffix_lin = "_vs_const_linear_parameters.csv"
input_file_suffix_exp = "_vs_const_exponential_parameters.csv"

output_file_suffix = "_vs_const"

# behavior_distribution_to_csv(exp_folder, exp_folder, transitions, input_file_suffix_exp, output_file_suffix)
