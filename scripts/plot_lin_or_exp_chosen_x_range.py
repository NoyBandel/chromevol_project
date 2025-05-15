import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal


def create_zoomed_graphs(
        input_folder: Path,
        output_folder: Path,
        function_type: Literal["linear", "exponential"]
) -> None:
    analysis_files = list(input_folder.glob("*_parameters_all.csv"))
    if not analysis_files:
        print(f"No input files found in {input_folder}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    for file in analysis_files:
        df = pd.read_csv(file)
        if df.empty:
            print(f"File {file} is empty.")
            continue

        if not {'p1_slope', 'p0_intersection', 'min_chrom', 'max_chrom'}.issubset(df.columns):
            print(f"File {file} does not contain the required columns.")
            continue

        functions_parameters = [
            (row['p1_slope'], row['p0_intersection'], (row['min_chrom'], row['max_chrom']))
            for _, row in df.iterrows()
        ]

        all_y_values = []
        for a, b, x_range in functions_parameters:
            x_min, x_max = x_range
            x_full = np.linspace(x_min, x_max, 500)
            if function_type == "linear":
                y_full = a * x_full + b  # Linear equation: y = p1_rate * x + p0
            elif function_type == "exponential":
                y_full = b * np.exp(a * x_full)  # Exponential equation: y = p0 * e^(p1_rate * x)
            all_y_values.extend(y_full)

        if not all_y_values:
            print(f"No valid data found in file {file}.")
            continue

        # Calculate y-axis limits proportional to the current data
        local_y_min, local_y_max = min(all_y_values), max(all_y_values)
        y_margin = (local_y_max - local_y_min) * 0.05
        local_y_min -= y_margin
        local_y_max += y_margin

        file_name = str(file.name).replace(f"_{function_type}_parameters_all.csv", "")
        output_file_path = output_folder / f"zoomed_{file_name}_{function_type}_plot.png"

        plt.figure(figsize=(10, 6))

        for a, b, x_range in functions_parameters:
            x_min, x_max = x_range
            x_full = np.linspace(x_min, x_max, 500)
            if function_type == "linear":
                y_in_range = a * x_full + b
            elif function_type == "exponential":
                y_in_range = b * np.exp(a * x_full)
            plt.plot(x_full, y_in_range, linestyle='solid')

        plt.xlim(0, 100)  # Global x-axis limits
        plt.ylim(local_y_min, local_y_max)  # Dynamic y-axis limits
        plt.xlabel('x (chromosome number)')
        plt.ylabel('y (event rate)')
        plt.title(f"{function_type.capitalize()} Dependency of {file_name}")
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.grid(True)
        plt.savefig(output_file_path)
        plt.close()

def create_zoomed_graphs_vs_const(
        input_folder: Path,
        output_folder: Path,
        function_type: Literal["linear", "exponential"]
) -> None:
    file_pattern = f"*vs_const_{function_type}_parameters.csv"
    analysis_files = list(input_folder.glob(file_pattern))
    if not analysis_files:
        print(f"No input files found in {input_folder} matching pattern {file_pattern}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    for file in analysis_files:
        df = pd.read_csv(file)
        if df.empty:
            print(f"File {file} is empty.")
            continue

        # Ensure the required columns are present with the correct names
        if not {'p1_rate', 'p0', 'min_chrom', 'max_chrom'}.issubset(df.columns):
            print(f"File {file} does not contain the required columns.")
            continue

        # Extract the slope (p1_rate), intercept (p0), and chromosome range
        functions_parameters = [
            (row['p1_rate'], row['p0'], (row['min_chrom'], row['max_chrom']))
            for _, row in df.iterrows()
        ]

        all_y_values = []
        for a, b, x_range in functions_parameters:
            x_min, x_max = x_range
            x_full = np.linspace(x_min, x_max, 500)
            if function_type == "linear":
                y_full = a * x_full + b  # Linear equation: y = p1_rate * x + p0
            elif function_type == "exponential":
                y_full = b * np.exp(a * x_full)  # Exponential equation: y = p0 * e^(p1_rate * x)
            all_y_values.extend(y_full)

        if not all_y_values:
            print(f"No valid data found in file {file}.")
            continue

        # Calculate y-axis limits proportional to the current data
        local_y_min, local_y_max = min(all_y_values), max(all_y_values)
        y_margin = (local_y_max - local_y_min) * 0.05
        local_y_min -= y_margin
        local_y_max += y_margin

        # Extract file name for output
        file_name = str(file.name).replace(f"_vs_const_{function_type}_parameters.csv", "")
        output_file_path = output_folder / f"zoomed_{file_name}_vs_const_{function_type}_plot.png"

        plt.figure(figsize=(10, 6))

        for a, b, x_range in functions_parameters:
            x_min, x_max = x_range
            x_full = np.linspace(x_min, x_max, 500)
            if function_type == "linear":
                y_in_range = a * x_full + b
            elif function_type == "exponential":
                y_in_range = b * np.exp(a * x_full)
            plt.plot(x_full, y_in_range, linestyle='solid')

        plt.xlim(0, 100)  # Global x-axis limits
        plt.ylim(local_y_min, local_y_max)  # Dynamic y-axis limits
        plt.xlabel('x (chromosome number)')
        plt.ylabel('y (event rate)')
        plt.title(f"{function_type.capitalize()} Dependency of {file_name} (better than CONST)")
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.grid(True)
        plt.savefig(output_file_path)
        plt.close()


# Directories
input_folder_linear = Path(
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/")
input_folder_exponential = Path(
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/")
output_folder_linear = Path(
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/graphs/")
output_folder_exponential = Path(
    "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/exponential_analysis/graphs/")

# run
create_zoomed_graphs_vs_const(
    input_folder=input_folder_linear,
    output_folder=output_folder_linear,
    function_type="linear"
)
create_zoomed_graphs_vs_const(
    input_folder=input_folder_exponential,
    output_folder=output_folder_exponential,
    function_type="exponential"
)
