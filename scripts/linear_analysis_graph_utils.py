import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from pathlib import Path
import numpy as np

from constants import *

# slope distribution histogram
def plot_slope_distribution_histogram(linear_analysis_df: DataFrame, save_fig_folder: Path, transition: str, analysis_type: str) -> None:
    data = linear_analysis_df["lin_slope"].tolist()

    # Calculate counts for annotation
    behavior_series = linear_analysis_df["lin_behavior_class"].astype(str)
    total = len(behavior_series)
    ascending = sum(val == "1" for val in behavior_series)
    descending = total - ascending

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')

    # Vertical line at 0 (neutral slope)
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1.5, label='Zero Slope')

    # Vertical line at median
    median_val = np.median(data)
    plt.axvline(median_val, color='red', linestyle='solid', linewidth=2, label=f'Median: {median_val:.4f}')

    plt.title(
        f"Slope Distribution: {transition} ({analysis_type})\n"
        f"Total: {total} | Ascending: {ascending} | Descending: {descending}",
        fontsize=14
    )
    plt.xlabel('Transition Rate')
    plt.ylabel('Frequency')

    max_abs = max(abs(np.min(data)), abs(np.max(data)))
    plt.xlim(left=-max_abs, right=max_abs)

    # Add legend for vertical lines
    plt.legend(loc='upper right', fontsize=10)

    output_file_path = save_fig_folder / f"{transition}_{analysis_type}_slope_histogram.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()


# binary pie chart
def plot_binary_pie_chart(linear_analysis_df: DataFrame, save_fig_folder: Path, transition: str, analysis_type: str) -> None:
    behavior_class_series = linear_analysis_df["lin_behavior_class"].astype(str)
    total = len(behavior_class_series)

    ascending = sum(val == "1" for val in behavior_class_series)
    descending = total - ascending

    ascending_pct = ascending / total * 100 if total > 0 else 0
    descending_pct = descending / total * 100 if total > 0 else 0

    sizes = [descending, ascending]  # Use counts for slice sizes
    colors = ["#ff9999", "#66b3ff"]

    plt.figure(figsize=(6, 6))

    # Plot pie with percentages inside slices, no labels
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else '',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'color': 'black'}
    )

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#66b3ff", label=f"Ascending: {ascending_pct:.1f}% ({ascending})"),
        Patch(color="#ff9999", label=f"Descending: {descending_pct:.1f}% ({descending})")
    ]

    # Legend below the chart
    plt.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        fontsize=12,
        frameon=True
    )

    plt.title(f"Binary Pie Chart: {transition} ({analysis_type})\nTotal Samples: {total}", fontsize=16)
    plt.tight_layout()

    output_file_path = save_fig_folder / f"{transition}_{analysis_type}_binary_pie_chart.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()




# all families linear function plot
def plot_line_functions(linear_analysis_df: DataFrame, save_fig_folder: Path, transition: str, analysis_type: str) -> None:
    x = np.linspace(0, 100, 500)  # chromosome number from 0 to 100

    plt.figure(figsize=(10, 6))

    for _, row in linear_analysis_df.iterrows():
        slope = row["lin_slope"]
        intercept = row["lin_intersection"]

        y = slope * x + intercept

        # Color by slope sign
        color = "#66b3ff" if slope > 0 else "#ff9999"

        plt.plot(x, y, color=color, alpha=0.6)

    plt.xlabel("Chromosome Number")
    plt.ylabel("Transition Rate")

    # Calculate behavior counts and percentages
    behavior_series = linear_analysis_df["lin_behavior_class"].astype(str)
    total = len(behavior_series)
    ascending = sum(val == "1" for val in behavior_series)
    descending = total - ascending

    ascending_pct = ascending / total * 100 if total > 0 else 0
    descending_pct = descending / total * 100 if total > 0 else 0

    # Title includes total samples
    plt.title(
        f"Linear Dependence of {transition} ({analysis_type})\n"
        f"Total Samples: {total}",
        fontsize=16
    )

    from matplotlib.patches import Patch

    # Create legend patches with behavior labels and percentages
    legend_patches = [
        Patch(color="#66b3ff", label=f"Ascending: {ascending_pct:.1f}% ({ascending})"),
        Patch(color="#ff9999", label=f"Descending: {descending_pct:.1f}% ({descending})")
    ]

    plt.legend(
        handles=legend_patches,
        loc='upper left',
        fontsize=12,
        frameon=True
    )

    output_file_path = save_fig_folder / f"{transition}_{analysis_type}_line_functions.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()


# all families linear function plot by chosen model
def plot_line_functions_by_aicc(linear_analysis_df: DataFrame, save_fig_folder: Path, transition: str, analysis_type: str) -> None:
    x = np.linspace(0, 100, 500)  # chromosome number from 0 to 100

    plt.figure(figsize=(10, 6))

    # Define colors for AICc columns
    color_map = {
        'linear_AICc': 'green',
        'constant_AICc': 'orange',
        'ignore_AICc': 'red'
    }

    for _, row in linear_analysis_df.iterrows():
        slope = row["lin_slope"]
        intercept = row["lin_intersection"]

        y = slope * x + intercept

        # Find which AICc score is lowest for this row
        scores = {
            'linear_AICc': row['linear_AICc'],
            'constant_AICc': row['constant_AICc'],
            'ignore_AICc': row['ignore_AICc']
        }
        best_model = min(scores, key=scores.get)
        color = color_map[best_model]

        plt.plot(x, y, color=color, alpha=0.6)

    plt.xlabel("Chromosome Number")
    plt.ylabel("Transition Rate")
    plt.title(f"Linear Functions Colored by Lowest AICc: {transition} ({analysis_type})", fontsize=16)

    from matplotlib.patches import Patch

    # Create legend patches with colors and labels
    legend_patches = [
        Patch(color='green', label='linear_AICc (green)'),
        Patch(color='orange', label='constant_AICc (orange)'),
        Patch(color='red', label='ignore_AICc (red)')
    ]

    plt.legend(handles=legend_patches, loc='best', fontsize=12, frameon=True)

    output_file_path = save_fig_folder / f"{transition}_{analysis_type}_line_functions_by_aicc.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()



# ---main---
# main_graphs_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/graphs/")
# main_linear_analysis_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/")
#
# for transition in LABEL_TRANSITIONS_LST:
#     input_folder_path = main_linear_analysis_folder / transition
#     output_folder_path = main_graphs_folder / transition / "linear_analysis"
#
#     for file_path in input_folder_path.glob(f"{transition}_*.csv"):
#         analysis_type = file_path.stem.replace(f"{transition}_", "")
#         linear_analysis_df = pd.read_csv(file_path)
#
#         print("plot_binary_pie_chart")
#         plot_binary_pie_chart(linear_analysis_df, output_folder_path, transition, analysis_type)
#
#         print("plot_slope_distribution_histogram")
#         plot_slope_distribution_histogram(linear_analysis_df, output_folder_path, transition, analysis_type)
#
#         print("plot_line_functions")
#         plot_line_functions(linear_analysis_df, output_folder_path, transition, analysis_type)
#
#         print("plot_line_functions_by_aicc")
#         plot_line_functions_by_aicc(linear_analysis_df, output_folder_path, transition, analysis_type)



# for after reciprocal runs
main_graphs_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/graphs/")
main_linear_analysis_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/linear_analysis/")

for transition in LABEL_TRANSITIONS_LST:
    if transition in [LABEL_BASE_NUM, LABEL_DEMI]:
        continue
    input_folder_path = main_linear_analysis_folder / transition
    output_folder_path = main_graphs_folder / transition / "linear_analysis_best_model_after_reciprocal_runs"

    file_names_dict = {
        f"{transition}_reciprocal_runs_linear_analysis.csv": "linear_analysis",
        f"{transition}_reciprocal_runs_best_linear_AICc.csv": "linear_AICc",
        f"{transition}_reciprocal_runs_ignore_not_best.csv": "ignore_not_best"
    }

    for file_path in input_folder_path.glob(f"{transition}_*.csv"):
        if file_path.name not in file_names_dict.keys():
            continue

        analysis_type = file_names_dict[file_path.name]
        linear_analysis_df = pd.read_csv(file_path)

        print("plot_binary_pie_chart")
        plot_binary_pie_chart(linear_analysis_df, output_folder_path, transition, analysis_type)

        print("plot_slope_distribution_histogram")
        plot_slope_distribution_histogram(linear_analysis_df, output_folder_path, transition, analysis_type)

        print("plot_line_functions")
        plot_line_functions(linear_analysis_df, output_folder_path, transition, analysis_type)

        print("plot_line_functions_by_aicc")
        plot_line_functions_by_aicc(linear_analysis_df, output_folder_path, transition, analysis_type)

