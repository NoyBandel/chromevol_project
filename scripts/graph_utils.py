import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import ast
import argparse
from constants import *
from collections import defaultdict
import os

# --- consts ---
RAINBOW = "rainbow"

# --- functions ---

def Const(params: List[float], x: np.ndarray) -> np.ndarray:
    return np.array([params[0] for i in range(len(x))])


def Linear_BD(params: List[float], x: np.ndarray) -> np.ndarray:
    return x * params[0]


def Linear(params: List[float], x: np.ndarray) -> np.ndarray:
    return params[0] + params[1]*(x-1)


def Exp(params: List[float], x: np.ndarray) -> np.ndarray:
    return params[0] * np.exp(params[1]*(x-1))


def Rev_Sigmoid(params: List[float], x: np.ndarray) -> np.ndarray:
    return params[0]/(1+np.exp(params[1]-(params[2]*x)))


def Log_normal(params: List[float], x: np.ndarray) -> np.ndarray:
    range_factor = params[0]
    scaling_factor = max(x)/4
    transformed = x / scaling_factor
    mu = params[1]
    sigma = params[2]
    pi = np.pi
    eq_part_1 = 1/(transformed * sigma * np.sqrt(2 * pi))
    eq_part_2 = np.exp(-((np.log(transformed)-mu)**2/(2*(sigma**2))))
    return range_factor * eq_part_1 * eq_part_2

# --- func_map ---
ce_func_name_to_func_map = {
    CE_LINEAR: Linear,
    CE_LINEAR_BD: Linear_BD,
    CE_EXP: Exp,
    CE_REVERSE_SIGMOID: Rev_Sigmoid,
    CE_LOGNORMAL: Log_normal,
    CE_CONSTANT: Const,
    CE_IGNORE: None
}

def all_chosen_func_one_graph(
    all_chosen_models_path: str,
    graphs_folder: str,
    family_data_with_chrom: str,
) -> None:
    chrom_range_df = pd.read_csv(family_data_with_chrom, index_col=0)
    families_chrom_range: Dict[str, Tuple[int, int]] = chrom_range_df[[LABEL_MIN_CHROM, LABEL_MAX_CHROM]].apply(tuple, axis=1).to_dict()
    all_chosen_models_df = pd.read_csv(all_chosen_models_path, index_col=0)

    color_map = {
        CE_CONSTANT: 'blue',
        CE_LINEAR: 'green',
        CE_LINEAR_BD: 'red',
        CE_EXP: 'orange',
        CE_LOGNORMAL: 'purple',
        CE_REVERSE_SIGMOID: 'brown',
    }

    for transition in LABEL_TRANSITIONS_LST:
        save_fig_folder = os.path.join(graphs_folder, transition, ALL_CHOSEN_GRAPH_FOLDER)
        group_y_values = defaultdict(list)
        family_lines = []
        counts_per_func = defaultdict(int)

        # --------------- Overview Plot ------------------
        plt.figure(figsize=(12, 8))
        total_lines = 0
        ignored_families = []
        max_x, max_y = 0, 0
        const_family_count = 0

        for family in all_chosen_models_df.index:
            ce_func_name = all_chosen_models_df.loc[family, f"{transition}_{LABEL_CHOSEN_MODEL}"]
            func = ce_func_name_to_func_map.get(ce_func_name)

            if func is None:
                if ce_func_name == CE_IGNORE:
                    ignored_families.append(family)
                continue

            if ce_func_name == CE_CONSTANT:
                const_family_count += 1

            try:
                params_str = all_chosen_models_df.loc[family, f"{transition}_{LABEL_PARAMS}"]
                params_list = [float(param) for param in ast.literal_eval(params_str)]
                chrom_range = families_chrom_range.get(family, (1, 100))
                x_values = np.linspace(chrom_range[0], chrom_range[1], 100)
                y_values = func(params_list, x_values)

                group_y_values[ce_func_name].append(y_values)
                family_lines.append((ce_func_name, family, x_values, y_values))

                counts_per_func[ce_func_name] += 1  # Track count

                max_x = max(max_x, x_values.max())
                max_y = max(max_y, y_values.max())
                total_lines += 1

            except Exception:
                continue

        if total_lines > 0:
            plt.xlabel("Number of Chromosomes")
            plt.ylabel(f"{transition} Rate")

            labels_drawn = set()
            for ce_func_name, family, x_vals, y_vals in family_lines:
                color = color_map.get(ce_func_name, 'gray')
                label = None
                if ce_func_name not in labels_drawn:
                    count = counts_per_func[ce_func_name]
                    label = f"{FUNC_CE_TO_LABEL[ce_func_name]} ({count})"
                    labels_drawn.add(ce_func_name)
                plt.plot(x_vals, y_vals, color=color, label=label)

            plt.title(f"Rate Functions for {transition}\n(n = {total_lines}, ignored = {len(ignored_families)})")
            plt.xlim(0, max_x * 1.1)
            plt.ylim(0, max_y * 1.1)
            plt.legend()
            plt.tight_layout()

            output_path = os.path.join(save_fig_folder, f"{transition}_{ALL_CHOSEN_GRAPH_SUFFIX}")
            plt.savefig(output_path)
            plt.close()

        # --------------- Plot without CONST ------------------
        plt.figure(figsize=(12, 8))
        max_x_nc, max_y_nc = 0, 0
        total_lines_nc = 0

        counts_per_func_nc = defaultdict(int)
        for ce_func_name, family, x_values, y_values in family_lines:
            if ce_func_name == CE_CONSTANT:
                continue
            counts_per_func_nc[ce_func_name] += 1

            max_x_nc = max(max_x_nc, x_values.max())
            max_y_nc = max(max_y_nc, y_values.max())
            total_lines_nc += 1

        if total_lines_nc > 0:
            labels_drawn_nc = set()
            for ce_func_name, family, x_vals, y_vals in family_lines:
                if ce_func_name == CE_CONSTANT:
                    continue
                color = color_map.get(ce_func_name, 'gray')
                label = None
                if ce_func_name not in labels_drawn_nc:
                    count = counts_per_func_nc[ce_func_name]
                    label = f"{FUNC_CE_TO_LABEL[ce_func_name]} ({count})"
                    labels_drawn_nc.add(ce_func_name)
                plt.plot(x_vals, y_vals, color=color, label=label)

            plt.xlabel("Number of Chromosomes")
            plt.ylabel(f"{transition} Rate")
            plt.title(
                f"Rate Functions for {transition} without const\n(n = {total_lines_nc}, const = {const_family_count}, ignored = {len(ignored_families)})"
            )
            plt.xlim(0, max_x_nc * 1.1)
            plt.ylim(0, max_y_nc * 1.1)
            plt.legend()
            plt.tight_layout()

            output_path_nc = os.path.join(save_fig_folder, f"{transition}_no_const_{ALL_CHOSEN_GRAPH_SUFFIX}")
            plt.savefig(output_path_nc)
            plt.close()

        # --------------- Annotated (with family name) Zoomed Plot WITHOUT CONST ------------------
        plt.figure(figsize=(12, 8))
        for ce_func_name, family, x_values, y_values in family_lines:
            if ce_func_name == CE_CONSTANT:
                continue

            color = color_map.get(ce_func_name, 'gray')
            plt.plot(x_values, y_values, color=color)
            mid_idx = len(x_values) // 2
            x_mid = float(x_values[mid_idx])
            y_mid = float(y_values[mid_idx])
            plt.text(x_mid, y_mid, family, fontsize=7, color=color)

        group_y_values_no_const = {k: v for k, v in group_y_values.items() if k != CE_CONSTANT}
        max_mean_y_no_const = max(np.mean(np.concatenate(v)) for v in group_y_values_no_const.values() if v)

        plt.xlabel("Number of Chromosomes")
        plt.ylabel(f"{transition} Rate")
        plt.title(
            f"Annotated Zoomed Rate Functions for {transition} without const\n(n = {total_lines - const_family_count}, const = {const_family_count}, ignored = {len(ignored_families)})"
        )
        plt.xlim(0, 100)
        plt.ylim(0, max_mean_y_no_const * 1.1)

        for ce_func_name in counts_per_func_nc:
            label = f"{FUNC_CE_TO_LABEL[ce_func_name]} ({counts_per_func.get(ce_func_name, 0)})"
            color = color_map.get(ce_func_name, 'gray')
            plt.plot([], [], label=label, color=color)

        plt.legend()
        plt.tight_layout()

        zoomed_no_const_path = os.path.join(save_fig_folder, f"{transition}_zoomed_no_const_{ZOOMED_SUFFIX}")
        plt.savefig(zoomed_no_const_path)
        plt.close()


# all_chosen_models_path = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/all_chosen_models.csv"
# graphs_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/graphs/"
# family_data_with_chrom = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data_with_chrom.csv"
# all_chosen_func_one_graph(all_chosen_models_path, graphs_folder, family_data_with_chrom)

def all_func_one_graph_for_each_family(
        csv_raw_results_folder: str,
        all_chosen_models_path: str,
        graphs_folder: str,
        family_data_size_and_chrom: str,
        functions_to_ignore: List[str],
        color_gradient: List[str]
) -> None:

    def generate_function_curves(params_list, x_values):
        curves = {}
        for label_func_type in params_list.index:
            raw_params = params_list[label_func_type]
            func_params = ast.literal_eval(raw_params)
            func_params = [float(param) for param in func_params]
            ce_func_type = FUNC_LABEL_TO_CE[label_func_type]
            func = ce_func_name_to_func_map.get(ce_func_type)
            if func is None:
                continue
            y_values = func(func_params, x_values)
            curves[label_func_type] = y_values
        return curves

    family_data_df = pd.read_csv(family_data_size_and_chrom, index_col=0)
    families_chrom_range = family_data_df[[LABEL_MIN_CHROM, LABEL_MAX_CHROM]].apply(tuple, axis=1).to_dict()
    families_size_dict = family_data_df[LABEL_FAMILY_SIZE].to_dict()
    all_chosen_models_df = pd.read_csv(all_chosen_models_path, index_col=0)

    for transition in LABEL_TRANSITIONS_LST:
        raw_results_file = os.path.join(csv_raw_results_folder, f"{transition}_{RAW_RESULTS}")
        raw_results_df = pd.read_csv(raw_results_file, index_col=0)

        params_values_rows = raw_results_df.loc[raw_results_df.index.str.endswith(f"_{LABEL_PARAM}")]
        params_values_rows.index = params_values_rows.index.str.replace(f"_{LABEL_PARAM}", "")
        filtered_columns = [col for col in params_values_rows.columns if col not in functions_to_ignore]
        params_values_rows = params_values_rows[filtered_columns]

        for family_name, params_list in params_values_rows.iterrows():
            AICc_values = raw_results_df.loc[f"{family_name}_{LABEL_AICc}"]
            AICc_values = AICc_values.drop(labels=functions_to_ignore)
            x_start, x_end = families_chrom_range[family_name]
            x_values = np.linspace(x_start, x_end, 100)

            family_size = families_size_dict[family_name]
            sorted_func_types = AICc_values.sort_values().index.tolist()
            ce_best_model = all_chosen_models_df.loc[family_name, f"{transition}_{LABEL_CHOSEN_MODEL}"]
            label_best_model = FUNC_CE_TO_LABEL[ce_best_model]

            curves = generate_function_curves(params_list, x_values)

            # --- Red-Blue Plot ---
            plt.figure(figsize=(10, 6))
            for label_func_type, y_values in curves.items():
                color = "red" if label_func_type == label_best_model else "blue"
                plt.plot(x_values, y_values, color=color, label=label_func_type)

            plt.legend(handles=[plt.Line2D([0], [0], color="red", lw=2, label=f"Best model: {label_best_model}")],
                       fontsize="small", loc="upper left")

            plt.title(f"Rate Functions for {family_name} - Transition: {transition}\n(family size = {family_size})")
            plt.xlabel("Number of Chromosomes")
            plt.ylabel(f"{transition} rate")
            plt.xlim(left=0)
            plt.ylim(bottom=0)

            output_path_red = os.path.join(graphs_folder, transition, EACH_FAMILY, f"{transition}_{family_name}.png")
            plt.tight_layout()
            plt.savefig(output_path_red)
            plt.close()

            # --- Rainbow Plot ---
            plt.figure(figsize=(10, 6))
            handles_labels = []
            color_map = {
                func_type: color_gradient[i % len(color_gradient)]
                for i, func_type in enumerate(sorted_func_types)
            }

            for func_label in sorted_func_types:
                color = color_map[func_label]
                label = f"{func_label} (AICc: {float(AICc_values[func_label]):.2f})"

                if func_label not in curves:
                    handles_labels.append((plt.Line2D([], [], color=color, linewidth=2), label))
                    continue

                y_values = curves[func_label]
                plt.plot(x_values, y_values, color=color, label=label)
                handles_labels.append((plt.Line2D([0], [0], color=color, lw=2), label))

            handles_labels = sorted(
                handles_labels,
                key=lambda hl: sorted_func_types.index(hl[1].split(" ")[0])
            )

            if handles_labels:
                handles, labels = zip(*handles_labels)
                plt.legend(handles, labels, fontsize="small", loc="upper left")

            plt.title(f"Rate Functions for {family_name} - Transition: {transition}\n(family size = {family_size})")
            plt.xlabel("Number of Chromosomes")
            plt.ylabel("{transition} rate")
            plt.xlim(left=0)
            plt.ylim(bottom=0)

            output_path = os.path.join(graphs_folder, transition, EACH_FAMILY, f"{transition}_{family_name}_rainbow.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()


# csv_raw_results_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/"
# all_chosen_models_path = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/all_chosen_models.csv"
# graphs_folder = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/graphs/"
# family_data_size_and_chrom = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data_with_chrom.csv"
# functions_to_ignore = [LABEL_LINEAR_BD, LABEL_REVERSE_SIGMOID]
# color_gradient = ["red", "orange", "green", "purple", "pink"]
# all_func_one_graph_for_each_family(csv_raw_results_folder, all_chosen_models_path, graphs_folder, family_data_size_and_chrom, functions_to_ignore, color_gradient)



#
#
# def main():
#     parser = argparse.ArgumentParser(description="Generate rate function plots for chromosome evolution.")
#     parser.add_argument("function", choices=["chosen", "family", "family_rainbow"],
#         help="Choose the function to use: 'chosen' for create_all_chosen_func_one_graph, "
#              "'family' for create_all_func_one_graph_for_each_family, "
#              "'family_rainbow' for create_all_func_one_graph_for_each_family_rainbow."
#     )
#     parser.add_argument("analysis_folder", type=str, help="Path to the folder containing analysis files.")
#     parser.add_argument("transitions", type=str, nargs='+', help="List of transitions.")
#     parser.add_argument("families_chrom_range", type=str,help="Family chromosome ranges as a dictionary string, e.g. \"{'family1': (10, 100), 'family2': (20, 120)}\"")
#     parser.add_argument("--each_family_graphs_folder", type=str,help="Path to the folder where the output graphs for each family will be saved (required for 'family' and 'family_rainbow').")
#     parser.add_argument("--save_fig_folder", type=str,help="Path to the folder where the output figure for 'chosen' function will be saved (required for 'chosen').")
#
#     args = parser.parse_args()
#
#     families_chrom_range = ast.literal_eval(args.families_chrom_range)
#
#     if args.function == "chosen":
#         if not args.save_fig_folder:
#             parser.error("The '--save_fig_folder' argument is required for 'chosen' function.")
#         create_all_chosen_func_one_graph(
#             args.analysis_folder,
#             args.save_fig_folder,
#             args.transitions,
#             families_chrom_range
#         )
#     elif args.function == "family":
#         if not args.each_family_graphs_folder:
#             parser.error("The '--each_family_graphs_folder' argument is required for 'family' function.")
#         create_all_func_one_graph_for_each_family(
#             args.analysis_folder,
#             args.each_family_graphs_folder,
#             args.transitions,
#             families_chrom_range
#         )
#     elif args.function == "family_rainbow":
#         if not args.each_family_graphs_folder:
#             parser.error("The '--each_family_graphs_folder' argument is required for 'family_rainbow' function.")
#         create_all_func_one_graph_for_each_family_rainbow(
#             args.analysis_folder,
#             args.each_family_graphs_folder,
#             args.transitions,
#             families_chrom_range
#         )
#
#
# if __name__ == "__main__":
#     main()
#
