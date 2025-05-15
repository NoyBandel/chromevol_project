import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import ast
import argparse


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

def create_all_chosen_func_one_graph(
    analysis_folder: str,
    save_fig_folder: str,
    transitions: List[str],
    families_chrom_range: Dict[str, Tuple[int, int]]
) -> None:
    func_map = {
        "linear": Linear,
        "linear-bd": Linear_BD,
        "exponential": Exp,
        "reverse-sigmoid": Rev_Sigmoid,
        "log-normal": Log_normal,
        "constant": Const,
        "ignore": None
    }

    for transition in transitions:
        plt.figure(figsize=(12, 8))
        chosen_model_file = f"{analysis_folder}{transition}_chosen_model_with_params.csv"
        try:
            chosen_model_df = pd.read_csv(chosen_model_file, index_col=0)
        except Exception:
            continue

        ignored_families = []
        legend_handles = []
        total_lines = 0
        max_x, max_y = 0, 0

        for family in chosen_model_df.index:
            func_name = chosen_model_df.loc[family, "Chosen_Model"].lower()
            func = func_map.get(func_name)

            if func is None:
                if func_name == "ignore":
                    ignored_families.append(family)
                continue

            try:
                params_str = chosen_model_df.loc[family, "func_parameters"]
                params_list = ast.literal_eval(params_str)
                params_list = [float(param) for param in params_list]
                chrom_range = families_chrom_range.get(family, (1, 100))
                x_values = np.linspace(chrom_range[0], chrom_range[1], 100)
                y_values = func(params_list, x_values)
                label = f"{family} - {func_name}"
                line, = plt.plot(x_values, y_values, label=label)
                legend_handles.append((line, label))
                total_lines += 1

                max_x = max(max_x, x_values.max())
                max_y = max(max_y, y_values.max())
            except Exception:
                continue

        if total_lines > 0:
            plt.xlabel("Number of Chromosomes (i)")
            plt.ylabel("Rate Function Value")
            plt.title(f"Rate Functions for {transition}\n(n = {total_lines}, ignored = {len(ignored_families)})")

            plt.xlim(0, max_x * 1.1)
            plt.ylim(0, max_y * 1.1)
            plt.tight_layout()

            output_path = f"{save_fig_folder}{transition}_all_chosen_rate_functions_plot.png"
            plt.savefig(output_path)

            legend_fig, legend_ax = plt.subplots(figsize=(12, 8))
            legend_ax.axis('off')

            legend_ax.legend(
                handles=[handle for handle, _ in legend_handles],
                labels=[label for _, label in legend_handles],
                loc='upper left',
                fontsize='small',
                title="Plotted Families"
            )

            ignored_text = f"Ignored Families ({len(ignored_families)}):\n" + "\n".join(ignored_families)
            legend_ax.text(
                1.05, 0.5, ignored_text,
                transform=legend_ax.transAxes,
                fontsize='small',
                verticalalignment='center'
            )

            legend_path = f"{save_fig_folder}/{transition}_all_chosen_rate_functions_legend.png"
            legend_fig.savefig(legend_path)


def create_all_func_one_graph_for_each_family(
        analysis_folder: str,
        each_family_graphs_folder: str,
        transitions: list[str],
        families_chrom_range: Dict[str, Tuple[int, int]]
) -> None:
    func_map = {
        "linear": Linear,
        "linear-bd": Linear_BD,
        "exponential": Exp,
        "reverse-sigmoid": Rev_Sigmoid,
        "log-normal": Log_normal,
        "constant": Const,
        "ignore": None
    }

    for transition in transitions:
        chosen_models_file = f"{analysis_folder}{transition}_chosen_model_with_params.csv"
        raw_results_file = f"{analysis_folder}{transition}_raw_results.csv"

        chosen_models_df = pd.read_csv(chosen_models_file, index_col=0)
        raw_results_df = pd.read_csv(raw_results_file, index_col=0)

        params_values_rows = raw_results_df.loc[raw_results_df.index.str.endswith("_param")]
        params_values_rows.index = params_values_rows.index.str.replace("_param", "")

        for family, params_list in params_values_rows.iterrows():
            if family in families_chrom_range:
                x_start, x_end = families_chrom_range[family]
                x_values = np.linspace(x_start, x_end, 100)
            else:
                continue

            plt.figure(figsize=(10, 6))
            blue_lines = []

            for func_type in params_list.index:
                func_params = ast.literal_eval(params_list[func_type])
                func_params = [float(param) for param in func_params]

                func = func_map.get(func_type)
                if func is None:
                    continue

                y_values = func(func_params, x_values)
                color = "blue"
                label = func_type

                if func_type == chosen_models_df.loc[family, "Chosen_Model"]:
                    continue

                blue_lines.append((x_values, y_values, color, label))

            for x_vals, y_vals, color, label in blue_lines:
                plt.plot(x_vals, y_vals, color=color, label=label)

            for func_type in params_list.index:
                if func_type == chosen_models_df.loc[family, "Chosen_Model"]:
                    func_params = ast.literal_eval(params_list[func_type])
                    func_params = [float(param) for param in func_params]
                    func = func_map.get(func_type)
                    if func is None:
                        continue

                    y_values = func(func_params, x_values)
                    color = "red"
                    label = f"Best model: {func_type}"
                    plt.plot(x_values, y_values, color=color, label=label)
                    break

            plt.title(f"Rate Functions for {family} - Transition: {transition}")
            plt.xlabel("Number of Chromosomes")
            plt.ylabel("Rate Function Value")
            plt.legend(handles=[plt.Line2D([0], [0], color="red", lw=2, label=f"Best model: {chosen_models_df.loc[family, 'Chosen_Model']}")], fontsize="small")

            plt.xlim(left=0)
            plt.ylim(bottom=0)

            output_path = f"{each_family_graphs_folder}/{transition}/{transition}_{family}.png"
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

def create_all_func_one_graph_for_each_family_rainbow(
    analysis_folder: str,
    each_family_graphs_folder: str,
    transitions: list[str],
    families_chrom_range: Dict[str, Tuple[int, int]]
) -> None:
    func_map = {
        "linear": Linear,
        "linear-bd": Linear_BD,
        "exponential": Exp,
        "reverse-sigmoid": Rev_Sigmoid,
        "log-normal": Log_normal,
        "constant": Const,
        "ignore": None
    }

    color_gradient = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

    for transition in transitions:
        raw_results_file = f"{analysis_folder}{transition}_raw_results.csv"
        raw_results_df = pd.read_csv(raw_results_file, index_col=0)
        params_values_rows = raw_results_df.loc[raw_results_df.index.str.endswith("_param")]
        params_values_rows.index = params_values_rows.index.str.replace("_param", "")

        for index, row_values in params_values_rows.iterrows():
            family_name = index
            params_list = row_values
            AICc_values = raw_results_df.loc[f"{family_name}_AICc"]

            sorted_func_types = AICc_values.sort_values().index
            color_map = {func_type: color_gradient[i % len(color_gradient)]
                         for i, func_type in enumerate(sorted_func_types)}

            x_start, x_end = families_chrom_range[family_name]
            x_values = np.linspace(x_start, x_end, 100)

            plt.figure(figsize=(10, 6))
            handles_labels = []

            for func_type in sorted_func_types:
                func_params = ast.literal_eval(params_list.get(func_type, "[]"))
                func_params = [float(param) for param in func_params]

                func = func_map.get(func_type)
                color = color_map.get(func_type, "gray")
                label = f"{func_type} (AICc: {float(AICc_values[func_type]):.2f})"

                if func is None:
                    handles_labels.append((plt.Line2D([], [], color=color, linewidth=2), label))
                    continue

                y_values = func(func_params, x_values)
                line, = plt.plot(x_values, y_values, color=color, linewidth=2)
                handles_labels.append((line, label))

            handles_labels = sorted(handles_labels, key=lambda hl: list(sorted_func_types).index(hl[1].split(" ")[0]))
            handles, labels = zip(*handles_labels)

            plt.title(f"Rate Functions for {family_name} - Transition: {transition}")
            plt.xlabel("Number of Chromosomes")
            plt.ylabel("Rate Function Value")
            plt.legend(handles, labels, fontsize="small", loc="upper left")

            plt.xlim(left=0)
            plt.ylim(bottom=0)

            output_path = f"{each_family_graphs_folder}/{transition}/{transition}_{family_name}_rainbow.png"
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate rate function plots for chromosome evolution.")
    parser.add_argument("function", choices=["chosen", "family", "family_rainbow"],
        help="Choose the function to use: 'chosen' for create_all_chosen_func_one_graph, "
             "'family' for create_all_func_one_graph_for_each_family, "
             "'family_rainbow' for create_all_func_one_graph_for_each_family_rainbow."
    )
    parser.add_argument("analysis_folder", type=str, help="Path to the folder containing analysis files.")
    parser.add_argument("transitions", type=str, nargs='+', help="List of transitions.")
    parser.add_argument("families_chrom_range", type=str,help="Family chromosome ranges as a dictionary string, e.g. \"{'family1': (10, 100), 'family2': (20, 120)}\"")
    parser.add_argument("--each_family_graphs_folder", type=str,help="Path to the folder where the output graphs for each family will be saved (required for 'family' and 'family_rainbow').")
    parser.add_argument("--save_fig_folder", type=str,help="Path to the folder where the output figure for 'chosen' function will be saved (required for 'chosen').")

    args = parser.parse_args()

    families_chrom_range = ast.literal_eval(args.families_chrom_range)

    if args.function == "chosen":
        if not args.save_fig_folder:
            parser.error("The '--save_fig_folder' argument is required for 'chosen' function.")
        create_all_chosen_func_one_graph(
            args.analysis_folder,
            args.save_fig_folder,
            args.transitions,
            families_chrom_range
        )
    elif args.function == "family":
        if not args.each_family_graphs_folder:
            parser.error("The '--each_family_graphs_folder' argument is required for 'family' function.")
        create_all_func_one_graph_for_each_family(
            args.analysis_folder,
            args.each_family_graphs_folder,
            args.transitions,
            families_chrom_range
        )
    elif args.function == "family_rainbow":
        if not args.each_family_graphs_folder:
            parser.error("The '--each_family_graphs_folder' argument is required for 'family_rainbow' function.")
        create_all_func_one_graph_for_each_family_rainbow(
            args.analysis_folder,
            args.each_family_graphs_folder,
            args.transitions,
            families_chrom_range
        )


if __name__ == "__main__":
    main()

