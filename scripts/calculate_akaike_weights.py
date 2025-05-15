from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

color_map = {
    'linear': [0.65098039, 0.80784314, 0.89019608, 1.0],
    'linear-bd': [0.69803922, 0.8745098, 0.54117647, 1.0],
    'exponential': [0.98431373, 0.60392157, 0.6, 1.0],
    'reverse-sigmoid': [0.99215686, 0.74901961, 0.43529412, 1.0],
    'log-normal': [0.79215686, 0.69803922, 0.83921569, 1.0],
    'ignore': [1.0, 1.0, 0.6, 1.0],
    'constant': [0.69411765, 0.34901961, 0.15686275, 1.0],
}


def update_color_map(exclude_func):
    updated_color_map = {key: value for key, value in color_map.items() if key not in exclude_func}
    return updated_color_map


def calculation(aic_score_list: list) -> list:
    aic_scores = np.array(aic_score_list)
    delta_aic = aic_scores - np.min(aic_scores)
    weights = np.exp(-0.5 * delta_aic)
    weights /= np.sum(weights)
    return weights


def calculate_akaike_weights(raw_results_folder: Path, exclude_func: list) -> None:
    raw_results_files = list(raw_results_folder.glob("*_raw_results.csv"))
    for transition_res_file in raw_results_files:
        transition_name = str(transition_res_file.name).replace("_raw_results.csv", "")
        df = pd.read_csv(str(transition_res_file), index_col=0)
        AIC_values_rows = df.loc[df.index.str.endswith("AICc")]
        columns_to_include = [col for col in df.columns if col not in exclude_func]
        AIC_values_rows = AIC_values_rows[columns_to_include]
        data = {}
        for family, AIC_values in AIC_values_rows.iterrows():
            family_name = family.replace("_AICc", "")
            aicc_values = pd.to_numeric(AIC_values).tolist()
            weights = calculation(aicc_values)
            data[family_name] = dict(zip(AIC_values.index, weights))
        weights_df = pd.DataFrame.from_dict(data, orient='index')
        weights_df.to_csv(f"{raw_results_folder}/{transition_name}_akaike_weights_exclude_{exclude_func[0]}.csv")


def akaike_weight_pie_chart(raw_results_folder: Path, save_fig_folder: str, exclude_func: list) -> None:
    updated_color_map = update_color_map(exclude_func)

    raw_results_files = list(raw_results_folder.glob(f"*_akaike_weights_exclude_{exclude_func[0]}.csv"))
    for transition_res_file in raw_results_files:
        transition_name = str(transition_res_file.name).replace(f"_akaike_weights_exclude_{exclude_func[0]}.csv", "")
        df = pd.read_csv(str(transition_res_file), index_col=0)
        data = {}
        for func_column in df.columns:
            column_sum = df[func_column].sum()
            tot_func_weight = column_sum / len(df)
            data[func_column] = tot_func_weight

        y = np.array(list(data.values()))
        labels = list(data.keys())

        colors = [updated_color_map[label] for label in labels if label in updated_color_map]

        plt.figure(figsize=(8, 8))

        wedges, texts, autotexts = plt.pie(y, labels=labels, autopct='%1.1f%%', startangle=90,
                                           wedgeprops={'edgecolor': 'black'}, pctdistance=0.85, colors=colors)

        for text in texts:
            text.set(size=10, fontweight='bold')

        n_samples = len(df)
        plt.text(0, -1.2, f'n = {n_samples}', fontsize=12, color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.title(f"Akaike Weights Pie Chart - {transition_name}")
        plt.savefig(f"{save_fig_folder}/{transition_name}_akaike_weights_pie_exclude_{exclude_func[0]}.png",
                    bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate Akaike weights and generate pie charts.")

    parser.add_argument("raw_results_folder", type=Path, help="Path to the folder containing the raw results CSV files")
    parser.add_argument("save_fig_folder", type=Path, help="Path to the folder to save the pie charts")
    parser.add_argument("exclude_func", type=str, nargs='*', default=[""], help="List of functions to exclude")

    args = parser.parse_args()

    calculate_akaike_weights(args.raw_results_folder, args.exclude_func)
    akaike_weight_pie_chart(args.raw_results_folder, args.save_fig_folder, args.exclude_func)


if __name__ == "__main__":
    main()
