import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def analyze_data_separately(input_file_path: str, transition_name: str, linear_folder_path: str, exp_folder_path: str) -> None:
    data = pd.read_csv(input_file_path)
    lin_rate: list[float] = data['lin_rate'].tolist()
    exp_rate: list[float] = data['exp_rate'].tolist()
    lin_class: list[int] = data['lin_class'].tolist()
    exp_class: list[int] = data['exp_class'].tolist()

    # def count_binary(data: list[int]) -> tuple[dict[int, int], dict[int, float]]:
    #     counts = dict(Counter(data))
    #     total = len(data)
    #     proportions = {key: value / total for key, value in counts.items()}
    #     return counts, proportions
    #
    # def describe_data(data: list[float]) -> dict[str, float]:
    #     stats = {
    #         "Mean": np.mean(data),
    #         "Median": np.median(data),
    #         "Std Dev": np.std(data),
    #         "Range": np.max(data) - np.min(data),
    #         "Variance": np.var(data),
    #     }
    #     return stats
    #
    # def format_binary_info(data: list[int]) -> str:
    #     counts, proportions = count_binary(data)
    #     return ", ".join([f"{key}: {value} ({proportions[key]:.2f})" for key, value in counts.items()])

    def plot_histogram(data: list[float], title: str, filename: str, folder: Path) -> None:
        plt.hist(data, bins=10, alpha=0.7)
        plt.title(title)
        plt.xlabel('Transition Rate')
        plt.ylabel('Frequency')
        plt.xlim(left=-max(abs(np.min(data)), abs(np.max(data))),
                 right=max(abs(np.min(data)), abs(np.max(data))))
        plt.savefig(folder / filename)
        plt.close()

    # def plot_boxplot(data: list[float], title: str, filename: str, folder: Path) -> None:
    #     plt.boxplot(data)
    #     plt.title(title)
    #     plt.ylabel('Value')
    #     plt.savefig(folder / filename)
    #     plt.close()
    #
    # def plot_violin(data: list[float], title: str, filename: str, folder: Path) -> None:
    #     sns.violinplot(data=data)
    #     plt.title(title)
    #     plt.ylabel('Value')
    #     plt.savefig(folder / filename)
    #     plt.close()

    linear_folder = Path(linear_folder_path)
    exp_folder = Path(exp_folder_path)

    # lin_rate_stats: dict[str, float] = describe_data(lin_rate)
    # lin_rate_stats["Binary Counts and Proportions"] = format_binary_info(lin_class)
    #
    # lin_df = pd.DataFrame([lin_rate_stats], index=["linear_stats"])
    # lin_df.to_csv(linear_folder / f"{transition_name}_lin_stats_summary.csv", index=True)

    plot_histogram(lin_rate, f"Histogram of {transition_name} linear rate", f"{transition_name}_lin_rate_histogram.png", linear_folder)
    # plot_boxplot(lin_rate, f"Boxplot of {transition_name} linear rate", f"{transition_name}_lin_rate_boxplot.png", linear_folder)
    # plot_violin(lin_rate, f"Violin plot of {transition_name} linear rate", f"{transition_name}_lin_rate_violin.png", linear_folder)
    #
    # exp_rate_stats: dict[str, float] = describe_data(exp_rate)
    # exp_rate_stats["Binary Counts and Proportions"] = format_binary_info(exp_class)
    #
    # exp_df = pd.DataFrame([exp_rate_stats], index=["exp_stats"])
    # exp_df.to_csv(exp_folder / f"{transition_name}_exp_stats_summary.csv", index=True)

    plot_histogram(exp_rate, f"Histogram of {transition_name} exponential rate", f"{transition_name}_exp_rate_histogram.png", exp_folder)
    # plot_boxplot(exp_rate, f"Boxplot of {transition_name} exponential rate", f"{transition_name}_exp_rate_boxplot.png", exp_folder)
    # plot_violin(exp_rate, f"Violin plot of {transition_name} exponential rate", f"{transition_name}_exp_rate_violin.png", exp_folder)


def main():
    parser = argparse.ArgumentParser(description="Analyze transition data and generate plots.")
    parser.add_argument("input_file_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("transition_name", type=str, help="The name of the transition.")
    parser.add_argument("linear_folder_path", type=str,
                        help="Path to the folder where linear analysis results will be saved.")
    parser.add_argument("exp_folder_path", type=str,
                        help="Path to the folder where exponential analysis results will be saved.")

    args = parser.parse_args()

    analyze_data_separately(args.input_file_path, args.transition_name, args.linear_folder_path,
                            args.exp_folder_path)

if __name__ == "__main__":
    main()


    # from scipy.stats import chi2_contingency
    # from sklearn.metrics import cohen_kappa_score
    # import numpy as np
    #
    # import pandas as pd

    # for each analysis - perform analysis for numerical data and for binary data:
    # 1. Separate Analysis
    # Analysis of linear data
    # Analysis of exp data

    # 2. Comparison Between Exp and Lin Data

    # 3. Combined Lin-Exp Data

    # Graphic Representation:
    # Visual Representation of Agreement - Heatmap for Agreement Across Variables

    # Visualizing the Agreement - You can use bar plots or scatter plots to visualize the agreement for each variable

    # Statistics:

    # Chi-Square Test for Independence

    # Correlation: Pearson correlation

    # Mean Absolute Error (MAE): To see the average absolute difference

    # Fisher's Exact Test

    # Cohen's Kappa: inter-rater agreement

    # Proportion of Agreement

    # McNemar’s Test

    # from sklearn.metrics import cohen_kappa_score, confusion_matrix
    # from scipy.stats import pearsonr
    # # 2. Comparison Between lin and exp data
    # corr, _ = pearsonr(lin_rate, exp_rate)
    # agreement = np.mean(np.array(lin_class) == np.array(exp_class))
    # kappa = cohen_kappa_score(lin_class, exp_class)
    # < 0.2 → Poor
    # agreement.
    # 0.2–0.4 → Fair
    # agreement.
    # 0.4–0.6 → Moderate
    # agreement.
    # 0.6–0.8 → Substantial
    # agreement.
    # 0.8–1.0 → Almost
    # perfect
    # agreemen
    #
    #
    #
    # plt.scatter(lin_rate, exp_rate)
    # plt.title("Scatter plot of lin_rate vs exp_rate")
    # plt.xlabel("lin_rate")
    # plt.ylabel("exp_rate")
    # plt.savefig("lin_vs_exp_scatter.png")
    # plt.close()
    #
    #
    #
    # # 3. Combined lin-exp Data
    # mae = np.mean(np.abs(np.array(lin_rate) - np.array(exp_rate)))
    # combined_agreement = np.mean(np.array(lin_class) == np.array(exp_class))
    # cm_combined = confusion_matrix(lin_class, exp_class)
    #
    # with open('combined_analysis.txt', 'w') as f:
    #     f.write(f"Mean Absolute Error (MAE) between lin_rate and exp_rate: {mae:.2f}\n")
    #     f.write(f"Combined binary agreement: {combined_agreement:.2f}\n")
    #     f.write(f"Confusion Matrix for combined lin-exp data:\n{cm_combined}\n")
    #
    # sns.heatmap(cm_combined, annot=True, fmt="d", cmap="Blues", xticklabels=['Exp Class 0', 'Exp Class 1'],
    #             yticklabels=['Lin Class 0', 'Lin Class 1'])
    # plt.title("Confusion Matrix for Combined lin-exp Data")
    # plt.savefig("combined_confusion_matrix.png")
    # plt.close()





