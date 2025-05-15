from pathlib import Path
import argparse
import pandas as pd

def create_lin_exp_combined_data(analysis_path: Path, lin_exp_analysis_path: str, transitions_prefix: list[str]) -> None:
    for transition in transitions_prefix:
        linear_df = pd.read_csv(analysis_path / "linear_analysis" / f'{transition}_linear_parameters.csv', index_col=0)
        exponential_df = pd.read_csv(
        analysis_path / "exponential_analysis" / f'{transition}_exponential_parameters.csv', index_col=0)
        merged_df = pd.DataFrame({
            'lin_rate': linear_df['p1_slope'],
            'lin_class': linear_df['behavior_class'],
            'exp_rate': exponential_df['p1_slope'],
            'exp_class': exponential_df['behavior_class']
        })
        merged_df.index = linear_df.index
        merged_df['accordance'] = (merged_df['lin_class'] == merged_df['exp_class']).astype(int)
        merged_df.to_csv(f'{lin_exp_analysis_path}_{transition}_lin_exp_data.csv', index=True)


def main():
    parser = argparse.ArgumentParser(description="Combine linear and exponential analysis results.")
    parser.add_argument("analysis_path", type=str, help="Path to the analysis root directory.")
    parser.add_argument("lin_exp_analysis_path", type=str, help="Path to save combined results.")
    parser.add_argument("transitions_prefix", nargs='+', type=str, help="List of transition prefixes.")

    args = parser.parse_args()
    analysis_path = Path(args.analysis_path)
    lin_exp_analysis_path = args.lin_exp_analysis_path
    transitions_prefix = args.transitions_prefix

    create_lin_exp_combined_data(analysis_path, lin_exp_analysis_path, transitions_prefix)

if __name__ == "__main__":
    main()


# transitions_prefix = ["baseNum", "demi", "dupl", "gain", "loss"]