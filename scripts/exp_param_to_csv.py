import ast
import pandas as pd
import argparse

def exp_param_to_csv(csv_data_folder: str, analysis_output_folder: str, transitions_res_file_name: list[str]) -> None:
    for transition_res_file_name in transitions_res_file_name:
        p0_scales: dict[str:str] = {}
        p1_rates: dict[str:str] = {}
        behavior_class: dict[str:str] = {}
        
        transition_res_path: str = f"{csv_data_folder}/{transition_res_file_name}"
        df = pd.read_csv(transition_res_path, index_col=0)
        filtered_values = df.loc[df.index.str.endswith("_param"), "exponential"]
        filtered_df = pd.DataFrame(filtered_values)
        for family in filtered_df.itertuples():
            family_name = str(str(family.Index).replace("_param", ""))
            exp_parameters_str = family.exponential
            exp_parameters = ast.literal_eval(exp_parameters_str)
            p0_scale = str(exp_parameters[0])
            p1_rate = str(exp_parameters[1])
            p0_scales[family_name] = p0_scale
            p1_rates[family_name] = p1_rate
            behavior_class[family_name] = str(int(float(p1_rate) > 0))
        
       
        ascending_percentage = (list(behavior_class.values()).count("1") / len(behavior_class)) * 100
        descending_percentage = 100 - ascending_percentage
        behavior_distribution = {
            "ascending_percentage": ascending_percentage,
            "descending_percentage": descending_percentage
        }
        behavior_distribution_df = pd.DataFrame([behavior_distribution])
        behavior_distribution_df.to_csv(f"{analysis_output_folder}/{transition_res_file_name.replace('_raw_results.csv', '')}_behavior_distribution.csv", index=False)
        
        data = {
        "p0_intersection": p0_scales,
        "p1_slope": p1_rates,
        "behavior_class": behavior_class
        }
        output_df = pd.DataFrame(data)
        output_df.to_csv(f"{analysis_output_folder}/{transition_res_file_name.replace("_raw_results.csv", "")}_exponential_parameters.csv", index=True)
        
def main():
    parser = argparse.ArgumentParser(description="?.")
    parser.add_argument("csv_data_folder", type=str, help="Directory containing csv result file.")
    parser.add_argument("analysis_output_folder", type=str, help="Directory of the analysis folder.")
    parser.add_argument("transitions_res_file_name", nargs='+', type=str, help="List of transitions result file names.")
    
    args = parser.parse_args()
    exp_param_to_csv(args.csv_data_folder, args.analysis_output_folder, args.transitions_res_file_name)

if __name__ == "__main__":
    main()

    
# transitions_res_file_name = ["baseNum_raw_results.csv", "demi_raw_results.csv", "dupl_raw_results.csv", "gain_raw_results.csv", "loss_raw_results.csv"]
   