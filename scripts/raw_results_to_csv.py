import pandas as pd
from pathlib import Path
import argparse
import copy
import ast

def raw_results_to_csv(raw_results_path: Path, analysis_output_folder: str) -> None:
    gain_df = pd.DataFrame()
    loss_df = pd.DataFrame()
    duple_df = pd.DataFrame()
    demi_df = pd.DataFrame()
    baseNumR_df = pd.DataFrame()

    transition_to_df_dict = {
        "loss": loss_df,
        "gain": gain_df,
        "dupl": duple_df,
        "demi": demi_df,
        "baseNum": baseNumR_df
    }

    for family_folder in raw_results_path.iterdir():
        family_name = family_folder.name
        for transition_folder in family_folder.iterdir():
            transition_dir_name = transition_folder.name
            if transition_dir_name == "allConst":
                AICc, likelihood, consts_params = allConst_raw_results_to_df(transition_folder)
                for transition_type, df in transition_to_df_dict.items():
                    df.at[f"{family_name}_param", "constant"] = f"['{consts_params[transition_type]}']"
                    df.at[f"{family_name}_AICc", "constant"] = AICc
                    df.at[f"{family_name}_likelihood", "constant"] = likelihood
                    curr_consts_params = {k: v for k, v in consts_params.items() if k != transition_type}
                    df.at[f"{family_name}_consts_params", "constant"] = str(curr_consts_params)
            else:
                if transition_dir_name == "demiDupl":
                    transition_dir_name = "demi"
                transition_raw_results = transition_raw_results_to_csv(transition_folder, family_name)
                transition_to_df_dict[transition_dir_name] = pd.concat([transition_to_df_dict[transition_dir_name], transition_raw_results], axis=0)
                
    for transition_type, df in transition_to_df_dict.items():
        df = df.groupby(df.index).first()
        df.to_csv(f"{analysis_output_folder}/{transition_type}_raw_results.csv", index=True)


def transition_raw_results_to_csv(fam_transition_results_path: Path, family_name: str) -> pd.DataFrame:
    param_dict = {
        "linear": [[None] * 2],
        "linear-bd": [[None]],
        "exponential": [[None] * 2],
        "reverse-sigmoid": [[None] * 3],
        "log-normal": [[None] * 3],
        "ignore": [[]]
    }
    AICc_dict = {
        "linear": [],
        "linear-bd": [],
        "exponential": [],
        "reverse-sigmoid": [],
        "log-normal": [],
        "ignore": []
    }
    likelihood_dict = copy.deepcopy(AICc_dict)
    consts_param_dict = copy.deepcopy(AICc_dict)
    consts_param = {
        "loss": "",
        "gain": "",
        "dupl": "",
        "demi": "",
        "baseNumR": "",
        "baseNum": ""
    }

    transition_dir_to_param_dict = {
    "loss": "loss",
    "gain": "gain",
    "dupl": "dupl",
    "demiDupl": "demi",
    "baseNum": "baseNumR"
}

    if fam_transition_results_path.is_dir():
        transition_folder = fam_transition_results_path
        transition_dir_name = transition_folder.name
        del consts_param[transition_dir_to_param_dict[transition_dir_name]]
        for key in consts_param_dict:
            consts_param_dict[key].append(copy.deepcopy(consts_param))
            
        for func_folder in transition_folder.iterdir():
            func_dir_name = func_folder.name
            results_file_path = func_folder / "Results/chromEvol.res"
            with open(str(results_file_path), 'r') as file:
                lines = file.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if "AICc of the best model" in line:
                        AICc = line.split("=")[-1].strip()
                        AICc_dict[func_dir_name].append(AICc)
                    elif "Final optimized likelihood" in line:
                        likelihood = line.split(":")[-1].strip()
                        likelihood_dict[func_dir_name].append(likelihood)
                        break
                    else:
                        if "Chromosome." in line:
                            for transition_type in transition_dir_to_param_dict.values():
                                if transition_type in line:
                                    param_index = line.strip("Chromosome." + transition_type)[0]
                                    param = line.split("=")[-1].strip()
                                    if transition_type == transition_dir_to_param_dict[transition_dir_name]:
                                        param_dict[func_dir_name][0][int(param_index)] = param
                                    else:
                                        consts_param_dict[func_dir_name][0][transition_type] = param
                                elif "baseNum_1" in line:
                                    param = line.split("=")[-1].strip()
                                    consts_param_dict[func_dir_name][0]["baseNum"] = param
    param_df = pd.DataFrame(param_dict, index=[f"{family_name}_param"])
    AICc_df = pd.DataFrame(AICc_dict, index=[f"{family_name}_AICc"])
    likelihood_df = pd.DataFrame(likelihood_dict, index=[f"{family_name}_likelihood"])
    consts_param_df = pd.DataFrame(consts_param_dict, index=[f"{family_name}_consts_params"])
    combined_df = pd.concat([param_df, AICc_df, likelihood_df, consts_param_df], axis=0)
    return combined_df

def allConst_raw_results_to_df(fam_allConst_results_path: Path) -> tuple[str, str, dict[str, str]]:
    AICc = ""
    likelihood = ""
    consts_params = {
        "loss": "",
        "gain": "",
        "dupl": "",
        "demi": "",
        "baseNumR": ""
    }

    if fam_allConst_results_path.is_dir():
        allConst_folder = fam_allConst_results_path
        results_file_path = allConst_folder / "Results/chromEvol.res"
        with open(str(results_file_path), 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                line = line.strip()
                if "AICc of the best model" in line:
                    AICc = line.split("=")[-1].strip()
                elif "Final optimized likelihood" in line:
                    likelihood = line.split(":")[-1].strip()
                    break
                elif "Chromosome." in line:
                    for transition_type in consts_params.keys():
                        if f"Chromosome.{transition_type}0_1" in line:
                            consts_params[transition_type] = line.split("=")[-1].strip()
                    if "baseNum_1" in line:
                        consts_params["baseNum"] = line.split("=")[-1].strip()
    return AICc, likelihood, consts_params


def correct_baseNum_constant(analysis_folder: str) -> None:
    dupl_raw_results_csv_path = f"{analysis_folder}dupl_raw_results.csv"
    baseNum_raw_results_csv_path = f"{analysis_folder}baseNum_raw_results.csv"
    dupl_df = pd.read_csv(dupl_raw_results_csv_path, index_col=0)
    baseNum_df = pd.read_csv(baseNum_raw_results_csv_path, index_col=0)
    dupl_df_filtered = dupl_df[dupl_df.index.str.endswith('_consts_params')][['constant']]
    for row_name, row in dupl_df_filtered.iterrows():
        family_name = row_name.replace('_consts_params', "")
        consts_params_dict = ast.literal_eval(row["constant"])
        baseNumR = consts_params_dict['baseNumR']
        baseNum_df.at[f"{family_name}_param", 'constant'] = f"[{baseNumR}]"
        consts_params_dict_output = {key: value for key, value in consts_params_dict.items() if key != 'baseNumR'}
        baseNum_df.at[f"{family_name}_consts_params", 'constant'] = str(consts_params_dict_output)
    baseNum_df.to_csv(f"{analysis_folder}baseNum_raw_results.csv")


def main():
    parser = argparse.ArgumentParser(description="Process chromEvol results.")
    parser.add_argument("function", choices=["raw_results_to_csv", "correct_baseNum_constant"], help="Function to execute.")
    parser.add_argument("--raw_results_path", type=Path, help="Directory for family result files.")
    parser.add_argument("--analysis_folder", type=str, help="Directory for analysis results.")
    args = parser.parse_args()
    if args.function == "raw_results_to_csv":
        if not args.raw_results_path or not args.analysis_folder:
            parser.error("'raw_results_to_csv' requires '--raw_results_path' and '--analysis_folder'.")
        raw_results_to_csv(args.raw_results_path, args.analysis_folder)
    elif args.function == "correct_baseNum_constant":
        if not args.analysis_folder:
            parser.error("'correct_baseNum_constant' requires '--analysis_folder'.")
        correct_baseNum_constant(args.analysis_folder)

if __name__ == "__main__":
    main()


### next round - insert correct_baseNum_constant ito the original run