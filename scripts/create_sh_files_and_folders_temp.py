from pathlib import Path
import argparse

def create_sh_slurm_files_and_folders(main_jobs_folder: Path, chosen_or_const: str, func_types: list[str], create_pfiles_and_run_jobs_python_script: str, input_data_folder_path: str, raw_results_folder_path: str, data_type_job_name: str, all_chosen_models_file_path: str="") -> None:
    def validate_choice(choice):
        valid_choices = ["chosen", "const"]
        if choice not in valid_choices:
            raise ValueError(f"Invalid choice: {choice}. Must be one of {valid_choices}.")

    def create_shared_command():
        if chosen_or_const == "chosen":
            return (
                "    --transitions_dir gain loss dupl demiDupl baseNum \\\n"
                "    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc \\\n"
                "    --function_types exponential linear log-normal reverse-sigmoid linear-bd ignore const \\\n"
                "    --functions_param EXP LINEAR LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE CONST \\\n"
                "    --transitions_abbr '{\"gain\": \"g\", \"loss\": \"l\", \"dupl\": \"dup\", \"demiDupl\": \"dem\", \"baseNum\": \"bN\"}' \\\n"
                "    --functions_abbr '{\"linear\": \"lin\", \"exponential\": \"exp\", \"log-normal\": \"logN\", \"reverse-sigmoid\": \"revS\", \"linear-bd\": \"linBD\", \"ignore\": \"ign\", \"const\": \"con\"}' \\\n"
                f"    --all_chosen_models_path {all_chosen_models_file_path} \\\n"
                "    --transitions_csv_to_param_dict '{\"gain\": \"_gainFunc\", \"loss\": \"_lossFunc\", \"dupl\": \"_duplFunc\", \"demi\": \"_demiDuplFunc\", \"baseNum\": \"_baseNumRFunc\"}' \\\n"
            )
        else:
            return (
                "    --transitions_dir gain loss dupl demiDupl baseNum allConst \\\n"
                "    --transitions_param _gainFunc _lossFunc _duplFunc _demiDuplFunc _baseNumRFunc allConst \\\n"
                "    --function_types linear exponential log-normal reverse-sigmoid linear-bd ignore \\\n"
                "    --functions_param LINEAR EXP LOGNORMAL REVERSE_SIGMOID LINEAR_BD IGNORE \\\n"
                "    --transitions_abbr '{\"gain\": \"g\", \"loss\": \"l\", \"dupl\": \"dup\", \"demiDupl\": \"dem\", \"baseNum\": \"bN\", \"allConst\": \"const\"}' \\\n"
                "    --functions_abbr '{\"linear\": \"lin\", \"exponential\": \"exp\", \"log-normal\": \"logN\", \"reverse-sigmoid\": \"revS\", \"linear-bd\": \"BD\", \"ignore\": \"ign\"}' \\\n"
            )

    validate_choice(chosen_or_const)

    chosen_or_const_str = f"{chosen_or_const}_except_for_tested"

    for func_type in func_types:
        job_file_name = f"main_job-{chosen_or_const_str}-{func_type}"
        function_sh_file_path = main_jobs_folder / job_file_name
        if function_sh_file_path.exists():
            print(f"Directory {function_sh_file_path} already exists. Skipping.")
            continue
        function_sh_file_path.mkdir(parents=True, exist_ok=True)
        if (chosen_or_const == "const") and (func_type == "const"):
            func_type = "allConst"
        job_path = function_sh_file_path / f"{job_file_name}.sh"
        job_name = f"{data_type_job_name}_{chosen_or_const_str}_{func_type}"
        job_content = create_sh_slurm_job_content(
            create_pfiles_and_run_jobs_python_script,
            chosen_or_const,
            input_data_folder_path,
            raw_results_folder_path,
            func_type,
            all_chosen_models_file_path,
            function_sh_file_path,
            job_name,
            create_shared_command()
        )
        with open(job_path, 'w') as file:
            file.write(job_content)

def create_sh_slurm_job_content(create_pfiles_and_run_jobs_python_script: str, chosen_or_const: str, input_data_folder_path: str, raw_results_folder_path: str, function_type_dir: str, all_chosen_models_file_path: str, main_job_folder_path: Path, job_name: str, shared_command: str) -> str:
    cmd = (
        f"python {create_pfiles_and_run_jobs_python_script} \\\n"
        f"    {input_data_folder_path} \\\n"
        f"    {raw_results_folder_path} \\\n"
        f"    --function_type_dir {function_type_dir} \\\n"
        f"{shared_command}"
        f"    > {main_job_folder_path}/log.txt \\\n"
        f"    2> {main_job_folder_path}/err.txt\n"
    )

    job_content = (
        f"#!/bin/bash\n\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --account=power-general-users\n"
        f"#SBATCH --partition=power-general\n"
        f"#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time=7-00:00:00\n"
        f"#SBATCH --mem-per-cpu=4G\n"
        f"#SBATCH --output={main_job_folder_path}/out.OU\n"
        f"#SBATCH --error={main_job_folder_path}/err.ER\n\n"
        f"source ~/.bashrc\n"
        f"hostname\n"
        f"conda activate chromevol\n"
        f"export PATH=$CONDA_PREFIX/bin:$PATH\n\n"
        f"cd {main_job_folder_path}\n"
        f"{cmd}\n\n"
    )
    return job_content

def main():
    parser = argparse.ArgumentParser(description="Create SLURM job scripts for ChromEvol project.")
    parser.add_argument("--main_jobs_folder", required=True, type=Path, help="Path to the main jobs folder.")
    parser.add_argument("--chosen_or_const", required=True, choices=["chosen", "const"], help="Specify if chosen or const models should be used.")
    parser.add_argument("--func_types", required=True, nargs='+', help="List of function types to include.")
    parser.add_argument("--create_pfiles_and_run_jobs_python_script", required=True, help="Path to the Python script for creating pfiles and running jobs.")
    parser.add_argument("--input_data_folder_path", required=True, help="Path to the input data folder.")
    parser.add_argument("--raw_results_folder_path", required=True, help="Path to the raw results folder.")
    parser.add_argument("--data_type_job_name", required=True, help="Name of the data type job.")
    parser.add_argument("--all_chosen_models_file_path", default="", help="Path to the file with all chosen models (optional).")

    args = parser.parse_args()

    create_sh_slurm_files_and_folders(
        main_jobs_folder=args.main_jobs_folder,
        chosen_or_const=args.chosen_or_const,
        func_types=args.func_types,
        create_pfiles_and_run_jobs_python_script=args.create_pfiles_and_run_jobs_python_script,
        input_data_folder_path=args.input_data_folder_path,
        raw_results_folder_path=args.raw_results_folder_path,
        data_type_job_name=args.data_type_job_name,
        all_chosen_models_file_path=args.all_chosen_models_file_path
    )

if __name__ == "__main__":
    main()
