
def create_job_format_for_all_jobs(path: str, job_name: str, memory: str, queue: str, ncpu: int, cmd: str,
                                   slurm: int) -> str:
    text = ""
    text += f"#!/bin/bash\n\n"
    if not slurm:
        text += f"#PBS -S /bin/bash\n"
        text += f"#PBS -r y\n"
        text += f"#PBS -q {queue}\n"
        text += f"#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH\n"
        text += f"#PBS -N {job_name}\n"
        text += f"#PBS -e {path}/{job_name}.ER\n"
        text += f"#PBS -o {path}/{job_name}.OU\n"
        text += f"#PBS -l select=ncpus={ncpu}:mem={memory}\n"
        text += f"cd {path}\n"
        text += f"{cmd}\n"
    else:
        text += f"#SBATCH --job-name={job_name}\n"
        text += f"#SBATCH --account=itaym-users\n"
        text += f"#SBATCH --partition=itaym\n"
        text += f"#SBATCH --ntasks={ncpu}\n"
        text += f"#SBATCH --cpus-per-task=1\n"
        text += f"#SBATCH --time=7-00:00:00\n"
        if ncpu > 1:
            text += f"#SBATCH --mem-per-cpu=3G\n" # for example 4G
        else:
            text += f"#SBATCH --mem-per-cpu={memory}\n"
        text += f"#SBATCH --output={path}/{job_name}_out.OU\n"
        text += f"#SBATCH --error={path}/{job_name}_err.ER\n"
        text += f"cd {path}\n"
        text += f"{cmd}\n"

    return text