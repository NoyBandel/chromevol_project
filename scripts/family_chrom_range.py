from pathlib import Path
import pandas as pd


def family_chrom_range(counts_file_dir: str) -> list[int]:
    min_chrom = float('inf')
    max_chrom = float('-inf')
    with open(counts_file_dir, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                chrom_num = int(line)
                if chrom_num < min_chrom:
                    min_chrom = chrom_num
                if chrom_num > max_chrom:
                    max_chrom = chrom_num
    if min_chrom == float('inf') or max_chrom == float('-inf'):
        return [0, 0, 0]
    diff = max_chrom - min_chrom
    return [min_chrom, max_chrom, diff]

def add_range_to_linear_analysis(families_chrom_count_folder: Path, linear_analysis_folder: Path) -> None:
    family_to_range_dict = {}
    for family_folder in families_chrom_count_folder.iterdir():
        family_name = family_folder.name
        counts_file = str(families_chrom_count_folder / family_folder / "counts.fasta")
        family_to_range_dict[family_name] = family_chrom_range(counts_file)

    analysis_files = list(linear_analysis_folder.glob("*_vs_const_linear_parameters.csv"))
    for file in analysis_files:
        df = pd.read_csv(file, index_col=0)
        dict_df = pd.DataFrame.from_dict(family_to_range_dict, orient='index', columns=['min_chrom', 'max_chrom', 'range_size'])
        df = df.merge(dict_df, left_index=True, right_index=True)
        df.to_csv(f"{str(file)}", index=True)

# add_range_to_linear_analysis(Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/all_families_over_100/"), Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/lin_exp_analysis/linear_analysis/"))


def average_species_num(input_folder: Path):
    species_num_list = [
        sum(1 for line in open(family_folder / "counts.fasta") if line.startswith('>'))
        for family_folder in input_folder.iterdir() if (family_folder / "counts.fasta").exists()
    ]
    average = sum(species_num_list) / len(species_num_list)
    print(species_num_list)
    print(average)

# average_species_num(Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/all_families_over_100/"))

def chrom_range_to_csv(families_chrom_count_folder: Path, family_data_csv: Path) -> None:
    family_data_df = pd.read_csv(family_data_csv, index_col=0)
    for family_folder in families_chrom_count_folder.iterdir():
        if family_folder.is_dir():
            file_path = family_folder / "counts.fasta"
            if file_path.is_file():
                print(f"Processing: {file_path}")
                family_name = family_folder.name
                counts_file = str(file_path)
                min_chrom, max_chrom, diff = family_chrom_range(counts_file)
                family_data_df.loc[family_name, "min_chrom"] = min_chrom
                family_data_df.loc[family_name, "max_chrom"] = max_chrom
                family_data_df.loc[family_name, "diff"] = diff
    family_data_df.to_csv("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data_with_chrom.csv")



# families_chrom_count_folder = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_chrom_input/")
# family_data_csv = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data.csv")
#
# chrom_range_to_csv(families_chrom_count_folder, family_data_csv)














