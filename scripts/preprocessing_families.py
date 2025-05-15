from pathlib import Path
import heapq
import random
import shutil
import pandas as pd

FAMILY_SIZE_LOWER_BOUND = 100
SOURCE_DIR = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_chrom_input/")
UNDER_K_DIR = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_under_k/all_families_under_k/")
OVER_K_DIR = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_over_k/")
ANALYZE_DIR = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_under_k/families_under_k_over_" + str(FAMILY_SIZE_LOWER_BOUND) + "/")
TEST_DIR = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/families_under_k/10_families_for_test/")
FILE_NAME = "counts.fasta"
FAMILIES_AND_SIZE = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_input_data/family_data.csv"
OVER_K_COL = "over_k"
UNDER_K_COL = "under_k"
ANALYZE_COL = "families_under_k_over_" + str(FAMILY_SIZE_LOWER_BOUND)
TEST_COL = "10_for_test"

over_k = []
under_k = []
under_k_over_bound = []
top_100_heap = []

data = {
    "family_name": [],
    "family_size": [],
    OVER_K_COL: [],
    UNDER_K_COL: [],
    ANALYZE_COL: [],
    TEST_COL: []
}

for folder in SOURCE_DIR.rglob('*'):
    if folder.is_dir():
        file_path = folder / FILE_NAME
        if file_path.is_file():
            with file_path.open('r') as file:
                tree_size = sum(1 for line in file if line.startswith('>'))

            family_name = folder.name
            data["family_name"].append(family_name)
            data["family_size"].append(tree_size)

            data[OVER_K_COL].append(0)
            data[UNDER_K_COL].append(0)
            data[ANALYZE_COL].append(0)
            data[TEST_COL].append(0)

            if tree_size >= 1000:
                over_k.append((tree_size, folder))
                data[OVER_K_COL][-1] = 1
            else:
                under_k.append((tree_size, folder))
                data[UNDER_K_COL][-1] = 1

                if tree_size > FAMILY_SIZE_LOWER_BOUND:
                    under_k_over_bound.append((tree_size, folder))
                    data[ANALYZE_COL][-1] = 1
                    if len(top_100_heap) < 100:
                        heapq.heappush(top_100_heap, (tree_size, folder))
                    else:
                        heapq.heappushpop(top_100_heap, (tree_size, folder))
                else:
                    data[ANALYZE_COL][-1] = 0

            data[TEST_COL][-1] = 0

random_selection = random.sample(top_100_heap, min(10, len(top_100_heap)))

folder_dicts = {
    OVER_K_COL: (over_k, OVER_K_DIR),
    UNDER_K_COL: (under_k, UNDER_K_DIR),
    ANALYZE_COL: (under_k_over_bound, ANALYZE_DIR),
    TEST_COL: (random_selection, TEST_DIR)
}

def copy_folders_to_new_dir(folder_list, new_dir, dir_column):
    for tree_size, folder in folder_list:
        family_name = folder.name
        data[dir_column][data["family_name"].index(family_name)] = 1
        relative_path = folder.relative_to(SOURCE_DIR)
        target_folder = new_dir / relative_path
        shutil.copytree(folder, target_folder, dirs_exist_ok=True)

for dir_column, (folder_list, path) in folder_dicts.items():
    copy_folders_to_new_dir(folder_list, path, dir_column)

df = pd.DataFrame(data)
df.to_csv(FAMILIES_AND_SIZE, index=False)
