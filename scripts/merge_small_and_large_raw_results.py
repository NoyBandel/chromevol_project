from constants import *
import pandas as pd
import os

# Define paths
FORMER_OUTPUT_FOLDER = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/raw_results/former_raw_results/"
MODIFIED_OUTPUT_FOLDER = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/all_families_over_50_modified_chosen/analysis_from_const_except_for_tested/raw_results/modified_raw_results/"
SMALL_FAMILIES_RAW_RESULTS_CSVs_FOLDER = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis_50_to_99/const_except_for_tested/"
LARGE_FAMILIES_RAW_RESULTS_CSVs_FOLDER = "/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_analysis/const_except_for_tested/"

# Columns to remove for the modified output
COLUMNS_TO_REMOVE = [LABEL_LINEAR_BD, LABEL_REVERSE_SIGMOID]

# Input transitions and file suffix
label_transitions_lst = LABEL_TRANSITIONS_LST
raw_results_suffix = RAW_RESULTS

# Ensure output directories exist
os.makedirs(FORMER_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODIFIED_OUTPUT_FOLDER, exist_ok=True)

# Process each transition
for transition in label_transitions_lst:
    small_path = os.path.join(SMALL_FAMILIES_RAW_RESULTS_CSVs_FOLDER, f"{transition}_{raw_results_suffix}")
    large_path = os.path.join(LARGE_FAMILIES_RAW_RESULTS_CSVs_FOLDER, f"{transition}_{raw_results_suffix}")

    try:
        df_small = pd.read_csv(small_path)
        df_large = pd.read_csv(large_path)
    except Exception as e:
        print(f"Failed to read files for {transition}: {e}")
        continue

    # Align columns by name and order
    common_cols = sorted(set(df_small.columns).union(df_large.columns))
    df_small = df_small.reindex(columns=common_cols)
    df_large = df_large.reindex(columns=common_cols)

    # Concatenate aligned DataFrames
    combined_df = pd.concat([df_small, df_large], ignore_index=True)

    # Save full (former) version
    output_path_former = os.path.join(FORMER_OUTPUT_FOLDER, f"{transition}_{raw_results_suffix}")
    combined_df.to_csv(output_path_former, index=False)

    # Save modified version with selected columns removed
    filtered_df = combined_df.drop(columns=[col for col in COLUMNS_TO_REMOVE if col in combined_df.columns])
    output_path_modified = os.path.join(MODIFIED_OUTPUT_FOLDER, f"{transition}_all_families_over_50_modified_{raw_results_suffix}")
    filtered_df.to_csv(output_path_modified, index=False)

    print(f"✅ Processed: {transition}")

