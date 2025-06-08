
### # cheomevol  # ###
# cheomevol functions names
CE_IGNORE = "IGNORE"
CE_CONSTANT = "CONST"
CE_LINEAR = "LINEAR"
CE_LINEAR_BD = "LINEAR_BD"
CE_EXP = "EXP"
CE_LOGNORMAL = "LOGNORMAL"
CE_REVERSE_SIGMOID = "REVERSE_SIGMOID"

# cheomevol transitions names
CE_GAIN_FUNC = "_gainFunc"
CE_LOSS_FUNC = "_lossFunc"
CE_DUPL_FUNC = "_duplFunc"
CE_DEMI_DUPL_FUNC = "_demiDuplFunc"
CE_BASE_NUM_FUNC = "_baseNumRFunc"

CE_GAIN_INIT = "_gain_1"
CE_LOSS_INIT = "_loss_1"
CE_DUPL_INIT = "_dupl_1"
CE_DEMI_INIT = "_demiPloidyR_1"
CE_BASE_NUM_R_INIT = "_baseNumR_1"
CE_BASE_CHROM_NUM_INIT = "_baseNum_1"


### # analysis # ###
# analysis functions names - for folders and csv files
LABEL_IGNORE = "ignore"
LABEL_CONSTANT = "constant" #"const" for previous folders
LABEL_LINEAR = "linear"
LABEL_LINEAR_BD = "linear-bd"
LABEL_EXP = "exponential"
LABEL_LOGNORMAL = "log-normal"
LABEL_REVERSE_SIGMOID = "reverse-sigmoid"

# analysis transitions names - for folders and csv files
LABEL_BASE_NUM = "baseNum"
LABEL_DEMI = "demi" #"demiDupl" for previous folders
LABEL_DUPL = "dupl"
LABEL_GAIN = "gain"
LABEL_LOSS = "loss"
LABEL_TRANSITIONS_LST = [
    LABEL_BASE_NUM,
    LABEL_DEMI,
    LABEL_DUPL,
    LABEL_GAIN,
    LABEL_LOSS
]

# family_data_with_chrom: chrom count and family size columns names
LABEL_MIN_CHROM = "min_chrom"
LABEL_MAX_CHROM = "max_chrom"
LABEL_CHROM_DIFF = "diff"
LABEL_FAMILY_SIZE = "family_size"

# all_chosen_models.csv contains columns of <transition type>_chosen_model and corresponding <transition type>_parameters
LABEL_CHOSEN_MODEL = "chosen_model"
LABEL_PARAMS = "parameters"

# for the raw results files
LABEL_PARAMS_SHORT = "param"


LABEL_AICc = "AICc"

# graph utils
# f"{save_fig_folder}{transition}_all_chosen_rate_functions_plot.png"
ALL_CHOSEN_GRAPH_FOLDER = "all_chosen_models"
ALL_CHOSEN_GRAPH_SUFFIX = "all_chosen_rate_functions.png"
ZOOMED_SUFFIX = "zoomed_annotated.png"




# <transition>_from_const_run_to_modified_chosen_model.csv
TRANSITION_MODIFIED_CHOSEN_SUFFIX = "from_const_run_to_modified_chosen_model.csv"

# <transition>_raw_results.csv
# rows: <family name>_ param, AICc, likelihood, consts_params
RAW_RESULTS = "raw_results.csv"

LABEL_PARAM = "param"

EACH_FAMILY = "each_family"




### # CE <--> LABEL # ##
FUNC_CE_TO_LABEL = {
    CE_IGNORE: LABEL_IGNORE,
    CE_CONSTANT: LABEL_CONSTANT,
    CE_LINEAR: LABEL_LINEAR,
    CE_LINEAR_BD: LABEL_LINEAR_BD,
    CE_EXP: LABEL_EXP,
    CE_LOGNORMAL: LABEL_LOGNORMAL,
    CE_REVERSE_SIGMOID: LABEL_REVERSE_SIGMOID
}

FUNC_LABEL_TO_CE = {v: k for k, v in FUNC_CE_TO_LABEL.items()}









