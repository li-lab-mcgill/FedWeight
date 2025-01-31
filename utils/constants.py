# Data
DATASET_DIR_CONFIG_KEY = "dataset_dir_path"
DATASET_DIR_DEFAULT = "../data/"
EICU_PATH = "one_hot_age_gender_eicu_data.csv"

# FL
LR = 0.0001
WEIGHT_DECAY = 0.001
MIN_HOSPITAL_DEATH_COUNT = 150
TOTAL_FL_ROUND = 1501
LOCAL_FL_EPOCHS = 1
TEST_SIZE = 0.3
VAL_SIZE = 0.2
TOTAL_FEATURE = 1420
FL_HIDDEN_LAYER_UNITS = "256,128,64"
BIAS_INIT_PRIOR_PROB = None
BATCH_SIZE = 128

# MADE
MADE_EPOCHS = 50
MADE_HIDDEN_LAYER_UNITS = "500"
MADE_NUM_MASKS = 1
MADE_SAMPLES = 1
MADE_RESAMPLE_EVERY = 20

# Weight
REWEIGHT_LAMBDA = 1.0

# Task
VENTILATOR = "ventilator"  # Ventilator usage in hospitals
SEPSIS = "sepsis"  # Sepsis diagnosis in hospitals
DEATH = "death"  # Death prediction in hospitals
LENGTH = "length"  # Length of stay in hospitals
UNSUPERVISED = "unsupervised"  # Unsupervised
CARDIOVASCULAR = "cardiovascular" # Cardiovascular
REGION_VENTILATOR = "region_ventilator"  # Ventilator usage in regions
REGION_SEPSIS = "region_sepsis"  # Sepsis diagnosis in regions
REGION_DEATH = "region_death"  # Death prediction in regions
REGION_LENGTH = "region_length"  # Length of stay in regions
REGION_TASK_PREFIX = "region"
SIMULATION = "simulation"  # Simulation dataset
SIMULATION_BY_DIR = "simulation_by_dir"  # Simulation dataset load by directory
COLOR_MNIST = "color_mnist"  # Color MNIST
BINARIZED_MNIST = "binarized_mnist"  # Binarized MNIST

# Algorithm
WEIGHTED = "weighted"  # Weighted
UNWEIGHTED = "unweighted"  # Unweighted
BOTH = "both"  # Both, used by paired

# Stage
GRID_SEARCH = "grid_search"
RETRAIN = "retrain"

# Label index
LABEL_IDX = {
    VENTILATOR: 3,
    SEPSIS: 4,
    DEATH: 2,
    COLOR_MNIST: 0
}

# Log
LOG_PATH_CONFIG_KEY = "log_dir_path"
LOG_PATH_DEFAULT = "../log/"
LOGGER_DEFAULT = "logger_default"
LOGGER_MADE = "logger_made"

# Output
OUTPUT_PATH_CONFIG_KEY = "output_dir_path"
OUTPUT_PATH_DEFAULT = "../output/"

# Simulate
SIMULATE_SOURCE_HOSPITAL_ID = 420
SIMULATE_TARGET_HOSPITAL_ID = 449

SIMULATE_X_SOURCE_PATH = "simulate_x_source.csv"
SIMULATE_Y_SOURCE_PATH = "simulate_y_source.csv"
SIMULATE_X_TARGET_PATH = "simulate_x_target.csv"
SIMULATE_Y_TARGET_PATH = "simulate_y_target.csv"


SIMULATE_DATA_DIR = "simulation/"

TARGET_HOSPITAL_ID = "Northeast"
TOTAL_SEED = 10

FED_WEIGHT_METHOD_SGD = "fed_weight_method_sgd"
FED_WEIGHT_METHOD_AVG = "fed_weight_method_avg"
