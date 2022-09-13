import os
from pathlib import Path
import torch


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)


PKL_PROTO = 4

DATA_ROOT = Path("./data")
WORKLOAD_ROOT = Path("./workload")
OUTPUT_ROOT = Path("./output")
mkdir(OUTPUT_ROOT)
TEMP_ROOT = OUTPUT_ROOT / "temp"
mkdir(TEMP_ROOT)
RESULT_ROOT = OUTPUT_ROOT / "result"
mkdir(RESULT_ROOT)
LOG_ROOT = OUTPUT_ROOT / "log"
mkdir(LOG_ROOT)
DATASET_PATH = RESULT_ROOT / "dataset"
mkdir(DATASET_PATH)
TEMP_DATASET_PATH = TEMP_ROOT / "dataset"
mkdir(TEMP_DATASET_PATH)
MODEL_PATH = TEMP_ROOT / "model"
mkdir(MODEL_PATH)

# Generating Workload Settings
WORKLOAD_PARA = {
    'attr': {'pred_number': 1.0},
    'center': {'distribution': 0.9, 'vocab_ood': 0.1},
    'width': {'uniform': 0.5, 'exponential': 0.5},
    'number': {'train': 100000, 'valid': 10000, 'test': 10000}
}

# sample info
MAX_SAMPLE_SPACE = 0.2
STANDARD_SAMPLE_PAR = {"ratio": 0.1, "seed": 6398084, "replace": False}
SAMPLE_RATIO_DIS = 0.02
SAMPLE_GROUP_PAR = {
    "s0.01r": {"ratio": 0.01, "seed": 3242356, "replace": True},
    "s0.01": {"ratio": 0.01, "seed": 217946543, "replace": False},
}

SAMPLE_IRR_COL_RATE = 0
SAMPLE_DIS_COL_RATE = 0.2
SAMPLE_DIS_NUM = 0

STANDARD_MULTI_SAMPLE_PAR = {"ratio": 0.1, "seed": 5, "replace": False}
MULTI_SAMPLE_GROUP_PAR = {
    "s0.05": {"ratio": 0.05, "seed": 3, "replace": False},
    "s0.01": {"ratio": 0.01, "seed": 20, "replace": False},
    # "s0.02r": {"ratio": 0.02, "seed": 217946543, "replace": True},
}

UD_SAMPLE_GROUP_PAR = {
    "s0.01": {"ratio": 0.01, "seed": 3242356, "replace": True},
    "s0.05": {"ratio": 0.05, "seed": 217946543, "replace": False},
}


# Features - Histogram
FEAT_MAX_BINS = 20
FEAT_MAX_CBINS = -1


# Model-settings
DEVICE = torch.device('cuda', 1) if torch.cuda.is_available() else 'cpu'
NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count()))
EPOCHES = 100

# JOB-settings
JOB_BASE_TABLE = "title"
JOB_TABLE_LIST = ["title", "movie_info", "movie_info_idx", "cast_info", "movie_keyword", "movie_companies"]
JOB_TABLE_DICT = {"title": "t", "movie_info": "mi", "movie_info_idx": "mi_idx",
                  "cast_info": "ci", "movie_keyword": "mk", "movie_companies": "mc"}
JOB_JOIN_COL = {"title": "id", "movie_info": "movie_id", "movie_info_idx": "movie_id",
                "cast_info": "movie_id", "movie_keyword": "movie_id", "movie_companies": "movie_id"}
JOB_COL_DIST = {
    "title": ["episode_nr", "episode_of_id", "id", "kind_id", "production_year", "season_nr"],
    "cast_info": ["id", "movie_id", "nr_order", "person_id", "person_role_id", "role_id"],
    "movie_info": ["id", "info_type_id", "movie_id"],
    "movie_info_idx": ["id", "info_type_id", "movie_id"],
    "movie_companies": ["company_id", "company_type_id", "id", "movie_id"],
    "movie_keyword": ["id", "keyword_id", "movie_id"],
}

CENSUS_TABLE_COL = ["age", "workclass", "education", "education_num", "marital_status", "occupation",
                    "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]
DMV_TABLE_COL = ['Record_Type', 'Registration_Class', 'State', 'County', 'Body_Type', 'Fuel_Type', 'Reg_Valid_Date',
                 'Color', 'Scofflaw_Indicator', 'Suspension_Indicator', 'Revocation_Indicator']
FOREST_TABLE_COL = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
POWER_TABLE_COL = ["Global_active_power", "Global_reactive_power", "Voltage",
                   "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
