# Data/dgp_config.py
import os
import os.path as op
from pathlib import Path

WORK_DIR = "/hpc2hdd/home/jliu043/CNN_Replicate"

def get_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

DATA_DIR = get_dir(op.join(WORK_DIR, "data"))
PROCESSED_DATA_DIR = get_dir(op.join(DATA_DIR, "processed_data"))
STOCKS_SAVEPATH = get_dir(os.path.join(DATA_DIR, "stocks_dataset"))

CACHE_DIR = Path(get_dir(op.join(WORK_DIR, "CACHE_DIR")))
PORTFOLIO = Path(get_dir(op.join(CACHE_DIR, "PORTFOLIO")))

BAR_WIDTH = 3
LINE_WIDTH = 1

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

VOLUME_CHART_GAP = 1
BACKGROUND_COLOR = 0
CHART_COLOR = 255

FREQ_DICT = {1: "daily", 5: "week", 20: "month", 60: "quarter", 65: "quarter", 260: "year"}

RESULTS_DIR = get_dir(op.join(WORK_DIR, "results"))
MODELS_DIR = get_dir(op.join(WORK_DIR, "models"))