# Misc/config.py
import os

def get_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

WORK_DIR = "/hpc2hdd/home/jliu043/CNN_Replicate"

EXP_DIR = get_dir(os.path.join(WORK_DIR, "results"))
PORTFOLIO_DIR = get_dir(os.path.join(EXP_DIR, "portfolio"))
LOG_DIR = get_dir(os.path.join(WORK_DIR, "logs"))
LATEX_DIR = get_dir(os.path.join(EXP_DIR, "latex"))

BATCH_SIZE = 128
TRUE_DATA_CNN_INPLANES = 64
BENCHMARK_MODEL_LAYERNUM_DICT = {5: 2, 20: 3, 60: 4}
EMP_CNN_BL_SETTING = {
    5: ([(5, 3)] * 10, [(1, 1)] * 10, [(1, 1)] * 10, [(2, 1)] * 10),
    20: (
        [(5, 3)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(2, 1)] + [(1, 1)] * 10,
        [(2, 1)] * 10,
    ),
    60: (
        [(5, 3)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(2, 1)] * 10,
    ),
}

TS1D_LAYERNUM_DICT = {5: 1, 20: 2, 60: 3}
EMP_CNN1d_BL_SETTING = {
    5: ([3] * 1, [1] * 1, [1] * 1, [2] * 1),
    20: ([3] * 2, [1] * 2, [1] * 2, [2] * 2),
    60: ([3] * 3, [1] * 3, [1] * 3, [2] * 3),
}

NUM_WORKERS = 4

IS_YEARS = list(range(1993, 2000))  # 训练集年份
OOS_YEARS = list(range(2001, 2019))  # 测试集年份

FREQ_DICT = {1: "daily", 5: "week", 20: "month", 60: "quarter", 65: "quarter", 260: "year"}