# CNN Stock Trend Prediction

> **INFO 5000 (Fall 2025) - Group 7 Final Project**

This project implements a Convolutional Neural Network (CNN) model for stock price trend prediction based on the paper **"(Re-)Imag(in)ing Price Trends"**. The model converts OHLC (Open, High, Low, Close) price data into images and uses CNN to predict future price movements.

## Team Members

| Name | Role / Year |
| :--- | :--- |
| **Jiaxin Liu** | '24 Mphil student (FTEC) |
| **Yushu He** | '24 PhD student (FTEC) |
| **Yurong Tian** | '24 PhD student (FTEC) |
| **Shurui Zhang** | '25 Mphil student |
| **Xin Xin** | '25 Mphil student |

---

## Project Structure

```text
CNN_REPLICATE/
├── CACHE_DIR/      # Cache directory for temporary files
├── data/           # Raw data files
├── Data/           # Data processing modules (scripts)
├── logs/           # Training and evaluation logs (See note below)
├── Misc/           # Utility functions and configurations
├── Model/          # CNN model implementation
├── Portfolio/      # Portfolio construction and analysis
├── scripts/        # Running scripts
└── cnn_env.yml     # Conda environment configuration
````

> **Note:** regarding the `logs/` directory, you can check the log files containing task numbers **7406071**, **7406238**, and **7406467** to see exactly how the training and testing were implemented.

## Key Components

### Data Processing (`Data/`)

  * **Image Generation:** Converts OHLC data into visual representations.
  * **Preprocessing:** Data normalization and cleaning.
  * **Dataset Splitting:**
      * Training: 1993-2000
      * Testing: 2001-2019
  * **Window Support:** Handles 5, 20, and 60-day windows.

### Model Architecture (`Model/`)

  * CNN implementation tailored for different input windows.
  * Training and validation procedures.
  * Model ensemble support.
  * Checkpoint management.

### Portfolio Analysis (`Portfolio/`)

  * Portfolio construction based on model predictions.
  * Incorporates transaction costs into analysis.

### Results Analysis (`results/`)

  * Storage for trained models and prediction CSVs.
  * Calculation of performance metrics and portfolio results.
  * See the reports for more details about the results. The results doesnot upload here because of the file size.

## Setup and Installation

**1. Create conda environment:**

```bash
conda env create -f cnn_env.yml
```

**2. Activate environment:**

```bash
conda activate cnn_env
```

## Usage

### Training

Run the training script with defined input and prediction windows:

```bash
python scripts/train_model.py --input_window 5 --predict_window 5
```

### Portfolio Analysis

Analyze portfolio performance for a specific model (e.g., I5R5) over a specific period:

```bash
python Portfolio/analyze_portfolio.py --model I5R5 --start_year 2001 --end_year 2019
```

## Model Configurations

The project uses specific abbreviations for Input (I) and Prediction (R) windows:

| Category | Tag | Description |
| :--- | :---: | :--- |
| **Input Window** | `I5` | 5-day historical window |
| | `I20` | 20-day historical window |
| | `I60` | 60-day historical window |
| **Prediction Window** | `R5` | Predict next 5 days |
| | `R20` | Predict next 20 days |
| | `R60` | Predict next 60 days |

### Results Directory Structure

```text
results/
├── CNN_I5R5/
│   ├── epoch5_ensem1_train1993-2000/
│   └── predictions/
├── CNN_I20R5/
└── CNN_I60R5/
```

## Performance Metrics

The model evaluation includes:

  * Prediction accuracy
  * Return prediction
  * Portfolio performance
  * Risk-adjusted returns

## Dependencies

Main dependencies (see `cnn_env.yml` for the complete list):

  * Python 3.8+
  * PyTorch
  * Pandas
  * NumPy
  * Matplotlib
  * scikit-learn

## Notes

  * The model uses **ensemble learning** with multiple CNN instances.
  * Predictions are generated on a **rolling basis**.
  * Portfolio construction explicitly considers **transaction costs**.
  * Results are saved in **CSV format** for further analysis.

## Citation

Original paper reference:
```
@article{jiang2023re,
  title={(Re-) Imag (in) ing price trends},
  author={Jiang, Jingwen and Kelly, Bryan and Xiu, Dacheng},
  journal={The Journal of Finance},
  volume={78},
  number={6},
  pages={3193--3249},
  year={2023},
  publisher={Wiley Online Library}
}

```

