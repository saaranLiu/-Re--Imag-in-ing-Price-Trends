# CNN Stock Trend Prediction

This project implements the CNN model for stock price trend prediction based on the paper "(Re-)Imag(in)ing Price Trends". The model converts OHLC (Open, High, Low, Close) price data into images and uses CNN to predict future price movements. And this is the final project of INFO 5000 (Fall 2025).
The group members include: Jiaxin Liu, Yushu He, Yurong Tian, Shurui Zhang and Xin Xin

## Project Structure
CNN_REPLICATE/
├── CACHE_DIR/ # Cache directory for temporary files
├── data/ # Raw data files
├── Data/ # Data processing modules (scripts)
├── logs/ # Training and evaluation logs // You can check the log files containing these task numbers: 7406071, 7406238, 7406467, to see exactly how the train and testing were implemented.
├── Misc/ # Utility functions and configurations
├── Model/ # CNN model implementation
├── patches/ # Code patches and fixes
├── Portfolio/ # Portfolio construction and analysis
├── results/ # Models, predictions and evaluation results
├── scripts/ # Running scripts
└── cnn_env.yml # Conda environment configuration


## Key Components

### Data Processing (`Data/`)
- Image generation from OHLC data
- Data preprocessing and normalization
- Dataset splitting (Training: 1993-2000, Testing: 2001-2019)
- Support for different window sizes (5, 20, 60 days)

### Model Architecture (`Model/`)
- CNN implementation for different input windows
- Training and validation procedures
- Model ensemble support
- Checkpoint management

### Portfolio Analysis (`Portfolio/`)
- Portfolio construction based on predictions

### Results Analysis (`results/`)
- Trained models
- Prediction results storage
- Performance metrics calculation
- Portfolio results

## Setup and Installation

1. Create conda environment:
```bash
conda env create -f cnn_env.yml
```

2. Activate environment:
```bash
conda activate cnn_env
```

## Usage

### Training
```bash
python scripts/train_model.py --input_window 5 --predict_window 5
```


### Portfolio Analysis
```bash
python Portfolio/analyze_portfolio.py --model I5R5 --start_year 2001 --end_year 2019
```

## Model Configurations

### Input Windows
- I5: 5-day window
- I20: 20-day window
- I60: 60-day window

### Prediction Windows
- R5: 5-day prediction
- R20: 20-day prediction
- R60: 60-day prediction

## Results Directory Structure
results/
├── CNN_I5R5/
│ └── epoch5_ensem1_train1993-2000/
│ └── predictions/
├── CNN_I20R5/
└── CNN_I60R5/


## Performance Metrics

The model evaluation includes:
- Prediction accuracy
- Return prediction
- Portfolio performance
- Risk-adjusted returns

## Dependencies

Main dependencies (see `cnn_env.yml` for complete list):
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Notes

- The model uses ensemble learning with multiple CNN instances
- Predictions are generated on a rolling basis
- Portfolio construction considers transaction costs
- Results are saved in CSV format for further analysis

## Citation
the original paper:
    Jiang, J., Kelly, B., & Xiu, D. (2023). (Re‐) Imag (in) ing price trends. The Journal of Finance, 78(6), 3193-3249.
