# Klaud Portfolio Predictor

## Overveiw

Predicts stock movement as up (+1), neutral (0), or down (-1) using tabular 
   and time series data using various ML models. Pulls 10 years of data via yfinance and evaluates 
   strategies using Sharpe ratio, Sortino, max drawdown, and more.

## Models

| Type | Models |
|------|--------|
| Tabular | Random Forest, SVM, Gradient Boosting |
| Time Series | RNN, LSTM |

## Usage

### Installation

```bash
# Clone repo
$ git clone https://github.com/coldmayo/KlaudPortfolioPredictor.git
# cd into project
$ cd KlaudPortfolioPredictor
# Install needed Python packages
$ pip install -r requirements.txt
```

### Data Generation

```bash
# cd into source code
$ cd src
# Run data gen code for tabular data
$ python build_dataset.py -t tabular
# Run data gen code for time series data
$ python build_dataset.py -t time
```

### Training and Backtesting

```bash
# On a HPC
# For LSTM model
$ sbatch LSTM.slurm
# For Parallelized Random Forest 
$ sbatch RF_MPI.slurm
# Running Locally
$ python train.py -c ../configs/SVM.json
```

## Project Structure
KlaudPortfolioPredictor/<br>
├── configs/       # Model and data configs<br>
├── notes/         # Research notes<br>
├── papers/        # Reference papers<br>
├── src/           # Source code<br>
└── TODO.md        # Roadmap<br>

## Roadmap

See [TODO.md](./TODO.md) for planned features and progress
