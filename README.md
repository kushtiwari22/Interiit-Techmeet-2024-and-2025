# Interiit-Techmeet-2024-and-2025
This repository contains the methods, code implementations, experiments, and results developed during the Inter IIT Tech Meet 2024 and 2025 competitions. The 2024 problem statement was provided by Zelta (Untrade), while the 2025 challenge was by Ebullient Securities.


# Multi-Strategy Financial Market Prediction and Algorithmic Trading Framework
## Overview
This repository contains three independent approaches for financial market prediction and algorithmic trading developed during **InterIIT 2024–2025**. The project focuses on using historical market data and advanced machine learning techniques to generate profitable trading signals and optimize portfolio performance.
The repository integrates:

- Deep Learning based forecasting using **N-BEATS**
- Reinforcement Learning based trading using **PPO**
- Deterministic signal generation using **Technical Indicators**

The objective is to analyze market behavior, generate trading decisions, and compare different approaches for algorithmic trading.

---

# Key Features

### Deep Learning Forecasting

- N-BEATS based time series prediction
- Rolling-window forecasting
- Future close-price prediction
- Signal generation

### Reinforcement Learning

- PPO Actor-Critic architecture
- Feature engineering pipeline
- Reward engineering
- Portfolio tracking
- Risk-aware trading

### Technical Analysis

- Multiple technical indicators
- Noise reduction filters
- Trend and momentum analysis
- Deterministic signal generation

### Trading Utilities

- Backtesting engine
- Portfolio evaluation
- Performance visualization
- Trading signal generation

---

# Repository Structure

```bash

Interiit_2024-2025/
│
├── Nbeats_approch/
│   │
│   ├── Engine/
│   │
│   ├── Result_and_architecture/
│   │
│   ├── Technical/
│   │
│   ├── backtesting_engine.ipynb
│   │
│   ├── Nbeats_final_submission.ipynb
│   │
│   ├── signal_generation.ipynb
│   │
│   └── README.md
│
│
├── reinforcement_learning_approch/
│   │
│   ├── __pycache__/
│   │
│   ├── Agents/
│   │
│   ├── configs/
│   │
│   ├── Data_preprocessing/
│   │
│   ├── Model_params/
│   │
│   ├── Other_experimentation/
│   │
│   ├── Trading environment/
│   │
│   ├── backtesting.py
│   │
│   ├── Mid_term_report.pdf
│   │
│   └── End_term_report.pdf
│
│
├── Techincal_analysis_approch/
│   │
│   ├── btc_data/
│   │
│   ├── eth_data/
│   │
│   ├── Result_and_reports/
│   │
│   ├── Algorithm_1_btc.py
│   │
│   ├── main_1_eth.py
│   │
│   ├── final_logs_btc.csv
│   │
│   ├── final_logs_eth.csv
│   │
│   └── README_2024.md
│
│
├── README.md
│
└── requirements.txt

```

---

# Approach 1: N-BEATS Forecasting Approach

## Objective

The N-BEATS approach uses deep neural networks for forecasting financial time series and generating future market signals.

### Features

- OHLCV data preprocessing
- Rolling-window sequence generation
- Deep neural forecasting
- Signal generation
- Result visualization

### Workflow

```text
OHLCV Data
    ↓
Preprocessing
    ↓
Rolling Window Generation
    ↓
N-BEATS Model
    ↓
Future Price Prediction
    ↓
Signal Generation
    ↓
Backtesting
```

---

# Approach 2: Reinforcement Learning Approach

## Objective

The reinforcement learning approach trains a trading agent using **Proximal Policy Optimization (PPO)**.

The model learns trading behavior directly from market data through interaction with an environment.

---

## Feature Engineering

Features include:

- Price-based features
- Volume features
- Price Behavior (PB) features
- Band Behavior (BB) features
- Time-lag features
- Difference features
- Feature normalization
- Data resampling

---

## PPO Architecture

Main components:

### Actor Network

Responsible for deciding:

- Buy
- Sell
- Hold

### Critic Network

Responsible for estimating state values.

### PPO Memory

Stores:

- States
- Rewards
- Actions
- Probabilities

### Portfolio Tracker

Tracks:

- Portfolio value
- Drawdown
- Rewards
- Trade performance

---

## Reward Engineering

The reward function considers:

- Realized profit/loss
- Unrealized profit/loss
- Transaction costs
- Holding penalties
- Drawdown penalties
- Directional rewards
- Risk-adjusted rewards

---

# Approach 3: Technical Analysis Based Strategy

## Objective

Generate trading signals using deterministic trading logic and technical indicators.

---

## Trend Indicators

- EMA
- MACD
- KST
- DMI
- Aroon Oscillator

---

## Momentum Indicators

- RSI
- Smoothed RSI
- Chande Momentum Oscillator

---

## Volatility Indicators

- Chaikin Volatility
- Donchian Channel

---

## Noise Reduction Methods

- Kalman Filter
- Chebyshev Filter
- Hawkes Process

---

## Candlestick Features

- Heiken Ashi

---

# Data Processing Pipeline

```text

Load OHLCV Data
        ↓
Handle Missing Values
        ↓
Feature Engineering
        ↓
Normalization
        ↓
Model Training
        ↓
Signal Generation
        ↓
Backtesting
        ↓
Performance Evaluation

```

---

# Performance Metrics

The following metrics are used for evaluating strategies:

- Portfolio Growth
- Total Return
- Maximum Drawdown
- Win Rate
- Sharpe Ratio
- Trade Statistics
- Signal Accuracy

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-repository-name.git

cd Interiit_2024-2025
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## N-BEATS Approach

```bash
cd Nbeats_approch

jupyter notebook Nbeats_final_submission.ipynb
```

---

## Reinforcement Learning Approach

```bash
cd reinforcement_learning_approch

python backtesting.py
```

---

## Technical Analysis Approach

```bash
cd Techincal_analysis_approch

python Algorithm_1_btc.py

python main_1_eth.py
```

---

# Future Improvements

Future work includes:

- Dynamic stop loss implementation
- Better portfolio optimization
- Hyperparameter tuning
- Multi-asset support
- Live trading integration
- Transformer-based forecasting
- Better risk management techniques

---

# Authors

InterIIT 2024–2025

Financial Market Prediction and Algorithmic Trading Research Project
