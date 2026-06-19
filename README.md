# MEDA Virtual Sensor Recovery — Mars Atmospheric Pressure Forecasting

A deep learning project for the **Mars Environmental Dynamics Analyzer (MEDA) Virtual Sensor Recovery** Kaggle competition, completed as part of MCTA 4363 (Deep Learning), Semester 2 2025/2026.

**Team:** Muhammad Ammar Zuhair bin Nor Azman Shah · Muhammad Basil bin Abdul Hakim · Muhammad Syafiq bin Nor Azman
**Lecturer:** Dr. Hasan Firdaus bin Mohd Zaki

## Overview

This project predicts Martian atmospheric **pressure** from a large multivariate time-series dataset (~9 million rows, millisecond-frequency sensor readings) covering rover orientation, solar angle, irradiance, and environmental variables. The core challenge was handling a dataset that is large in row count but highly redundant due to its sampling frequency, while still capturing the temporal dynamics that drive Martian pressure cycles.

## Approach

A phased modeling strategy was used, progressing from simple baselines to temporal architectures:

1. **MLP** — feedforward baseline, no explicit temporal modeling
2. **XGBoost** — tree-based ensemble for structured/tabular comparison
3. **LSTM** — recurrent sequence model to capture temporal dependencies
4. **TCN** — temporal convolutional network with causal/dilated convolutions and residual connections (final, best-performing model)

### Preprocessing
- Missing-value handling: high-missing columns dropped, remainder filled via linear interpolation + forward/backward fill
- Cyclical time encoding (sine/cosine) from Local Mean Sidereal Time
- Moving-average smoothing (window ≈ 1000 rows ≈ 1 second) to reduce sensor noise
- MinMax scaling for neural network inputs
- Sequence construction with a sampling rate of 1000, so each timestep represents ~1 second instead of 1 millisecond

### TCN Configuration
| Parameter | Value |
|---|---|
| Timesteps | 60 |
| Sampling rate | 1000 |
| Filters | 64 |
| Kernel size | 3 |
| Dropout | 0.10 |
| Learning rate | 0.001 |
| Batch size | 1024 |
| Optimizer | Adam |
| Loss | MSE |

## Results

| Model | Kaggle Score |
|---|---|
| MLP | 8803.98 |
| XGBoost | 8791.79 |
| LSTM | 7458.85 |
| **TCN** | **2352.01** |

The TCN model achieved the best leaderboard score, placing **4th** as of 11 June. Its causal + dilated convolutions and residual connections allowed it to capture both short- and long-term temporal patterns more effectively and efficiently than the recurrent (LSTM) or tabular (MLP/XGBoost) approaches.

## Key Takeaways

- High-frequency data ≠ high information content — adjacent millisecond rows were highly correlated, making sampling strategy critical
- Architecture must match data structure — convolutional sequence models outperformed both feedforward and recurrent approaches on this dataset
- Feature selection by missing-value threshold alone risks discarding predictive columns; importance-based selection is preferable
- The TCN reached peak performance at epoch 1, indicating fast convergence and a need for stronger overfitting controls

## Future Work

- Importance-based feature selection instead of missing-percentage thresholds
- Explore additional architectures (TiDE, N-BEATS, Transformers, hybrid TCN-LSTM)
- Systematic tuning of sampling rate and moving-average window
- Time-based cross-validation for robustness across different periods/sensor states
- Deeper error analysis via actual-vs-predicted plots across time intervals

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · XGBoost · Matplotlib · Jupyter Notebook

## Reference

Uprety, S., Bennaceur, A., Gavidia-Calderon, C., Holmes, J. A., Patel, M. R., & Rajendran, K. (2025). *Weather Prediction on Mars as a Multivariate Time Series Forecasting Problem.*
