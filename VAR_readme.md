# VAR (Vector Autoregression) Experimentation

This document describes the **Vector Autoregression (VAR)** baseline used to compare with the LEADS neural network on the same datasets (LV, GS, NS). VAR is a standard time-series / dynamical-systems method from the course that models the state as a linear function of its past values.

## Background

- **VAR(p)** model: \( x_t = c + A_1 x_{t-1} + \cdots + A_p x_{t-p} \).
- **VAR(1)** is the discrete-time linear model \( x_t = c + A_1 x_{t-1} \); with \( c=0 \), one-step prediction is equivalent to DMD.
- Coefficients \( (c, A_1, \ldots, A_p) \) are fitted by **OLS** (least squares) on the stacked lagged design matrix, per environment or globally.

## Script: `run_var.py`

- **Input**: Same datasets as LEADS, via `datasets.init_dataloaders()`, so data and train/test splits match the NN experiments.
- **Modes**:
  - **One per environment** (default): Fit a separate VAR(p) per environment.
  - **One for all** (`--one_for_all`): Fit a single VAR(p) on all training data.

## Usage

```bash
# VAR(1) on Lotka-Volterra (default: one per env)
python run_var.py lv -p ./exp

# VAR(2) on LV
python run_var.py lv -p ./exp --order 2

# Gray-Scott
python run_var.py gs -p ./exp

# Navier-Stokes (requires buffer in ./exp)
python run_var.py ns -p ./exp

# Single global VAR(1)
python run_var.py lv -p ./exp --one_for_all

# No constant term (e.g. zero-mean assumption)
python run_var.py lv -p ./exp --no_const
```

## Outputs and Validation

- **Metrics** (aligned with LEADS):
  - **Train one-step MSE**: Mean squared error of one-step-ahead prediction on training data.
  - **Test one-step MSE**: Same on test data (mean ± std over envs).
  - **Test rollout MSE**: Multi-step rollout from initial \( p \) states using the fitted VAR(p), then MSE vs. ground-truth trajectory (comparable to NN rollout).

- **Files** (under `-p` path, e.g. `./exp/var_results/`):
  - `var_{lv,gs,ns}_p{p}_{one_per_env|one_for_all}.txt`: Train/test one-step and test rollout MSE.
  - `var_lv_p1_trajectories.png` (for LV, p=1): True vs VAR-rolled trajectory for a few test envs.

## Comparison with LEADS (NN) and DMD

| Aspect        | VAR (`run_var.py`)        | DMD (`run_dmd.py`)   | LEADS (NN)           |
|---------------|---------------------------|----------------------|----------------------|
| Model         | Linear, p lags + constant | Linear, 1 step       | ODE-based NN         |
| Training      | OLS (closed-form)         | Closed-form \( A=YX^+ \) | Gradient-based       |
| Generalization| Per-env or global         | Per-env or global    | Shared + env-specific|
| Metric        | One-step & rollout MSE    | One-step & rollout   | Trajectory MSE       |

Use **test rollout MSE** as the main comparable metric to LEADS. Comparing **VAR(1)** with **DMD** (and with LEADS) on the same dataset and same train/test split gives a direct comparison of linear baselines vs the neural approach.

## Dependencies

Same as LEADS: `numpy`, `matplotlib`, and the project’s `datasets` module. No `statsmodels` or other extra packages required; VAR is implemented with `np.linalg.lstsq`.
