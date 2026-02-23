# DMD (Dynamic Mode Decomposition) Experimentation

This document describes the **Dynamic Mode Decomposition (DMD)** baseline used to compare with the LEADS neural network on the same datasets (LV, GS, NS). DMD is a course-relevant, linear data-driven method that fits a discrete-time linear model from snapshot pairs.

## Background

- **DMD** approximates the dynamics with a linear map \( A \) such that \( x_{t+1} \approx A x_t \).
- Snapshot pairs \( (X, Y) \) are built from trajectories: \( X \) contains \( x_t \), \( Y \) contains \( x_{t+1} \).
- **Exact DMD** solves \( A = Y X^+ \) (minimize \( \|Y - A X\|_F \) in the Frobenius norm).
- For high-dimensional or rank-deficient data, optional **reduced-rank DMD** uses an SVD truncation of \( X \) (via `--rank`).

## Script: `run_dmd.py`

- **Input**: Same datasets as LEADS (Lotka-Volterra, Gray-Scott, Navier-Stokes), loaded via `datasets.init_dataloaders()` so train/test splits and data generation match the NN experiments.
- **Modes**:
  - **One per environment** (default): Fit a separate DMD matrix \( A_e \) per environment \( e \) (aligned with LEADS “one per env” / multi-environment setup).
  - **One for all** (`--one_for_all`): Fit a single global DMD from all training data (shared across environments).

## Usage

```bash
# Lotka-Volterra (default: one DMD per env)
python run_dmd.py lv -p ./exp

# Gray-Scott
python run_dmd.py gs -p ./exp

# Navier-Stokes (requires precomputed buffer in ./exp, e.g. from running LEADS once)
python run_dmd.py ns -p ./exp

# Single global DMD (one for all envs)
python run_dmd.py lv -p ./exp --one_for_all

# Reduced-rank DMD (e.g. rank 5)
python run_dmd.py lv -p ./exp --rank 5
```

## Outputs and Validation

- **Metrics** (aligned with LEADS):
  - **Train one-step MSE**: \( \mathbb{E}\|x_{t+1} - A x_t\|^2 \) on training snapshots.
  - **Test one-step MSE**: Same on test snapshots (mean ± std over envs).
  - **Test rollout MSE**: Multi-step rollout from \( x_0 \) with \( x_{t+1} = A x_t \), then MSE vs. ground-truth trajectory (comparable to NN rollout evaluation).

- **Files** (under `-p` path, e.g. `./exp/dmd_results/`):
  - `dmd_{lv,gs,ns}_{one_per_env|one_for_all}.txt`: Summary of train/test one-step and test rollout MSE.
  - `dmd_lv_trajectories.png` (for LV): Plot of true vs DMD-rolled trajectory for a few test envs.

## Comparison with LEADS (NN)

| Aspect        | DMD (`run_dmd.py`)     | LEADS (NN)                    |
|---------------|------------------------|--------------------------------|
| Model         | Linear \( x_{t+1}=A x_t \) | ODE-based NN, env-specific     |
| Training      | Closed-form \( A=YX^+ \)  | Gradient-based, many steps     |
| Generalization| Per-env or global A    | Shared + env-specific modules  |
| Metric        | One-step & rollout MSE | Trajectory MSE (rollout)       |

Use **test rollout MSE** as the main comparable metric to the LEADS test MSE reported in the main README. One-step MSE is useful to see how much error comes from the first step vs. accumulation over time.

## Dependencies

Uses the same environment as LEADS: `numpy`, `scipy`, `matplotlib`, and the project’s `datasets` module. No extra packages beyond the main `requirements.txt`.
