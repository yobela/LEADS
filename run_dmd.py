#!/usr/bin/env python3
"""
Dynamic Mode Decomposition (DMD) baseline for comparison with LEADS (NN) on the same datasets.

DMD fits a linear model x_{t+1} = A x_t from snapshot pairs (X, Y). We fit one DMD matrix per
environment (one_per_env) or a single global DMD (one_for_all), then evaluate one-step and
multi-step rollout MSE on the test set to match LEADS metrics.
"""
from __future__ import division, print_function

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse LEADS dataset config and loaders
from datasets import init_dataloaders


def _collect_trajectories(dataloader, dataset_name):
    """Collect (state, env) from dataloader. state: (B, C, T) or (B, C, T, H, W)."""
    states_per_env = {}
    for batch in dataloader:
        state = batch['state']  # (B, C, T) or (B, C, T, H, W)
        envs = batch['env']    # (B,) or list
        if hasattr(envs, 'numpy'):
            envs = envs.numpy()
        state_np = state.numpy()
        B = state_np.shape[0]
        for i in range(B):
            e = int(envs[i]) if hasattr(envs, '__getitem__') else envs
            if e not in states_per_env:
                states_per_env[e] = []
            states_per_env[e].append(state_np[i])
    return states_per_env


def _to_snapshot_pairs(state_list, flatten_spatial=True):
    """
    From list of arrays each (C, T) or (C, T, H, W), build X and Y so that Y[:, k] = state at t+1, X[:, k] = state at t.
    Returns X, Y each (n_state, n_snapshots).
    """
    pairs_x, pairs_y = [], []
    for arr in state_list:
        if arr.ndim == 2:
            # (C, T)
            x = arr[:, :-1]   # (C, T-1)
            y = arr[:, 1:]    # (C, T-1)
            pairs_x.append(x)
            pairs_y.append(y)
        else:
            # (C, T, H, W) -> flatten to (C*H*W, T)
            C, T = arr.shape[0], arr.shape[1]
            flat = arr.reshape(C, T, -1)  # (C, T, H*W)
            flat = flat.transpose(0, 2, 1).reshape(-1, T)  # (C*H*W, T)
            x = flat[:, :-1]
            y = flat[:, 1:]
            pairs_x.append(x)
            pairs_y.append(y)
    X = np.hstack(pairs_x)  # (n_state, n_snapshots)
    Y = np.hstack(pairs_y)
    return X.astype(np.float64), Y.astype(np.float64)


def fit_dmd(X, Y, rank=None):
    """
    Exact DMD: minimize ||Y - A X||_F => A = Y X^+.
    X, Y: (n_state, n_snapshots). Returns A (n_state, n_state).
    """
    n_state, n_snap = X.shape
    if rank is not None and rank < min(n_state, n_snap):
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        U, s, Vh = U[:, :rank], s[:rank], Vh[:rank]
        Xinv = (Vh.T * (1 / s)) @ U.T
    else:
        Xinv = np.linalg.pinv(X)
    A = Y @ Xinv
    return A


def rollout_dmd(A, x0, n_steps):
    """Roll out x_{t+1} = A x_t for n_steps. x0: (n_state,) or (batch, n_state). Returns (n_steps+1, ...)."""
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim == 1:
        x0 = x0[np.newaxis, :]
    batch, n_state = x0.shape
    out = np.zeros((n_steps + 1, batch, n_state), dtype=np.float64)
    out[0] = x0
    for t in range(n_steps):
        out[t + 1] = (A @ out[t].T).T
    return out


def mse_one_step(A, X, Y):
    """One-step MSE: mean over snapshots of ||Y - A X||^2."""
    pred = A @ X
    return np.mean((Y - pred) ** 2)


def mse_rollout(A, state_list, max_steps=None, flatten_spatial=True):
    """
    Multi-step rollout MSE: for each trajectory, start from x0, roll out with A, compare to ground truth.
    state_list: list of (C, T) or (C, T, H, W). Uses same trajectory length - 1 steps.
    """
    errors = []
    for arr in state_list:
        if arr.ndim == 2:
            C, T = arr.shape
            n_steps = T - 1
            x0 = arr[:, 0]
            true_traj = arr[:, 1:].T  # (n_steps, n_state)
        else:
            C, T = arr.shape[0], arr.shape[1]
            n_steps = T - 1
            flat = arr.reshape(C, T, -1).transpose(0, 2, 1).reshape(-1, T)
            x0 = flat[:, 0]
            true_traj = flat[:, 1:].T  # (n_steps, n_state)
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)
            true_traj = true_traj[:n_steps]
        traj = rollout_dmd(A, x0, n_steps)
        pred_traj = traj[1:]  # (n_steps, 1, n_state) -> squeeze
        if pred_traj.shape[1] == 1:
            pred_traj = pred_traj[:, 0, :]
        err = np.mean((true_traj - pred_traj) ** 2)
        errors.append(err)
    return np.mean(errors), np.std(errors)


def run_dmd(dataset_name, path, one_for_all=False, rank=None):
    if dataset_name == 'lv':
        train_dl, test_dl = init_dataloaders('lv')
        n_env = 10
    elif dataset_name == 'gs':
        train_dl, test_dl = init_dataloaders('gs')
        n_env = 3
    elif dataset_name == 'ns':
        path_exp = path or './exp'
        buffer_path = os.path.join(path_exp, 'ns_buffer')
        if not os.path.exists(buffer_path + '_train.db'):
            print('NS buffer not found at', buffer_path, '- run LEADS once to create it, or use lv/gs.')
            sys.exit(1)
        train_dl, test_dl = init_dataloaders('ns', buffer_filepath=buffer_path)
        n_env = 4
    else:
        raise ValueError('dataset must be lv, gs, or ns')

    train_by_env = _collect_trajectories(train_dl, dataset_name)
    test_by_env = _collect_trajectories(test_dl, dataset_name)

    # Fit DMD
    if one_for_all:
        all_x, all_y = [], []
        for e in range(n_env):
            if e in train_by_env:
                Xe, Ye = _to_snapshot_pairs(train_by_env[e])
                all_x.append(Xe)
                all_y.append(Ye)
        X_global = np.hstack(all_x)
        Y_global = np.hstack(all_y)
        A_global = fit_dmd(X_global, Y_global, rank=rank)
        A_per_env = [A_global] * n_env
        mode = 'one_for_all'
    else:
        A_per_env = []
        for e in range(n_env):
            if e in train_by_env:
                Xe, Ye = _to_snapshot_pairs(train_by_env[e])
                Ae = fit_dmd(Xe, Ye, rank=rank)
                A_per_env.append(Ae)
            else:
                A_per_env.append(None)
        mode = 'one_per_env'

    # Train one-step MSE (per env)
    train_mse_1step = []
    for e in range(n_env):
        if A_per_env[e] is None:
            continue
        Xe, Ye = _to_snapshot_pairs(train_by_env.get(e, []))
        if Xe.size == 0:
            continue
        train_mse_1step.append(mse_one_step(A_per_env[e], Xe, Ye))
    train_mse_1step_mean = np.mean(train_mse_1step) if train_mse_1step else np.nan

    # Test one-step and rollout MSE
    test_mse_1step_per_env = []
    test_rollout_per_env = []
    for e in range(n_env):
        if A_per_env[e] is None or e not in test_by_env:
            continue
        Xe, Ye = _to_snapshot_pairs(test_by_env[e])
        test_mse_1step_per_env.append(mse_one_step(A_per_env[e], Xe, Ye))
        mean_r, std_r = mse_rollout(A_per_env[e], test_by_env[e])
        test_rollout_per_env.append((mean_r, std_r))

    test_mse_1step_mean = np.mean(test_mse_1step_per_env) if test_mse_1step_per_env else np.nan
    test_mse_1step_std = np.std(test_mse_1step_per_env) if test_mse_1step_per_env else np.nan
    test_rollout_mean = np.mean([x[0] for x in test_rollout_per_env]) if test_rollout_per_env else np.nan
    test_rollout_std = np.mean([x[1] for x in test_rollout_per_env]) if test_rollout_per_env else np.nan

    results = {
        'dataset': dataset_name,
        'mode': mode,
        'train_mse_one_step': train_mse_1step_mean,
        'test_mse_one_step_mean': test_mse_1step_mean,
        'test_mse_one_step_std': test_mse_1step_std,
        'test_mse_rollout_mean': test_rollout_mean,
        'test_mse_rollout_std': test_rollout_std,
        'n_env': n_env,
    }

    # Save results and optional plots
    out_dir = Path(path or './exp') / 'dmd_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'dmd_{dataset_name}_{mode}.txt'
    with open(log_path, 'w') as f:
        f.write('DMD results\n')
        f.write('dataset={} mode={} n_env={}\n'.format(dataset_name, mode, n_env))
        f.write('train_mse_one_step = {:.6e}\n'.format(results['train_mse_one_step']))
        f.write('test_mse_one_step  = {:.6e} +- {:.6e}\n'.format(
            results['test_mse_one_step_mean'], results['test_mse_one_step_std']))
        f.write('test_mse_rollout   = {:.6e} +- {:.6e}\n'.format(
            results['test_mse_rollout_mean'], results['test_mse_rollout_std']))

    # Plot: optional trajectory comparison for LV (2D)
    if dataset_name == 'lv' and test_by_env:
        fig, axes = plt.subplots(2, min(3, n_env), figsize=(4 * min(3, n_env), 8))
        if n_env == 1:
            axes = np.array([axes])
        axes = axes.flat
        for idx, e in enumerate(range(min(3, n_env))):
            if A_per_env[e] is None or e not in test_by_env:
                continue
            traj = test_by_env[e][0]  # first test trajectory (2, T)
            x0 = traj[:, 0]
            T = traj.shape[1]
            rolled = rollout_dmd(A_per_env[e], x0, T - 1)
            pred = rolled[1:, 0, :].T  # (2, T-1); align with true[:, 1:]
            true = traj[:, 1:]
            ax = axes[idx]
            ax.plot(true[0], true[1], 'b-', label='True', alpha=0.7)
            ax.plot(pred[0], pred[1], 'r--', label='DMD', alpha=0.7)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Env {}'.format(e))
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('DMD vs true trajectory (test, first trajectory per env)')
        plt.tight_layout()
        plt.savefig(out_dir / f'dmd_{dataset_name}_trajectories.png', dpi=150)
        plt.close()

    print('DMD results saved to', out_dir)
    print('train_mse_one_step = {:.6e}'.format(results['train_mse_one_step']))
    print('test_mse_one_step  = {:.6e} +- {:.6e}'.format(
        results['test_mse_one_step_mean'], results['test_mse_one_step_std']))
    print('test_mse_rollout   = {:.6e} +- {:.6e}'.format(
        results['test_mse_rollout_mean'], results['test_mse_rollout_std']))
    return results


def main():
    parser = argparse.ArgumentParser(description='Run DMD baseline on LEADS datasets.')
    parser.add_argument('dataset', type=str, choices=['lv', 'gs', 'ns'],
                        help='Dataset: lv (Lotka-Volterra), gs (Gray-Scott), ns (Navier-Stokes)')
    parser.add_argument('-p', '--path', type=str, default='./exp',
                        help='Experiment root (for NS buffer and output)')
    parser.add_argument('--one_for_all', action='store_true',
                        help='Fit a single global DMD instead of one per environment')
    parser.add_argument('--rank', type=int, default=None,
                        help='SVD rank for DMD (default: full)')
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    run_dmd(args.dataset, args.path, one_for_all=args.one_for_all, rank=args.rank)


if __name__ == '__main__':
    main()
