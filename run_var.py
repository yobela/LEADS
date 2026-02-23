#!/usr/bin/env python3
"""
Vector Autoregression (VAR) baseline for comparison with LEADS (NN) on the same datasets.

VAR(p) model: x_t = c + A_1 x_{t-1} + ... + A_p x_{t-p}. Fitted via OLS per environment
(or one global model). Evaluates one-step and multi-step rollout MSE on the test set.
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

from datasets import init_dataloaders


def _collect_trajectories(dataloader, dataset_name):
    """Collect (state, env) from dataloader."""
    states_per_env = {}
    for batch in dataloader:
        state = batch['state']
        envs = batch['env']
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


def _trajectories_to_flat(state_list):
    """Convert list of (C, T) or (C, T, H, W) to list of (n_state, T) arrays."""
    out = []
    for arr in state_list:
        if arr.ndim == 2:
            out.append(arr.astype(np.float64))
        else:
            C, T = arr.shape[0], arr.shape[1]
            flat = arr.reshape(C, T, -1).transpose(0, 2, 1).reshape(-1, T)
            out.append(flat.astype(np.float64))
    return out


def build_var_design(seq, p, include_const=True):
    """
    seq: (n_state, T). Build Y = [x_p, x_{p+1}, ..., x_{T-1}] and
    X = [1, x_{p-1}, ..., x_0; 1, x_p, ..., x_1; ...] (each row is lagged vector).
    Returns X (n_obs, 1 + n_state*p or n_state*p), Y (n_obs, n_state).
    """
    n_state, T = seq.shape
    n_obs = T - p
    if n_obs <= 0:
        return None, None
    Y = seq[:, p:].T   # (n_obs, n_state)
    if include_const:
        X = np.ones((n_obs, 1 + n_state * p), dtype=np.float64)
    else:
        X = np.zeros((n_obs, n_state * p), dtype=np.float64)
    for i in range(n_obs):
        for j in range(p):
            X[i, (1 if include_const else 0) + j * n_state:(1 if include_const else 0) + (j + 1) * n_state] = seq[:, p - 1 - j + i]
    return X, Y


def fit_var(seq_list, p, include_const=True):
    """
    Fit VAR(p) from list of (n_state, T) arrays. Returns (c, A_1, ..., A_p) where
    x_t = c + A_1 x_{t-1} + ... + A_p x_{t-p}.
    c is (n_state,), A_i is (n_state, n_state). If include_const False, c = 0.
    """
    X_all, Y_all = [], []
    for seq in seq_list:
        X, Y = build_var_design(seq, p, include_const=include_const)
        if X is None:
            continue
        X_all.append(X)
        Y_all.append(Y)
    if not X_all:
        return None
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    n_state = Y_all.shape[1]
    # OLS: Y_all = X_all @ B.T  => B = (X_all.T @ X_all)^{-1} @ X_all.T @ Y_all
    B = np.linalg.lstsq(X_all, Y_all, rcond=None)[0]  # (n_pred, n_state)
    if include_const:
        c = B[0]
        A_list = [B[1 + i * n_state:1 + (i + 1) * n_state].T for i in range(p)]
    else:
        c = np.zeros(n_state)
        A_list = [B[i * n_state:(i + 1) * n_state].T for i in range(p)]
    return (c, A_list)


def rollout_var(c, A_list, x_init, n_steps):
    """
    x_init: initial states, shape (n_state,) for p=1 or (n_state, p) for p>1.
    Columns are x_0, x_1, ..., x_{p-1}. Roll out for n_steps.
    Returns (n_state, p + n_steps) - full trajectory.
    """
    p = len(A_list)
    n_state = A_list[0].shape[0]
    x_init = np.asarray(x_init, dtype=np.float64)
    if x_init.ndim == 1:
        hist = x_init.reshape(n_state, -1)
    else:
        hist = x_init
    if hist.shape[1] < p:
        raise ValueError('Need at least p initial states')
    hist = hist[:, -p:]  # (n_state, p)
    out = [hist[:, j].copy() for j in range(p)]
    for _ in range(n_steps):
        x_new = c.copy()
        for j in range(p):
            x_new += A_list[j] @ out[-1 - j]
        out.append(x_new)
    return np.column_stack(out)  # (n_state, p + n_steps)


def mse_one_step_var(c, A_list, seq_list):
    """One-step MSE for VAR: for each trajectory, predict one step and compare."""
    p = len(A_list)
    errors = []
    for seq in seq_list:
        n_state, T = seq.shape
        for t in range(p, T):
            pred = c.copy()
            for j in range(p):
                pred += A_list[j] @ seq[:, t - 1 - j]
            errors.append(np.sum((seq[:, t] - pred) ** 2))
    return np.mean(errors) if errors else np.nan


def mse_rollout_var(c, A_list, state_list, max_steps=None):
    """Multi-step rollout MSE: roll out from first p points, compare to rest of trajectory."""
    p = len(A_list)
    errors = []
    for seq in state_list:
        n_state, T = seq.shape
        n_steps = T - p
        if n_steps <= 0:
            continue
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)
        hist = seq[:, :p]  # (n_state, p)
        rolled = rollout_var(c, A_list, hist, n_steps)
        # rolled is (n_state, p + n_steps); true is seq[:, p:p+n_steps]
        true_traj = seq[:, p:p + n_steps]
        pred_traj = rolled[:, p:]
        err = np.mean((true_traj - pred_traj) ** 2)
        errors.append(err)
    return np.mean(errors), np.std(errors) if errors else (np.nan, np.nan)


def run_var(dataset_name, path, p=1, one_for_all=False, include_const=True):
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
            print('NS buffer not found. Use lv/gs or create NS buffer first.')
            sys.exit(1)
        train_dl, test_dl = init_dataloaders('ns', buffer_filepath=buffer_path)
        n_env = 4
    else:
        raise ValueError('dataset must be lv, gs, or ns')

    train_by_env = _collect_trajectories(train_dl, dataset_name)
    test_by_env = _collect_trajectories(test_dl, dataset_name)

    train_flat = {e: _trajectories_to_flat(train_by_env.get(e, [])) for e in range(n_env)}
    test_flat = {e: _trajectories_to_flat(test_by_env.get(e, [])) for e in range(n_env)}

    if one_for_all:
        all_seqs = []
        for e in range(n_env):
            all_seqs.extend(train_flat[e])
        model = fit_var(all_seqs, p, include_const=include_const)
        if model is None:
            print('Not enough data for VAR(p) with p=', p)
            sys.exit(1)
        c, A_list = model
        models_per_env = [(c, A_list)] * n_env
        mode = 'one_for_all'
    else:
        models_per_env = []
        for e in range(n_env):
            model = fit_var(train_flat[e], p, include_const=include_const)
            models_per_env.append(model)
        mode = 'one_per_env'

    # Train one-step MSE
    train_mse_1step = []
    for e in range(n_env):
        if models_per_env[e] is None:
            continue
        c, A_list = models_per_env[e]
        train_mse_1step.append(mse_one_step_var(c, A_list, train_flat[e]))
    train_mse_1step_mean = np.mean(train_mse_1step) if train_mse_1step else np.nan

    # Test one-step and rollout
    test_mse_1step_per_env = []
    test_rollout_per_env = []
    for e in range(n_env):
        if models_per_env[e] is None or not test_flat[e]:
            continue
        c, A_list = models_per_env[e]
        test_mse_1step_per_env.append(mse_one_step_var(c, A_list, test_flat[e]))
        mean_r, std_r = mse_rollout_var(c, A_list, test_flat[e])
        test_rollout_per_env.append((mean_r, std_r))

    test_mse_1step_mean = np.mean(test_mse_1step_per_env) if test_mse_1step_per_env else np.nan
    test_mse_1step_std = np.std(test_mse_1step_per_env) if test_mse_1step_per_env else np.nan
    test_rollout_mean = np.mean([x[0] for x in test_rollout_per_env]) if test_rollout_per_env else np.nan
    test_rollout_std = np.mean([x[1] for x in test_rollout_per_env]) if test_rollout_per_env else np.nan

    results = {
        'dataset': dataset_name,
        'mode': mode,
        'p': p,
        'train_mse_one_step': train_mse_1step_mean,
        'test_mse_one_step_mean': test_mse_1step_mean,
        'test_mse_one_step_std': test_mse_1step_std,
        'test_mse_rollout_mean': test_rollout_mean,
        'test_mse_rollout_std': test_rollout_std,
        'n_env': n_env,
    }

    out_dir = Path(path or './exp') / 'var_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'var_{dataset_name}_p{p}_{mode}.txt'
    with open(log_path, 'w') as f:
        f.write('VAR(p) results\n')
        f.write('dataset={} p={} mode={} n_env={}\n'.format(dataset_name, p, mode, n_env))
        f.write('train_mse_one_step = {:.6e}\n'.format(results['train_mse_one_step']))
        f.write('test_mse_one_step  = {:.6e} +- {:.6e}\n'.format(
            results['test_mse_one_step_mean'], results['test_mse_one_step_std']))
        f.write('test_mse_rollout   = {:.6e} +- {:.6e}\n'.format(
            results['test_mse_rollout_mean'], results['test_mse_rollout_std']))

    # Plot trajectories for LV
    if dataset_name == 'lv' and test_flat and p == 1:
        fig, axes = plt.subplots(2, min(3, n_env), figsize=(4 * min(3, n_env), 8))
        if n_env == 1:
            axes = np.array([axes])
        axes = axes.flat
        for idx, e in enumerate(range(min(3, n_env))):
            if models_per_env[e] is None or not test_flat[e]:
                continue
            seq = test_flat[e][0]
            n_state, T = seq.shape
            c, A_list = models_per_env[e]
            hist = seq[:, :1].T  # (1, n_state) -> we need p columns; p=1 so hist (n_state, 1)
            rolled = rollout_var(c, A_list, seq[:, 0], T - 1)
            pred = rolled[:, 1:]
            true = seq[:, 1:]
            ax = axes[idx]
            ax.plot(true[0], true[1], 'b-', label='True', alpha=0.7)
            ax.plot(pred[0], pred[1], 'r--', label='VAR(1)', alpha=0.7)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Env {}'.format(e))
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('VAR(1) vs true trajectory (test)')
        plt.tight_layout()
        plt.savefig(out_dir / f'var_{dataset_name}_p{p}_trajectories.png', dpi=150)
        plt.close()

    print('VAR results saved to', out_dir)
    print('train_mse_one_step = {:.6e}'.format(results['train_mse_one_step']))
    print('test_mse_one_step  = {:.6e} +- {:.6e}'.format(
        results['test_mse_one_step_mean'], results['test_mse_one_step_std']))
    print('test_mse_rollout   = {:.6e} +- {:.6e}'.format(
        results['test_mse_rollout_mean'], results['test_mse_rollout_std']))
    return results


def main():
    parser = argparse.ArgumentParser(description='Run VAR baseline on LEADS datasets.')
    parser.add_argument('dataset', type=str, choices=['lv', 'gs', 'ns'],
                        help='Dataset: lv, gs, or ns')
    parser.add_argument('-p', '--path', type=str, default='./exp',
                        help='Experiment root')
    parser.add_argument('--order', type=int, default=1, dest='p',
                        help='VAR order (default: 1)')
    parser.add_argument('--one_for_all', action='store_true',
                        help='Fit single global VAR')
    parser.add_argument('--no_const', action='store_true',
                        help='Do not include constant term')
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    run_var(args.dataset, args.path, p=args.p, one_for_all=args.one_for_all, include_const=not args.no_const)


if __name__ == '__main__':
    main()
