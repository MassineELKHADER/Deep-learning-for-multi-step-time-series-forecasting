import numpy as np
import pickle
import random
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from tslearn.metrics import dtw_path
from loss.dilate_loss import dilate_loss
import os

PLOT_DIR = "baselines/ridge_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_features(X_input):
    """
    X_input: (N, N_input) or (N, N_input, 1)
    returns: (N, N_input) flat features
    """
    if X_input.ndim == 3:
        X_input = X_input[:, :, 0]
    return X_input.astype(np.float32)

def to_targets(X_target):
    """
    X_target: (N, N_output) or (N, N_output, 1)
    returns: (N, N_output)
    """
    if X_target.ndim == 3:
        X_target = X_target[:, :, 0]
    return X_target.astype(np.float32)

def eval_metrics_numpy(y_true, y_pred, gamma=0.01, alpha=0.5, device="cpu"):
    """
    y_true, y_pred: (N, N_output)
    returns dict with mean and std for each metric
    """
    N, N_output = y_true.shape

    # --- Per-sample MSE ---
    mse_i = np.mean((y_pred - y_true) ** 2, axis=1)

    # --- Per-sample Huber ---
    delta = 1.0
    r = y_pred - y_true
    abs_r = np.abs(r)
    huber_i = np.mean(
        np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta)),
        axis=1,
    )

    # --- DTW + TDI ---
    dtw_i = []
    tdi_i = []
    for i in range(N):
        path, dtw_val = dtw_path(y_true[i], y_pred[i])
        dtw_i.append(dtw_val)
        tdi = sum((a - b) ** 2 for a, b in path) / (N_output * N_output)
        tdi_i.append(tdi)

    dtw_i = np.array(dtw_i)
    tdi_i = np.array(tdi_i)

    # --- DILATE (per-sample) ---
    yt = torch.tensor(y_true[:, :, None], dtype=torch.float32, device=device)
    yp = torch.tensor(y_pred[:, :, None], dtype=torch.float32, device=device)
    dil_i, _, _ = dilate_loss(yt, yp, alpha=alpha, gamma=gamma, device=device)
    dil_i = dil_i.detach().cpu().numpy()

    def mean_std(x):
        return float(np.mean(x)), float(np.std(x))

    return {
        "MSE": mean_std(mse_i),
        "Huber": mean_std(huber_i),
        "DTW": mean_std(dtw_i),
        "TDI": mean_std(tdi_i),
        "DILATE": mean_std(dil_i),
    }

def save_input_pred_gt(X_input, Y_true, Y_pred, idx, save_dir):
    """
    Plot input (past), GT future, and predicted future.
    """
    input_len = X_input.shape[1]
    output_len = Y_true.shape[1]

    t_input = np.arange(1, input_len+1)
    t_output = np.arange(input_len, input_len + output_len)

    plt.figure(figsize=(6, 4))

    # Input (past)
    plt.plot(
        t_input,
        X_input[idx],
        label="Input (past)",
        color="black",
        linestyle="--",
        linewidth=2,
    )

    # Ground truth future
    plt.plot(
        t_output,
        Y_true[idx],
        label="GT future",
        linewidth=2,
    )

    # Predicted future
    plt.plot(
        t_output,
        Y_pred[idx],
        label="Ridge prediction",
        linewidth=2,
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Test sample {idx}: input -> future forecast")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(save_dir, f"sample_{idx}_input_pred_gt.png")
    plt.savefig(fname)
    plt.close()

def save_dtw_alignment(y_true, y_pred, idx, save_dir):
    path, _ = dtw_path(y_true, y_pred)
    path = np.array(path)

    T = len(y_true)

    plt.figure(figsize=(4, 4))
    plt.plot(path[:, 0], path[:, 1], linewidth=2)
    plt.plot([0, T], [0, T], "--", color="gray", label="Perfect alignment")
    plt.xlabel("GT time index")
    plt.ylabel("Predicted time index")
    plt.title(f"DTW alignment (sample {idx})")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(save_dir, f"sample_{idx}_dtw.png")
    plt.savefig(fname)
    plt.close()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gamma = 0.01
    alpha = 0.5

    # -----------------------
    # Load dataset
    # -----------------------
    with open("synthetic_dataset.pkl", "rb") as f:
        X_train_input, X_train_target, X_test_input, X_test_target, train_bkp, test_bkp = pickle.load(f)

    Xtr = to_features(X_train_input)
    Ytr = to_targets(X_train_target)
    Xte = to_features(X_test_input)
    Yte = to_targets(X_test_target)

    # -----------------------
    # Train Ridge (multi-output)
    # -----------------------
    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(Xtr, Ytr)

    # -----------------------
    # Predict on test set
    # -----------------------
    Ypred = model.predict(Xte).astype(np.float32)

    # -----------------------
    # Compute metrics (TEST)
    # -----------------------
    metrics = eval_metrics_numpy(Yte, Ypred, gamma=gamma, alpha=alpha, device=device)

    print("\n=== Ridge multi-output (test metrics) ===")
    for k, (m, s) in metrics.items():
        print(f"{k:7s}: {m:.4f} ± {s:.4f}")

    # -----------------------
    # Sanity-check plots
    # -----------------------

    n_plots = 5
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] 

    for idx in indices:
        save_input_pred_gt(Xte, Yte, Ypred, idx, PLOT_DIR)
        save_dtw_alignment(Yte[idx], Ypred[idx], idx, PLOT_DIR)

    print(f"\nSaved {n_plots} qualitative Ridge examples to {PLOT_DIR}")


# === Ridge multi-output (test metrics) ===
# MSE    : 0.0607
# Huber  : 0.0303
# DTW    : 0.8425
# TDI    : 2.0752
# DILATE : 1.4549