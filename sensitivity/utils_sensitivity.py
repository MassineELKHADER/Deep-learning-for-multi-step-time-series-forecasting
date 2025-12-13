import numpy as np
import torch
from tslearn.metrics import dtw_path
import pickle
import matplotlib.pyplot as plt
import os

def evaluate_metrics(net, loader, device):
    criterion = torch.nn.MSELoss()
    mse_list, dtw_list, tdi_list = [], [], []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            outputs = net(inputs)

            mse = criterion(outputs, targets).item()
            mse_list.append(mse)

            B, T, _ = outputs.shape
            for k in range(B):
                y = targets[k,:,0].cpu().numpy()
                yhat = outputs[k,:,0].cpu().numpy()

                path, dist = dtw_path(y, yhat)
                dtw_list.append(dist)

                tdi = sum((i-j)**2 for i,j in path) / (T*T)
                tdi_list.append(tdi)

    return {
        "mse": np.mean(mse_list),
        "dtw": np.mean(dtw_list),
        "tdi": np.mean(tdi_list)
    }

def plot_sensitivity(
    results_path,
    x_key,
    save_path,
    title,
    x_label,
    log_x=False,
    Omega = False
):
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    x = [r[x_key] for r in results]
    mse = [r["mse"] for r in results]
    dtw = [r["dtw"] for r in results]
    tdi = [r["tdi"] for r in results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, mse, marker="o", label="MSE")
    plt.plot(x, dtw, marker="s", label="DTW")
    plt.plot(x, tdi, marker="^", label="TDI")

    if log_x:
        plt.xscale("log")
    if Omega:
        omega_labels = ["l2", "l1", "asymmetric", "huber"]
        plt.xticks(range(len(omega_labels)), omega_labels)
    plt.xlabel(x_label)
    plt.ylabel("Metric value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
