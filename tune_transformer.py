import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.transformer import Net_Transformer
from data.synthetic_dataset import SyntheticDataset
from loss.dilate_loss import dilate_loss
from tslearn.metrics import dtw_path
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
# -----------------------
# Load dataset
# -----------------------
with open("synthetic_dataset.pkl", "rb") as f:
    Xtr_in, Xtr_out, Xte_in, Xte_out, tr_bkp, te_bkp = pickle.load(f)

dataset = SyntheticDataset(Xtr_in, Xtr_out, tr_bkp)

# ---- train / val split ----
n = len(dataset)
idx = np.random.permutation(n)
split = int(0.8 * n)

train_idx = idx[:split]
val_idx   = idx[split:]

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

trainloader = DataLoader(train_ds, batch_size=100, shuffle=True)
valloader   = DataLoader(val_ds, batch_size=100, shuffle=False)

# -----------------------
# Objective
# -----------------------
def objective(trial):

    trial_start = time.time()

    # ---- hyperparameters to tune ----
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4])
    ff_mult = trial.suggest_categorical("ff_mult", [2, 4])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    dim_ff = ff_mult * d_model

    print(
        f"\n[Trial {trial.number}] "
        f"d_model={d_model}, nhead={nhead}, dim_ff={dim_ff}, lr={lr:.2e}"
    )

    # ---- model ----
    model = Net_Transformer(
        input_size=1,
        target_length=20,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=dim_ff,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # ---- short training ----
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()

        for x, y, _ in trainloader:
            x = x.to(device).float()
            y = y.to(device).float()

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start
        print(
            f"  Epoch {epoch+1}/{epochs} | "
            f"train_loss={loss.item():.4f} | "
            f"time={epoch_time:.2f}s"
        )

    # ---- validation (DTW) ----
    model.eval()
    dtw_vals = []

    with torch.no_grad():
        for x, y, _ in valloader:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            pred = model(x)

            for i in range(pred.shape[0]):
                gt = y[i,:,0].cpu().numpy()
                pr = pred[i,:,0].cpu().numpy()
                _, d = dtw_path(gt, pr)
                dtw_vals.append(d)

    score = np.mean(dtw_vals)
    trial_time = time.time() - trial_start

    print(
        f"[Trial {trial.number} DONE] "
        f"DTW={score:.4f} | "
        f"trial_time={trial_time:.2f}s"
    )

    return score


# -----------------------
# Run study
# -----------------------
study_start = time.time()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

total_time = time.time() - study_start

print("\n=== Optuna study finished ===")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Average time per trial: {total_time/20:.2f} seconds")

best = study.best_trial

print("\n=== Best hyperparameters ===")
print(f"DTW score : {best.value:.4f}")
print(f"d_model   : {best.params['d_model']}")
print(f"nhead     : {best.params['nhead']}")
print(f"ff_mult   : {best.params['ff_mult']}")
print(f"dim_ff    : {best.params['ff_mult'] * best.params['d_model']}")
print(f"lr        : {best.params['lr']:.2e}")


import json

best_params = {
    "d_model": best.params["d_model"],
    "nhead": best.params["nhead"],
    "dim_feedforward": best.params["ff_mult"] * best.params["d_model"],
    "learning_rate": best.params["lr"],
    "best_dtw": best.value,
}

with open("best_transformer_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("\nSaved best hyperparameters to best_transformer_params.json")
