import torch
import pickle
import numpy as np
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from sensitivity.utils_sensitivity import evaluate_metrics, plot_sensitivity
from sensitivity.train_script import train_model   # import your existing function
from torch.utils.data import DataLoader
from data.synthetic_dataset import SyntheticDataset
import time
from utils import format_time
batch_size = 100
Omega_choices = ["l2", "l1", "asymmetric", "huber"]
Omega_to_id = {name: i for i, name in enumerate(Omega_choices)}

alpha = 0.5
gamma = 0.01
epochs = 250
# epochs = 2

# Load the synthetic dataset (same split as training)
with open("synthetic_dataset.pkl", "rb") as f:
    X_train_input, X_train_target, X_test_input, X_test_target, train_bkp, test_bkp = pickle.load(f)

dataset_train = SyntheticDataset(X_train_input, X_train_target, train_bkp)
dataset_test = SyntheticDataset(X_test_input, X_test_target, test_bkp)
trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


results = []

for Omega in Omega_choices:
    print(f"\n=== Omega = {Omega} ===")

    encoder = EncoderRNN(1, 128, 1, batch_size=100).to(device)
    decoder = DecoderRNN(1, 128, 1, 16, 1).to(device)
    net = Net_GRU(encoder, decoder, target_length=20, device=device).to(device)
    t0 = time.time()
    train_model(
        trainloader, 
        device,
        net,
        loss_type="dilate",
        learning_rate=1e-3,
        gamma=gamma,
        alpha=alpha,
        epochs=epochs,
        Omega=Omega,
    )

    metrics = evaluate_metrics(net, testloader, device)
    metrics["Omega_id"] = Omega_to_id[Omega]
    results.append(metrics)
    print("Runtime : ", format_time(time.time() - t0))

with open("sensitivity/Omega/results_Omega.pkl", "wb") as f:
    pickle.dump(results, f)

plot_sensitivity(
    results_path="sensitivity/Omega/results_Omega.pkl",
    x_key="Omega_id",
    x_label="Omega choice", 
    save_path="sensitivity/Omega/Omega_sensitivity.png",
    title="DILATE Sensitivity to Omega (Shape vs Temporal Trade-off)",
    log_x=False,
    Omega=True
)
