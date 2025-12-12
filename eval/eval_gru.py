# eval/eval_gru.py
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tslearn.metrics import dtw_path

from data.synthetic_dataset import SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU

# -----------------------
# config
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
N_input = 20
N_output = 20
weights_path = "weights/gru/gru_mse_500.pth" 

# -----------------------
# load test data
# -----------------------
with open("synthetic_dataset.pkl", "rb") as f:
    X_train_input, X_train_target, X_test_input, X_test_target, train_bkp, test_bkp = pickle.load(f)

dataset_test = SyntheticDataset(X_test_input, X_test_target, test_bkp)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# -----------------------
# Load the model
# -----------------------
encoder = EncoderRNN(
    input_size=1,
    hidden_size=128,
    num_grulstm_layers=1,
    batch_size=batch_size
).to(device)

decoder = DecoderRNN(
    input_size=1,
    hidden_size=128,
    num_grulstm_layers=1,
    fc_units=16,
    output_size=1
).to(device)

net = Net_GRU(encoder, decoder, N_output, device).to(device)

# -----------------------
# load weights
# -----------------------
net.load_state_dict(torch.load(weights_path, map_location=device))
net.eval()

print(f"Loaded weights from: {weights_path}")

# -----------------------
# evaluation
# -----------------------
criterion = torch.nn.MSELoss()

mse_list = []
dtw_list = []
tdi_list = []

with torch.no_grad():
    for inputs, targets, _ in testloader:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

        outputs = net(inputs)

        # ---- MSE ----
        mse = criterion(outputs, targets).item()
        mse_list.append(mse)

        # ---- DTW + TDI ----
        for k in range(outputs.size(0)):
            y_true = targets[k, :, 0].cpu().numpy()
            y_pred = outputs[k, :, 0].cpu().numpy()

            path, dtw_val = dtw_path(y_true, y_pred)
            dtw_list.append(dtw_val)

            tdi = sum((i - j) ** 2 for i, j in path) / (N_output * N_output)
            tdi_list.append(tdi)

# -----------------------
# RESULTS
# -----------------------
print("\n=== TEST RESULTS ===")
print(f"MSE : {np.mean(mse_list):.4f}")
print(f"DTW : {np.mean(dtw_list):.4f}")
print(f"TDI : {np.mean(tdi_list):.4f}")
