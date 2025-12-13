import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.transformer import Net_Transformer
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import wandb
import time
from utils import format_time
import os
import pickle

save_dir = "results/Transformer"
os.makedirs(save_dir, exist_ok=True)
weights_dir = "weights/transformer"
os.makedirs(weights_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)
random.seed(0)

# parameters
batch_size = 100
N = 500
N_input = 20
N_output = 20  
sigma = 0.01
gamma = 0.01
# epochs = 5
epochs = 500

def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    start_time = time.time()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs): 
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]                     

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse       
 
            if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        elapsed_time = time.time() - start_time
        # ---- wandb logging ----
        if loss_type=='dilate':
            wandb.log({
                f"{loss_type}/total_loss": loss.item(),
                f"{loss_type}/shape_loss": loss_shape.item(),
                f"{loss_type}/temporal_loss": loss_temporal.item(),
                f"{loss_type}/elapsed_time": elapsed_time,
                "epoch": epoch
            })

        else:
            wandb.log({
                f"{loss_type}/total_loss": loss.item(),
                f"{loss_type}/elapsed_time": elapsed_time,
                "epoch": epoch
            })
            
        if(verbose):
            if (epoch % print_every == 0) or (epoch==epochs-1):
                print('epoch ', epoch+1, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                eval_model(net,testloader, gamma,verbose=1)
        
        

def eval_model(net,loader, gamma,verbose=1):   
    net.eval()

    mse_i = []
    huber_i = []
    dtw_i = []
    tdi_i = []

    delta = 1.0  # Huber parameter

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            targets = torch.tensor(targets, dtype=torch.float32).to(device)

            outputs = net(inputs)

            B, T, _ = targets.shape

            # --- per-sample MSE & Huber ---
            r = outputs - targets
            mse_batch = torch.mean(r**2, dim=(1, 2))
            abs_r = torch.abs(r)
            huber_batch = torch.mean(
                torch.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta)),
                dim=(1, 2)
            )

            mse_i.extend(mse_batch.cpu().numpy())
            huber_i.extend(huber_batch.cpu().numpy())

            # --- DTW + TDI ---
            for k in range(B):
                y_true = targets[k, :, 0].cpu().numpy()
                y_pred = outputs[k, :, 0].cpu().numpy()

                path, dtw_val = dtw_path(y_true, y_pred)
                dtw_i.append(dtw_val)

                tdi = sum((i - j) ** 2 for i, j in path) / (T * T)
                tdi_i.append(tdi)

    def mean_std(x):
        return float(np.mean(x)), float(np.std(x))

    results = {
        "MSE": mean_std(mse_i),
        "Huber": mean_std(huber_i),
        "DTW": mean_std(dtw_i),
        "TDI": mean_std(tdi_i),
    }
    net.train()
    return results



if __name__ == '__main__':
    
    wandb.init(project="dilate-transformer-forecasting",
            group="Transformer",
            name="Transformer_DILATE_vs_MSE",
            config={
                "batch_size": batch_size,
                "N_input": N_input,
                "N_output": N_output,
                "gamma": gamma,
                "epochs": epochs,
            })

    with open("synthetic_dataset.pkl", "rb") as f:
        X_train_input, X_train_target, X_test_input, X_test_target, train_bkp, test_bkp = pickle.load(f)

    dataset_train = SyntheticDataset(X_train_input, X_train_target, train_bkp)
    dataset_test  = SyntheticDataset(X_test_input, X_test_target, test_bkp)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    net_trans_dilate = Net_Transformer(input_size=1, target_length=N_output, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, device=device).to(device)
    print('Training Transformer with DILATE loss...')
    train_model(net_trans_dilate, loss_type='dilate', learning_rate=0.001, epochs=epochs, gamma=gamma, print_every=50, eval_every=50, verbose=1)
    save_path = os.path.join(weights_dir, f"net_trans_dilate_{epochs}.pth")
    torch.save(net_trans_dilate.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")

    net_trans_mse = Net_Transformer(input_size=1, target_length=N_output, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, device=device).to(device)
    print("Training Transformer with MSE loss...")
    train_model( net_trans_mse, loss_type='mse', learning_rate=0.001, epochs=epochs, gamma=gamma, print_every=50, eval_every=50, verbose=1)
    save_path = os.path.join(weights_dir, f"net_trans_mse_{epochs}.pth")
    torch.save(net_trans_mse.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")

    print("\n=== Transformer (MSE-trained) test metrics ===")
    metrics_trans_mse = eval_model(
        net_trans_mse, testloader, gamma=gamma
    )
    for k, (m, s) in metrics_trans_mse.items():
        print(f"{k:6s}: {m:.4f} ± {s:.4f}")

    print("\n=== Transformer (DILATE-trained) test metrics ===")
    metrics_trans_dilate = eval_model(
        net_trans_dilate, testloader, gamma=gamma
    )
    for k, (m, s) in metrics_trans_dilate.items():
        print(f"{k:6s}: {m:.4f} ± {s:.4f}")

    # Visualize results
    gen_test = iter(testloader)
    test_inputs, test_targets, breaks = next(gen_test)

    test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
    criterion = torch.nn.MSELoss()

    nets = [net_trans_mse, net_trans_dilate]
    names = ["Transformer + MSE", "Transformer + DILATE"]


    for ind in range(1, 10):
        with torch.no_grad():
            preds = [net(test_inputs).to(device) for net in nets]

        input_seq  = test_inputs[ind].detach().cpu().numpy()
        target_seq = test_targets[ind].detach().cpu().numpy()

        plt.figure(figsize=(17, 5))

        # ---- Plot input and target once ----
        plt.plot(range(N_input),
                input_seq,
                label="input",
                linewidth=3)

        plt.plot(range(N_input-1, N_input+N_output),
                np.concatenate([input_seq[N_input-1:N_input], target_seq]),
                label="target",
                linewidth=3,
                color="black")

        # ---- Plot predictions from both models ----
        for pred, name in zip(preds, names):
            pred_seq = pred[ind].detach().cpu().numpy()
            plt.plot(
                range(N_input-1, N_input+N_output),
                np.concatenate([input_seq[N_input-1:N_input], pred_seq]),
                label=f"prediction ({name})",
                linewidth=3,
            )

        plt.title(f"Sample {ind}: MSE vs DILATE on Transformer")
        plt.legend()

        # ---- Save ----
        filename = f"Transformer_sample_{ind}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
