import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU, Net_Transformer
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
epochs = 5
# epochs = 500

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
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                eval_model(net,testloader, gamma,verbose=1)
        
        

def eval_model(net,loader, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
            loss_dtw += sim
                       
            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)
                        
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    print( ' Eval mse= ', np.array(losses_mse).mean() ,' dtw= ',np.array(losses_dtw).mean() ,' tdi= ', np.array(losses_tdi).mean()) 


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

    net_trans_dilate = Net_Transformer(input_size=1, target_length=N_output, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, device=device).to(device)
    print('Training Transformer with DILATE loss...')
    train_model(net_trans_dilate, loss_type='dilate', learning_rate=0.001, epochs=epochs, gamma=gamma, print_every=50, eval_every=50, verbose=1)
    torch.save(net_trans_dilate.state_dict(), "net_trans_dilate.pth")
    print("Saved weights: net_trans_dilate.pth")

    net_trans_mse = Net_Transformer(input_size=1,target_length=N_output,d_model=128,nhead=4,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=256,device=device).to(device)
    print("Training Transformer with MSE loss...")
    train_model( net_trans_mse, loss_type='mse', learning_rate=0.001, epochs=epochs, gamma=gamma, print_every=50, eval_every=50, verbose=1)
    torch.save(net_trans_mse.state_dict(), "net_trans_mse.pth")
    print("Saved weights: net_trans_mse.pth")

    # Visualize results
    gen_test = iter(testloader)
    test_inputs, test_targets, breaks = next(gen_test)

    test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
    criterion = torch.nn.MSELoss()

    nets = [net_trans_mse, net_trans_dilate]
    names = ["Transformer + MSE", "Transformer + DILATE"]


    for ind in range(1, 5):
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
