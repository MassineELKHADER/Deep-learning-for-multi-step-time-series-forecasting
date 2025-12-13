import torch
from . import soft_dtw
from . import path_soft_dtw 

def flexible_dilate_loss(outputs, targets, alpha, gamma, device, Omega=None):
    """
    DILATE loss where the user can specify Omega.
    If Omega=None, a default asymmetric temporal penalty is used.
    """
    batch_size, N_output = outputs.shape[0:2]

    # ---------- Shape term ----------
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output), device=device)

    for k in range(batch_size):
        D[k] = soft_dtw.pairwise_distances(
            targets[k,:,:].view(-1,1),
            outputs[k,:,:].view(-1,1)
        )
    loss_shape = softdtw_batch(D, gamma)

    # ---------- Temporal path ----------
    path = path_soft_dtw.PathDTWBatch.apply(D, gamma)
    
    
    # ---------- Omega definition ----------
    if Omega is None or Omega == "l2":  # default option, use the paper's Omega
        t = torch.arange(1, N_output+1).float().view(-1,1).to(device)
        Omega = soft_dtw.pairwise_distances(x=t, y=t)
    elif Omega == "l1": # L1 temporal penalty
        t = torch.arange(1, N_output+1).float().view(-1,1).to(device)
        Omega = soft_dtw.pairwise_distances_l1(t, t)
    elif Omega == "asymmetric": # asymmetric temporal penalty (late >> early)
        t = torch.arange(1, N_output+1).float().view(-1,1).to(device)
        Omega = soft_dtw.pairwise_temporal_asymmetric(t, t, late_weight=3.0, early_weight=1.0)
    elif Omega == 'huber':
        t = torch.arange(1, N_output+1).float().view(-1,1).to(device)
        Omega = soft_dtw.pairwise_distances_huber(t, t, delta=5.0)
    else:
        raise ValueError("Unknown Omega type, please choose among None, 'l2', 'l1', 'asymmetric', 'huber'.")
    
    # ---------- Temporal loss ----------
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)

    # ---------- Final DILATE ----------
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
