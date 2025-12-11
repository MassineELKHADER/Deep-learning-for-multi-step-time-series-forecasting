import torch
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, alpha, gamma, device, Omega=None):
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
    if Omega is None:  # default option
        t = torch.arange(1, N_output+1).float().view(-1,1).to(device)
        Omega = soft_dtw.pairwise_temporal_asymmetric(t_true=t, t_pred=t)

    # ---------- Temporal loss ----------
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)

    # ---------- Final DILATE ----------
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
