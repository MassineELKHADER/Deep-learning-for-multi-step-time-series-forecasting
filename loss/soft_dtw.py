import numpy as np
import torch
from numba import jit
from torch.autograd import Function

def pairwise_distances_l1(x, y=None):
    """
    L1 pairwise distances.
    x: (N, d), y: (M, d) or None -> uses y = x
    returns D[i,j] = ||x_i - y_j||_1
    """
    if y is None:
        y = x
    # (N, 1, d) - (1, M, d) -> (N, M, d)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist = diff.abs().sum(dim=-1)  # (N, M)
    return dist


def pairwise_distances_huber(x, y=None, delta=1.0):
    """
    Huber-style distance on the Euclidean norm.
    D[i,j] = huber(||x_i - y_j||_2).
    delta: transition between quadratic and linear.
    """
    if y is None:
        y = x
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, d)
    r = diff.norm(p=2, dim=-1)              # (N, M)  Euclidean distance

    abs_r = r.abs()
    quadratic = 0.5 * abs_r**2
    linear = delta * (abs_r - 0.5 * delta)
    dist = torch.where(abs_r <= delta, quadratic, linear)
    return dist


def pairwise_temporal_asymmetric(t_true, t_pred=None,
                                 late_weight=1.0, early_weight=1.0):
    """
    Asymmetric quadratic penalty for temporal distances.
    t_true: (N, 1) time indices of target
    t_pred: (M, 1) time indices of prediction (or None -> same)
    
    We define s_ij = t_pred_j - t_true_i:
        s_ij > 0  -> prediction is LATE  relative to true time
        s_ij <= 0 -> prediction is EARLY (or on time)
    
    D[i,j] = (late_weight if s_ij>0 else early_weight) * s_ij^2
    """
    if t_pred is None:
        t_pred = t_true

    # (1, M, 1) - (N, 1, 1) -> (N, M, 1) -> squeeze -> (N, M)
    s = t_pred.unsqueeze(0) - t_true.unsqueeze(1)
    s = s.squeeze(-1)

    weights = torch.where(s > 0,
                          torch.as_tensor(late_weight, device=s.device),
                          torch.as_tensor(early_weight, device=s.device))
    dist = weights * (s ** 2)
    return dist


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size,N,N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N+2 ,N+2)).to(dev)   
        for k in range(0, batch_size): # loop over all D in the batch    
            Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(dev)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size,N,N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N ,N)).to(dev) 
        for k in range(batch_size):         
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(dev)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None


