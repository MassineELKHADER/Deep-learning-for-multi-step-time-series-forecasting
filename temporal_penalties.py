import torch

def pairwise_distances_l1(x, y=None):
    if y is None: y = x
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return diff.abs().sum(dim=-1)

def pairwise_distances_huber(x, y=None, delta=1.0):
    if y is None: y = x
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    r = diff.norm(p=2, dim=-1)
    abs_r = r.abs()
    quadratic = 0.5 * abs_r**2
    linear = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quadratic, linear)

def pairwise_temporal_asymmetric(t_true, t_pred=None, late_weight=3.0, early_weight=1.0):
    if t_pred is None: t_pred = t_true
    s = (t_pred.unsqueeze(0) - t_true.unsqueeze(1)).squeeze(-1)
    weights = torch.where(s > 0,
                          torch.tensor(late_weight, device=s.device),
                          torch.tensor(early_weight, device=s.device))
    return weights * (s ** 2)

def pairwise_distances_l2(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = y.t()
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = x.t()
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))
