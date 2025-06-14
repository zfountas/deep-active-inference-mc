import torch
import torch.nn as nn

def compute_loss_causal(x_recon, o1, s, pi0, log_Ppi, model):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_recon, o1)
    kl_div_s = torch.sum(-0.5 * torch.sum(1 + s - s.pow(2) - s.exp(), dim=1))
    omega = model.beta_s * kl_div_s + model.beta_o * recon_loss
    F = recon_loss + omega
    return F, kl_div_s, omega