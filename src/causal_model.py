import torch
import torch.nn as nn

class StructuralCausalModel(nn.Module):
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels, resolution):
        super(StructuralCausalModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(colour_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (resolution // 8) * (resolution // 8), s_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(s_dim, 128 * (resolution // 8) * (resolution // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, resolution // 8, resolution // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, colour_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.beta_s = beta_s
        self.beta_o = beta_o
        self.gamma = gamma

    def forward(self, x):
        s = self.encoder(x)
        x_recon = self.decoder(s)
        return x_recon, s

    def counterfactual(self, x, intervention):
        s = self.encoder(x)
        s_intervened = s + intervention
        x_recon = self.decoder(s_intervened)
        return x_recon, s_intervened