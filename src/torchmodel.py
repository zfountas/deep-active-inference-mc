import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from shutil import copyfile
from src.torchutils import *


class ModelTop(nn.Module):
    def __init__(self, s_dim, pi_dim):
        super(ModelTop, self).__init__()
        # For activation function we used ReLU.
        # For weight initialization we used He Uniform

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.qpi_net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pi_dim)
        )

    def encode_s(self, s0):
        logits_pi = self.qpi_net(s0)
        q_pi = F.softmax(logits_pi, dim=-1)
        log_q_pi = torch.log(q_pi + 1e-20)
        return logits_pi, q_pi, log_q_pi


class ModelMid(nn.Module):
    def __init__(self, s_dim, pi_dim):
        super(ModelMid, self).__init__()

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.ps_net = nn.Sequential(
            nn.Linear(pi_dim + s_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, s_dim * 2)
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def transition(self, pi, s0):
        output = self.ps_net(torch.cat([pi, s0], dim=1))
        mean, logvar = torch.split(output, self.s_dim, dim=1)
        return mean, logvar

    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar


class ModelDown(nn.Module):
    def __init__(self, s_dim, pi_dim, colour_channels, resolution):
        super(ModelDown, self).__init__()

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.colour_channels = colour_channels
        self.resolution = resolution
        if self.resolution == 64:
            last_strides = 2
        elif self.resolution == 32:
            last_strides = 1
        else:
            raise ValueError("Unknown resolution")

        self.qs_net = nn.Sequential(
            nn.Conv2d(self.colour_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, s_dim * 2)
        )

        self.po_net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=last_strides, padding=1, output_padding=last_strides-1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.colour_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def encoder(self, o):
        output = self.qs_net(o)
        mean_s, logvar_s = torch.split(output, self.s_dim, dim=1)
        return mean_s, logvar_s

    def decoder(self, s):
        po = self.po_net(s)
        return po

    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar


class ActiveInferenceModel:
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        if self.pi_dim > 0:
            self.model_top = ModelTop(s_dim, pi_dim).to(self.device)
            self.model_mid = ModelMid(s_dim, pi_dim).to(self.device)
        self.model_down = ModelDown(s_dim, pi_dim, colour_channels, resolution).to(self.device)

        self.beta_s = torch.tensor(beta_s, device=self.device)
        self.gamma = torch.tensor(gamma, device=self.device)
        self.beta_o = torch.tensor(beta_o, device=self.device)
        self.pi_one_hot = torch.eye(4, device=self.device)
        self.pi_one_hot_3 = torch.eye(3, device=self.device)

    def save_weights(self, folder_chp):
        torch.save(self.model_down.state_dict(), f"{folder_chp}/checkpoint_down.pth")
        if self.pi_dim > 0:
            torch.save(self.model_top.state_dict(), f"{folder_chp}/checkpoint_top.pth")
            torch.save(self.model_mid.state_dict(), f"{folder_chp}/checkpoint_mid.pth")

    def load_weights(self, folder_chp):
        self.model_down.load_state_dict(torch.load(f"{folder_chp}/checkpoint_down.pth"))
        if self.pi_dim > 0:
            self.model_top.load_state_dict(torch.load(f"{folder_chp}/checkpoint_top.pth"))
            self.model_mid.load_state_dict(torch.load(f"{folder_chp}/checkpoint_mid.pth"))

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        optimizers_for_pickle = {k: v.state_dict() for k, v in optimizers.items()}
        self.save_weights(folder_chp)
        with open(f"{folder_chp}/stats.pkl", "wb") as ff:
            pickle.dump(stats, ff)
        with open(f"{folder_chp}/optimizers.pkl", "wb") as ff:
            pickle.dump(optimizers_for_pickle, ff)
        copyfile("src/torchmodel.py", f"{folder_chp}/torchmodel.py")
        copyfile("src/torchloss.py", f"{folder_chp}/torchloss.py")
        if script_file:
            copyfile(script_file, f"{folder_chp}/{script_file}")

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(f"{folder_chp}/stats.pkl", "rb") as ff:
            stats = pickle.load(ff)
        try:
            with open(f"{folder_chp}/optimizers.pkl", "rb") as ff:
                optimizers = {k: torch.optim.Adam(self.parameters()) for k in pickle.load(ff)}
                for k, v in optimizers.items():
                    v.load_state_dict(pickle.load(ff)[k])
        except:
            optimizers = {}
        if stats["var_beta_s"]:
            self.beta_s = torch.tensor(stats["var_beta_s"][-1], device=self.device)
        if stats["var_gamma"]:
            self.gamma = torch.tensor(stats["var_gamma"][-1], device=self.device)
        if stats["var_beta_o"]:
            self.beta_o = torch.tensor(stats["var_beta_o"][-1], device=self.device)
        return stats, optimizers

    def check_reward(self, o):
        if self.model_down.resolution == 64:
            return torch.mean(calc_reward(o), dim=[1, 2, 3]) * 10.0
        elif self.model_down.resolution == 32:
            return torch.sum(calc_reward_animalai(o), dim=[1, 2, 3])

    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi

    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [torch.zeros(o.shape[0], device=self.device) for _ in range(3)]
        sum_G = torch.zeros(o.shape[0], device=self.device)

        s0_temp = qs0_mean if calc_mean else qs0

        for _ in range(steps):
            G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)

            for i in range(3):
                sum_terms[i] += terms[i]
            sum_G += G

            s0_temp = ps1_mean if calc_mean else s1

        return sum_G, sum_terms, po1

    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [torch.zeros(4, device=self.device) for _ in range(3)]
        sum_G = torch.zeros(4, device=self.device)

        s0_temp = qs0_mean if calc_mean else qs0

        for _ in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp, self.pi_one_hot)
            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, self.pi_one_hot, samples=samples)

            for i in range(3):
                sum_terms[i] += terms[i]
            sum_G += G

            s0_temp = ps1_mean if calc_mean else s1

        return sum_G, sum_terms, po1

    def calculate_G(self, s0, pi0, samples=10):
        term0 = torch.zeros(s0.shape[0], device=self.device)
        term1 = torch.zeros(s0.shape[0], device=self.device)
        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

            logpo1 = self.check_reward(po1)
            term0 += logpo1

            term1 += -torch.sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), dim=1)
        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = torch.zeros(s0.shape[0], device=self.device)
        term2_2 = torch.zeros(s0.shape[0], device=self.device)
        for _ in range(samples):
            po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[0])
            term2_1 += torch.sum(entropy_bernoulli(po1_temp1), dim=[1, 2, 3])

            po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2_2 += torch.sum(entropy_bernoulli(po1_temp2), dim=[1, 2, 3])
        term2_1 /= float(samples)
        term2_2 /= float(samples)

        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2

        return G, [term0, term1, term2], ps1, ps1_mean, po1

    def calculate_G_mean(self, s0, pi0):
        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1 = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        logpo1 = self.check_reward(po1)
        term0 = logpo1

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = -torch.sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), dim=1)

        # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
        term2_1 = torch.sum(entropy_bernoulli(po1_temp1), dim=[1, 2, 3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = torch.sum(entropy_bernoulli(po1_temp2), dim=[1, 2, 3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2

        return G, [term0, term1, term2], ps1_mean, po1

    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        po1 = self.model_down.decoder(ps1_traj)
        qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(po1)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = -torch.sum(entropy_normal_from_logvar(ps1_logvar_traj) + entropy_normal_from_logvar(qs1_logvar), dim=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = torch.sum(entropy_bernoulli(po1_temp1), dim=[1, 2, 3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = torch.sum(entropy_bernoulli(po1_temp2), dim=[1, 2, 3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return -term0 + term1 + term2

    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = torch.zeros((depth, self.s_dim), dtype=self.precision, device=self.device)
        ps1 = torch.zeros((depth, self.s_dim), dtype=self.precision, device=self.device)
        ps1_mean = torch.zeros((depth, self.s_dim), dtype=self.precision, device=self.device)
        ps1_logvar = torch.zeros((depth, self.s_dim), dtype=self.precision, device=self.device)
        pi0 = torch.zeros((depth, self.pi_dim), dtype=self.precision, device=self.device)

        s0[0] = starting_s
        try:
            Qpi_t_to_return = self.model_top.encode_s(s0[0].unsqueeze(0))[1][0]
            pi0[0, torch.multinomial(Qpi_t_to_return, 1)] = 1.0
        except:
            pi0[0, 0] = 1.0
            Qpi_t_to_return = pi0[0]
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[0].unsqueeze(0), s0[0].unsqueeze(0))
        ps1[0] = ps1_new[0]
        ps1_mean[0] = ps1_mean_new[0]
        ps1_logvar[0] = ps1_logvar_new[0]
        if 1 < depth:
            if use_means:
                s0[1] = ps1_mean_new[0]
            else:
                s0[1] = ps1_new[0]
        for t in range(1, depth):
            try:
                pi0[t, torch.multinomial(self.model_top.encode_s(s0[t].unsqueeze(0))[1][0], 1)] = 1.0
            except:
                pi0[t, 0] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[t].unsqueeze(0), s0[t].unsqueeze(0))
            ps1[t] = ps1_new[0]
            ps1_mean[t] = ps1_mean_new[0]
            ps1_logvar[t] = ps1_logvar_new[0]
            if t + 1 < depth:
                if use_means:
                    s0[t + 1] = ps1_mean_new[0]
                else:
                    s0[t + 1] = ps1_new[0]

        G = torch.mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).item()
        return G, pi0, Qpi_t_to_return
