import os, time, numpy as np, argparse, matplotlib.pyplot as plt
from sys import argv
from distutils.dir_util import copy_tree
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom libraries
from src.game_environment import Game
import src.util as u
import src.torchloss as loss
from src.torchmodel import ActiveInferenceModel
from src.torchutils import *
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot

parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-r", "--resume", action="store_true", help="If this is used, the script tries to load existing weights and resume training.")
parser.add_argument("-b", "--batch", type=int, default=50, help="Select batch size.")
args = parser.parse_args()

"""
a: The sum a+d show the maximum value of omega
b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
c: This moves the steepness of the sigmoid
d: This is the minimum omega (when sigmoid is zero)
"""
var_a = 1.0
var_b = 25.0
var_c = 5.0
var_d = 1.5
s_dim = 10
pi_dim = 4
beta_s = 1.0
beta_o = 1.0
gamma = 0.0
gamma_rate = 0.01
gamma_max = 0.8
gamma_delay = 30
deepness = 1
samples = 1
repeats = 5
l_rate_top = 1e-04
l_rate_mid = 1e-04
l_rate_down = 0.001
ROUNDS = 1000
TEST_SIZE = 1000
epochs = 1000

signature = "final_model_"
signature += str(gamma_rate) + "_" + str(gamma_delay) + "_" + str(var_a) + "_" + str(args.batch) + "_" + str(s_dim) + "_" + str(repeats)
folder = "/content/drive/MyDrive/QuantumActiveInference/figs_" + signature
folder_chp = folder + "/checkpoints"

os.makedirs(folder, exist_ok=True)
os.makedirs(folder_chp, exist_ok=True)

games = Game(args.batch)
game_test = Game(1)
model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=gamma, beta_s=beta_s, beta_o=beta_o, colour_channels=1, resolution=64)

stats_start = {
    "F": [], "F_top": [], "F_mid": [], "F_down": [], "mse_o": [], "TC": [],
    "kl_div_s": [], "kl_div_s_anal": [], "omega": [], "learning_rate": [],
    "current_lr": [], "mse_r": [], "omega_std": [], "kl_div_pi": [],
    "kl_div_pi_min": [], "kl_div_pi_max": [], "kl_div_pi_med": [],
    "kl_div_pi_std": [], "kl_div_pi_anal": [], "deep_mse_o": [],
    "var_beta_o": [], "var_beta_s": [], "var_gamma": [], "var_a": [],
    "var_b": [], "var_c": [], "var_d": [], "kl_div_s_naive": [],
    "kl_div_s_naive_anal": [], "score": [], "train_scores_m": [],
    "train_scores_std": [], "train_scores_sem": [], "train_scores_min": [],
    "train_scores_max": [],
}

if args.resume:
    stats, optimizers = model.load_all(folder_chp)
    for k in stats_start.keys():
        if k not in stats:
            stats[k] = []
        while len(stats[k]) < len(stats["F"]):
            stats[k].append(0.0)
    start_epoch = len(stats["F"]) + 1
else:
    stats = stats_start
    start_epoch = 1
    optimizers = {}

if not optimizers:
    optimizers = {
        "top": optim.Adam(model.model_top.parameters(), lr=l_rate_top),
        "mid": optim.Adam(model.model_mid.parameters(), lr=l_rate_mid),
        "down": optim.Adam(model.model_down.parameters(), lr=l_rate_down)
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()
for epoch in range(start_epoch, epochs + 1):
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
        model.model_down.gamma += gamma_rate

    train_scores = np.zeros(ROUNDS)
    for i in range(ROUNDS):
        # -- MAKE TRAINING DATA FOR THIS BATCH ---------------------------------
        games.randomize_environment_all()
        o0, o1, pi0, log_Ppi = u.make_batch_dsprites_active_inference(games=games, model=model, deepness=deepness, samples=samples, calc_mean=True, repeats=repeats)
        o0, o1, pi0, log_Ppi = map(lambda x: torch.tensor(x).to(device), (o0, o1, pi0, log_Ppi))

        # -- TRAIN TOP LAYER ---------------------------------------------------
        qs0, _, _ = model.model_down.encoder_with_sample(o0)
        D_KL_pi = loss.train_model_top(model_top=model.model_top, s=qs0, log_Ppi=log_Ppi, optimizer=optimizers["top"])
        if isinstance(D_KL_pi, np.ndarray):
            D_KL_pi = torch.tensor(D_KL_pi)
        D_KL_pi = D_KL_pi.detach().cpu().numpy()

        current_omega = loss.compute_omega(D_KL_pi, a=var_a, b=var_b, c=var_c, d=var_d).reshape(-1, 1)
        current_omega = torch.tensor(current_omega).to(device)

        # -- TRAIN MIDDLE LAYER ------------------------------------------------
        qs1_mean, qs1_logvar = model.model_down.encoder(o1)
        ps1_mean, ps1_logvar = loss.train_model_mid(model_mid=model.model_mid, s0=qs0, qs1_mean=qs1_mean, qs1_logvar=qs1_logvar, Ppi_sampled=pi0, omega=current_omega, optimizer=optimizers["mid"])

        # -- TRAIN DOWN LAYER --------------------------------------------------
        loss.train_model_down(model_down=model.model_down, o1=o1, ps1_mean=ps1_mean, ps1_logvar=ps1_logvar, omega=current_omega, optimizer=optimizers["down"])

    if epoch % 2 == 0:
        model.save_all(folder_chp, stats, argv[0], optimizers=optimizers)
    if epoch % 25 == 0:
        # keep the checkpoints every 25 steps
        copy_tree(folder_chp, folder_chp + f"_epoch_{epoch}")
        os.remove(folder_chp + f"_epoch_{epoch}/optimizers.pkl")

    # Evaluation
    with torch.no_grad():
        o0, o1, pi0, S0_real, _ = u.make_batch_dsprites_random(game=game_test, index=0, size=TEST_SIZE, repeats=repeats)
        o0, o1, pi0, S0_real = map(lambda x: torch.tensor(x).to(device), (o0, o1, pi0, S0_real))
        log_Ppi = torch.log(pi0 + 1e-15)
        
        s0, _, _ = model.model_down.encoder_with_sample(o0)
        F_top, kl_div_pi, kl_div_pi_anal, Qpi = loss.compute_loss_top(model_top=model.model_top, s=s0, log_Ppi=log_Ppi)
        qs1_mean, qs1_logvar = model.model_down.encoder(o1)
        qs1 = model.model_down.reparameterize(qs1_mean, qs1_logvar)
        F_mid, loss_terms_mid, ps1, ps1_mean, ps1_logvar = loss.compute_loss_mid(model_mid=model.model_mid, s0=s0, Ppi_sampled=pi0, qs1_mean=qs1_mean, qs1_logvar=qs1_logvar, omega=var_a / 2.0 + var_d)
        F_down, loss_terms, po1, qs1 = loss.compute_loss_down(model_down=model.model_down, o1=o1, ps1_mean=ps1_mean, ps1_logvar=ps1_logvar, omega=var_a / 2.0 + var_d)

    # Update stats
    stats["F"].append((F_down + F_mid + F_top).mean().item())
    stats["F_top"].append(F_top.mean().item())
    stats["F_mid"].append(F_mid.mean().item())
    stats["F_down"].append(F_down.mean().item())
    stats["mse_o"].append(loss_terms[0].mean().item())
    stats["kl_div_s"].append(loss_terms[1].mean().item())
    stats["kl_div_s_anal"].append(loss_terms[2].mean(0).cpu().numpy())
    stats["kl_div_s_naive"].append(loss_terms[3].mean().item())
    stats["kl_div_s_naive_anal"].append(loss_terms[4].mean(0).cpu().numpy())
    stats["omega"].append(current_omega.mean().item())
    stats["omega_std"].append(current_omega.std().item())
    stats["kl_div_pi"].append(kl_div_pi.mean().item())
    stats["kl_div_pi_min"].append(kl_div_pi.min().item())
    stats["kl_div_pi_max"].append(kl_div_pi.max().item())
    stats["kl_div_pi_med"].append(kl_div_pi.median().item())
    stats["kl_div_pi_std"].append(kl_div_pi.std().item())
    stats["kl_div_pi_anal"].append(kl_div_pi_anal.mean(0).cpu().numpy())
    stats["var_beta_s"].append(model.model_down.beta_s.item())
    stats["var_gamma"].append(model.model_down.gamma.item())
    stats["var_beta_o"].append(model.model_down.beta_o.item())
    stats["var_a"].append(var_a)
    stats["var_b"].append(var_b)
    stats["var_c"].append(var_c)
    stats["var_d"].append(var_d)
    stats["TC"].append(total_correlation(qs1.cpu().numpy()).mean())
    stats["learning_rate"].append(optimizers["down"].param_groups[0]['lr'])
    stats["current_lr"].append(optimizers["down"].param_groups[0]['lr'])

    generate_traversals(model=model, s_dim=s_dim, s_sample=s0.cpu().numpy(), S_real=S0_real.cpu().numpy(), filenames=[f"{folder}/traversals_at_epoch_{epoch:04d}.png"], colour=False)
    reconstructions_plot(o0.cpu().numpy(), o1.cpu().numpy(), po1.cpu().numpy(), filename=f"{folder}/imagination_{signature}_{epoch}.png", colour=False)

    # Test how well the agent learnt the dynamics related to the reward..
    o0, o1, pi0 = u.make_batch_dsprites_random_reward_transitions(game=game_test, index=0, size=TEST_SIZE, repeats=repeats)
    o0, o1, pi0 = map(lambda x: torch.tensor(x).to(device), (o0, o1, pi0))
    po1 = model.imagine_future_from_o(o0, pi0)
    reconstructions_plot(o0.cpu().numpy(), o1.cpu().numpy(), po1.cpu().numpy(), filename=f"{folder}/reward_imagination_{signature}_{epoch}.png")
    mse_reward = u.compare_reward(o1=o1.cpu().numpy(), po1=po1.cpu().numpy())
    stats["mse_r"].append(mse_reward)
    stats_plot(stats, f"{folder}/1_result_{signature}")

    print(
        f"{epoch}, F: {stats['F'][-1]:.2f}, MSEo: {stats['mse_o'][-1]:.3f}, KLs: {stats['kl_div_s'][-1]:.2f}, "
        f"omega: {stats['omega'][-1]:.2f}+-{stats['omega_std'][-1]:.2f}, KLpi: {stats['kl_div_pi'][-1]:.2f}, "
        f"TC: {stats['TC'][-1]:.2f}, dur. {time.time() - start_time:.2f}s"
    )
    start_time = time.time()

#
