import os, time, numpy as np, argparse, matplotlib.pyplot as plt
from sys import argv
from distutils.dir_util import copy_tree
import torch
import torch.optim as optim

# Import custom libraries
from src.game_environment import Game
import src.util_causal as u
import src.torchloss_causal as loss
#from src.causal_model import CausalInferenceModel
from src.torchutils import *
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot

parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-r", "--resume", action="store_true", help="If this is used, the script tries to load existing weights and resume training.")
parser.add_argument("-b", "--batch", type=int, default=50, help="Select batch size.")
args = parser.parse_args()

# Hyperparameters
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
l_rate = 1e-04
ROUNDS = 1000
TEST_SIZE = 1000
epochs = 1000

signature = "causal_model_"
signature += str(gamma_rate) + "_" + str(gamma_delay) + "_" + str(var_a) + "_" + str(args.batch) + "_" + str(s_dim) + "_" + str(repeats)
folder = "/content/drive/MyDrive/QuantumActiveInference/figs_" + signature
folder_chp = folder + "/checkpoints"

os.makedirs(folder, exist_ok=True)
os.makedirs(folder_chp, exist_ok=True)

games = Game(args.batch)
game_test = Game(1)
model = CausalInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=gamma, beta_s=beta_s, beta_o=beta_o, colour_channels=1, resolution=64)

stats_start = {
    "F": [], "mse_o": [], "kl_div_s": [], "omega": [], "learning_rate": [], "current_lr": [], "mse_r": [], "omega_std": [], "var_beta_o": [], "var_beta_s": [], "var_gamma": [], "var_a": [], "var_b": [], "var_c": [], "var_d": [], "score": [], "train_scores_m": [], "train_scores_std": [], "train_scores_sem": [], "train_scores_min": [], "train_scores_max": [],
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
        "main": optim.Adam(model.parameters(), lr=l_rate)
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()
for epoch in range(start_epoch, epochs + 1):
    if epoch > gamma_delay and model.gamma < gamma_max:
        model.gamma += gamma_rate

    train_scores = np.zeros(ROUNDS)
    for i in range(ROUNDS):
        # -- MAKE TRAINING DATA FOR THIS BATCH ---------------------------------
        games.randomize_environment_all()
        o0, o1, pi0, log_Ppi = u.make_batch_dsprites_causal_inference(games=games, model=model, deepness=deepness, samples=samples, calc_mean=True, repeats=repeats)
        o0, o1, pi0, log_Ppi = map(lambda x: torch.tensor(x).to(device), (o0, o1, pi0, log_Ppi))

        # -- TRAIN MODEL -------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        x_recon, s = model(o0)
        loss_value = loss.compute_loss_causal(x_recon, o1, s, pi0, log_Ppi, model)
        loss_value.backward()
        optimizer.step()

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
        
        x_recon, s = model(o0)
        F, kl_div_s, omega = loss.compute_loss_causal(x_recon, o1, s, pi0, log_Ppi, model)

    # Update stats
    stats["F"].append(F.mean().item())
    stats["mse_o"].append(loss_terms[0].mean().item())
    stats["kl_div_s"].append(loss_terms[1].mean().item())
    stats["omega"].append(current_omega.mean().item())
    stats["omega_std"].append(current_omega.std().item())
    stats["var_beta_s"].append(model.beta_s.item())
    stats["var_gamma"].append(model.gamma.item())
    stats["var_beta_o"].append(model.beta_o.item())
    stats["var_a"].append(var_a)
    stats["var_b"].append(var_b)
    stats["var_c"].append(var_c)
    stats["var_d"].append(var_d)
    stats["learning_rate"].append(optimizers["main"].param_groups[0]['lr'])
    stats["current_lr"].append(optimizers["main"].param_groups[0]['lr'])

    generate_traversals(model=model, s_dim=s_dim, s_sample=s.cpu().numpy(), S_real=S0_real.cpu().numpy(), filenames=[f"{folder}/traversals_at_epoch_{epoch:04d}.png"], colour=False)
    reconstructions_plot(o0.cpu().numpy(), o1.cpu().numpy(), x_recon.cpu().numpy(), filename=f"{folder}/imagination_{signature}_{epoch}.png", colour=False)

    print(
        f"{epoch}, F: {stats['F'][-1]:.2f}, MSEo: {stats['mse_o'][-1]:.3f}, KLs: {stats['kl_div_s'][-1]:.2f}, "
        f"omega: {stats['omega'][-1]:.2f}+-{stats['omega_std'][-1]:.2f}, dur. {time.time() - start_time:.2f}s"
    )
    start_time = time.time()