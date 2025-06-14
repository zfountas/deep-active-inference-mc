import os, time, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

# Set PyTorch to use CPU if CUDA is not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-n", "--network", type=str, default="", required=True, help="The path of a checkpoint to be loaded.")
parser.add_argument("-m", "--mean", action="store_true", help="Whether expected free energy should be calculated using the mean instead of sampling.")
parser.add_argument("-d", "--duration", type=int, default=50001, help="Duration of experiment.")
parser.add_argument("-method", "--method", type=str, default="mcts", help="Pre-select method used by the agent for action selection. Available: t1, t12, ai, mcts or habit!")
parser.add_argument("-steps", "--steps", type=int, default=7, help="How many steps ahead the agent can imagine!")
parser.add_argument("-temp", "--temperature", type=float, default=1, help="Initialize testing routine!")
parser.add_argument("-jumps", "--jumps", type=int, default=5, help="Mental jumps: How many steps ahead the agent has learnt to predict in a single step!")
# MCTS
parser.add_argument("-C", "--C", type=float, help="MCTS parameter: C: Balance between exploration and exploitation.", default=1.0)
parser.add_argument("-repeats", "--repeats", type=int, help="MCTS parameter: Simulation repeats", default=300)
parser.add_argument("-threshold", "--threshold", type=float, help="MCTS parameter: Threshold to make decision prematurely", default=0.5)
parser.add_argument("-depth", "--depth", type=int, help="MCTS parameter: Simulation depth", default=3)
parser.add_argument("-no_habit", "--no_habit", action="store_true", help="MCTS parameter: Disable habitual control as a first choice of the MCTS algorithm.")

args = parser.parse_args()
if args.network[-1] in ["/", "\\"]:
    args.network = args.network[:-1]

from src.game_environment import Game
import src.util as u
from src.mcts import MCTS_Params, active_inference_mcts
from src.torchmodel import ActiveInferenceModel

params = MCTS_Params()
params.C = args.C
params.repeats = args.repeats
params.threshold = args.threshold
params.simulation_depth = args.depth
params.use_habit = args.no_habit

game = Game(1)  # Pass the number of games to the Game constructor

s_dim = 10
pi_dim = 4
BATCH_SIZE = 1

model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=1.0, beta_s=1.0, beta_o=1.0, colour_channels=1, resolution=64).to(device)
model.load_all(args.network)

game.randomize_environment(0)
game.current_s[0, -1] = 0.0

pi0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
o0 = torch.from_numpy(game.current_frame(0).reshape(1, 1, 64, 64)).float().to(device)
qs0_mean, qs0_logvar = model.model_down.encoder(o0)
s0 = model.model_down.reparameterize(qs0_mean, qs0_logvar)

duration_of_experiment = 1000
duration_of_round = 100
CURRENT_STATES = np.zeros((duration_of_experiment, game.current_s.shape[1]))
last_pi = None

G = np.zeros(4)
term0 = np.zeros(4)
term1 = np.zeros(4)
term2 = np.zeros(4)

executing_steps = []
if args.method in ["t1", "t12", "ai", "habit"]:
    if args.steps == -1:
        args.steps = 10
    samples = 10
else:
    if args.steps == -1:
        args.steps = 1
    samples = 1

COLOR = False


def softmax(x, temp):
    e_x = np.exp(x / temp)
    return e_x / e_x.sum(axis=0)


def make_mask(all_paths, pos_x, pos_y):
    mask = np.zeros((32, 32))
    for path in all_paths:
        turtle_x = pos_x
        turtle_y = pos_y
        for p_i in path:
            if p_i == 0:  # up
                for _ in range(args.jumps):
                    if turtle_x < 31:
                        turtle_x += 1
                        mask[turtle_x, turtle_y] += 1.0
            elif p_i == 1:  # down
                for _ in range(args.jumps):
                    if turtle_x > 0:
                        turtle_x -= 1
                        mask[turtle_x, turtle_y] += 1.0
            elif p_i == 2:  # left
                for _ in range(args.jumps):
                    if turtle_y < 31:
                        turtle_y += 1
                        mask[turtle_x, turtle_y] += 1.0
            elif p_i == 3:  # right
                for _ in range(args.jumps):
                    if turtle_y > 0:
                        turtle_y -= 1
                        mask[turtle_x, turtle_y] += 1.0
    return mask / mask.max()


start_time = time.time()
t = 0
while t < args.duration:
    CURRENT_STATES[int(t % duration_of_experiment)] = game.current_s[0]
    if args.method in ["t1", "t12", "ai", "mcts", "habit"]:
        if (t % duration_of_experiment) == 0 and t > 0:
            print(t, "ROUND SCORE:", game.get_reward(0), "t:", time.time() - start_time)
            game.current_s[0, 6] = 0.0
            start_time = time.time()
        if (t % duration_of_round) == 0:
            temp_score = game.current_s[0, 6]
            game.randomize_environment(0)
            game.current_s[0, 6] = temp_score
            executing_steps = []

    if len(executing_steps) == 0:
        # Get observation from the environment
        o_single = game.current_frame(0)

        if args.method == "habit":
            qs_mean, _ = model.model_down.encoder(torch.from_numpy(o_single).unsqueeze(0).unsqueeze(0).float().to(device))
            _, Qpi, _ = model.model_top.encode_s(qs_mean)
            Qpi_choices = Qpi.cpu().numpy()[0]
            G_choices = [0.0, 0.0, 0.0, 0.0]
            R_choices = [0.0, 0.0, 0.0, 0.0]
        elif args.method == "mcts":
            mcts_path, repeats_done, states_explored, all_paths, all_paths_G = active_inference_mcts(model=model, frame=o_single, params=params, o_shape=(1, 64, 64))
            path_pos_x = int(game.current_s[0, 5])
            path_pos_y = int(game.current_s[0, 4])
            mask = make_mask(all_paths, path_pos_x, path_pos_y)
            G = term0 = term1 = term2 = np.zeros(4)
            R_choices = term12_choices = G_choices = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            o1 = torch.from_numpy(np.repeat(o_single[np.newaxis, np.newaxis, :, :], 4, axis=0)).float().to(device)
            sum_G, sum_terms, po2 = model.calculate_G_4_repeated(o1, steps=args.steps, samples=samples, calc_mean=args.mean)

            G = sum_G.cpu().numpy() / float(args.steps)
            term0 = -sum_terms[0].cpu().numpy() / float(args.steps)
            term1 = sum_terms[1].cpu().numpy() / float(args.steps)
            term2 = sum_terms[2].cpu().numpy() / float(args.steps)

            R_choices = softmax(-term0, args.temperature)
            term12_choices = softmax(-(term0 + term1), args.temperature)
            G_choices = softmax(-G, args.temperature)

        try:
            if args.method == "ai":
                pi = np.random.choice(4, p=G_choices)
                for _ in range(args.steps):
                    for _ in range(args.jumps):
                        executing_steps.append(pi)
            if args.method == "mcts":
                for pp in mcts_path:
                    for _ in range(args.jumps):
                        executing_steps.append(pp)
            elif args.method == "t12":
                pi = np.random.choice(4, p=term12_choices)
                for _ in range(args.steps):
                    for _ in range(args.jumps):
                        executing_steps.append(pi)
            elif args.method == "t1":
                pi = np.random.choice(4, p=R_choices)
                for _ in range(args.steps):
                    for _ in range(args.jumps):
                        executing_steps.append(pi)
            elif args.method == "habit":
                pi = np.random.choice(4, p=Qpi_choices)
                for _ in range(args.steps):
                    executing_steps.append(pi)
        except:
            print("Not executing anything")
            executing_steps = []

    if len(executing_steps) > 0:
        pi = executing_steps[0]
        changed = False
        if pi == 0:
            changed = game.up(0)
        if pi == 1:
            game.down(0)
        if pi == 2:
            game.left(0)
        if pi == 3:
            game.right(0)
        if changed:
            executing_steps = []
        else:
            # pop front..
            executing_steps = executing_steps[1:]

    frame = game.current_frame(0)
    frame[59:63, 31] = 1.0

    if args.method == "mcts":
        frame[16:48, 16:48] += mask.reshape(32, 32, 1)

    if COLOR:
        frame = Image.fromarray((frame * 255).astype(np.uint8)).convert('RGB')
    else:
        frame = Image.fromarray((frame * 255).astype(np.uint8))
    frame = T.Resize((500, 500), interpolation=T.InterpolationMode.NEAREST)(frame)
    frame = T.ToTensor()(frame)

    plt.clf()
    plt.imshow(frame.permute(1, 2, 0))
    plt.text(15, 25, f"score: {game.get_reward(0)} ({float(duration_of_experiment) * game.get_reward(0) / float(t)})", fontsize=10, color='white')
    plt.text(15, 50, f"s: {game.current_s[0]}", fontsize=8, color='white')
    if args.method != "mcts":
        plt.text(15, 70, f"G: {np.around(G, 2)}", fontsize=8, color='white')
        plt.text(15, 100, f"Term a: {np.around(term0 - term0.min(), 2)}", fontsize=8, color='white')
        plt.text(15, 120, f"Term b: {np.around(term1 - term1.min(), 2)}", fontsize=8, color='white')
        plt.text(15, 140, f"Term c: {np.around(term2 - term2.min(), 2)}", fontsize=8, color='white')
        plt.text(15, 170, f"softmax(term a):   {np.around(R_choices, 2)}", fontsize=8, color='white')
        plt.text(15, 190, f"softmax(terms a+b): {np.around(term12_choices, 2)}", fontsize=8, color='white')
        plt.text(15, 210, f"softmax(G):   {np.around(G_choices, 2)}", fontsize=8, color='white')
    plt.pause(0.001)

    # -- KEYBOARD SHORTCUTS ------------------------------------------------
    k = input("Press a key (q to quit): ")
    if k == 'q':
        break
    elif k == 'm':
        args.mean = not args.mean
        print("Using mean:", args.mean)
    elif k == 's':
        last_pi = 0
        game.up(0)
    elif k == 'w':
        last_pi = 1
        game.down(0)
    elif k == 'd':
        last_pi = 2
        game.left(0)
    elif k == 'a':
        last_pi = 3
        game.right(0)
    elif k == 'r':
        game.current_s[0, 6] = 0.0
        t = 0
        print("Restart scoring")
    elif k == '1':
        args.method = "mcts"
        print("Active inference with full-scale planner in control (all terms of G used)")
    elif k == '2':
        args.method = "ai"
        print("1-step active inference in control (all terms of G used)")
    elif k == '3':
        args.method = "habit"
        print("Habitual mode")
    elif k == '4':
        args.method = "no"
        print("Stopped. You can control the agent now!")
    elif k == '5':
        args.method = "t1"
        print("Term a in control (reward-based agent)")
    elif k == '6':
        args.method = "t12"
        print("Terms a+b in control")
    elif k == 'o' or k == '[':
        if args.steps > 1:
            args.steps -= 1
        print("STEPS", args.steps)
    elif k == 'p' or k == ']':
        args.steps += 1
        print("STEPS", args.steps)
    elif k == '8':
        if args.temperature > 5.0:
            args.temperature -= 5.0
        print("Temperature for softmax:", args.temperature)
    elif k == '9':
        args.temperature += 5.0
        print("Temperature for softmax:", args.temperature)

    t += 1

plt.close()

exit("Exiting ok...!")


#
