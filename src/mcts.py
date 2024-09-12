import os
import argparse
import pickle
import time
import cv2
import numpy as np
import torch
from src.torchmodel import ActiveInferenceModel

NODE_ID = 0
class Node:
    def __init__(self, s, model, C, pi_dim=4, verbose=False, using_prior_for_exploration=False):
        global NODE_ID

        # The latent state that corresponds to this node!
        self.pi_dim = pi_dim
        self.s = torch.stack([s]*self.pi_dim) # NOTE: It's saved self.pi_dim times to simplify calculations.
        self.model = model
        self.verbose = verbose
        self.using_prior_for_exploration = using_prior_for_exploration
        self.visited = False # For visualization
        self.NODE_ID = NODE_ID
        NODE_ID += 1

        if self.verbose: print('NEW NODE:', self.s.shape, self.NODE_ID)

        self.W = torch.zeros(self.pi_dim) # The total value of G so far in the next state given an edge..
        self.N = torch.zeros(self.pi_dim) # The total number of times an action was explored from here..
        self.Qpi = torch.zeros(self.pi_dim) # Prior probability distribution for actions..
        self.children_nodes = [None for _ in range(self.pi_dim)]
        self.C = C # A constant in AlphaGo Zero... [it was 1.0 in that paper..]
        # This takes the index of the child node that is currently under investigation.
        # It's important for the back-propagation of G, to remember which action was taken!
        self.in_progress = -1

    def Q(self):
        return self.W/self.N

    def probs_for_selection(self):
        # Initially normalize Q() to make it a distribution.
        Qnormed = self.Q()
        Qnormed -= Qnormed.min()
        Qnormed = Qnormed/Qnormed.sum()
        if self.using_prior_for_exploration:
            return Qnormed + self.C * self.Qpi * 1.0/(self.N)
        else:
            return Qnormed + self.C * 1.0/(self.N)

    def select(self, deterministic=True):
        path = []
        actions_path = []
        if deterministic: self.in_progress = torch.argmax(self.probs_for_selection())
        else: self.in_progress = torch.multinomial(self.probs_for_selection(), 1).item()
        actions_path.append(self.in_progress)
        path.append(self.children_nodes[self.in_progress])
        while None not in path[-1].children_nodes: # If not leaf!
            if deterministic: path[-1].in_progress = torch.argmax(path[-1].probs_for_selection())
            else: path[-1].in_progress = torch.multinomial(path[-1].probs_for_selection(), 1).item()
            actions_path.append(path[-1].in_progress)
            path.append(path[-1].children_nodes[path[-1].in_progress])
        if self.verbose: print('Select:', [p.NODE_ID for p in path], actions_path)
        return path, actions_path

    def expand(self, use_means=False, samples=1):
        """
        Note: It works if 'self' is a leaf.
        Expanding assigning a state and an initial expected free energy in all
        children nodes of the current leaf!
        """
        if self.pi_dim == 4:
            PI_HOT = self.model.pi_one_hot
        elif self.pi_dim == 3:
            PI_HOT = self.model.pi_one_hot_3
        else:
            raise ValueError(f'Errr: {self.pi_dim}')

        if use_means:
            G, _, ps_next_mean, _ = self.model.calculate_G_mean(self.s, PI_HOT)
            ps_next = ps_next_mean
        else:
            G, _, ps_next, _, _ = self.model.calculate_G(self.s, PI_HOT, samples=samples)
        self.W -= G.detach() # Note: Negative expected free energy to be used as a Q value in RL applications.
        self.N += 1.0
        for i in range(self.pi_dim):
            self.children_nodes[i] = Node(s=ps_next[i], model=self.model, C=self.C, pi_dim=self.pi_dim, using_prior_for_exploration=self.using_prior_for_exploration)
        if self.verbose: print('Expand:', G.detach(), self.W, self.N)

    def backpropagate(self, path, G):
        if self.verbose: print('Back-propagate:', [p.NODE_ID for p in path], G.detach())
        for i in range(len(path)):
            if path[i].in_progress < 0:
                raise ValueError(f'Back-propagation error: {path} {i}')
            path[i].W[path[i].in_progress] -= G
            path[i].N[path[i].in_progress] += 1
            path[i].in_progress = -2 # just to remember it's been examined..
            if self.verbose: print('Propagating to node', path[i].NODE_ID, 'with N:', path[i].N)

    def action_selection(self, deterministic=True):
        path = []
        if deterministic: path.append(torch.argmax(self.N).item())
        else: path.append(torch.multinomial(self.N.float(), 1).item())
        node = self.children_nodes[path[-1]]
        if self.verbose: print(len(path), node.NODE_ID)
        while None not in node.children_nodes:
            if deterministic: path.append(torch.argmax(node.N).item())
            else: path.append(torch.multinomial(node.N.float(), 1).item())
            node = node.children_nodes[path[-1]]
            if self.verbose: print(len(path), node.NODE_ID)

        trimmed_path = []
        i = 0
        while i < len(path)-1:
            if self.pi_dim == 4:
                if (path[i] == 0 and path[i+1] == 1) or (path[i] == 1 and path[i+1] == 0) or (path[i] == 2 and path[i+1] == 3) or (path[i] == 3 and path[i+1] == 2):
                    i += 2
                else:
                    trimmed_path.append(path[i])
                    i += 1
            elif self.pi_dim == 3:
                if (path[i] == 1 and path[i+1] == 2) or (path[i] == 2 and path[i+1] == 1):
                    i += 2
                else:
                    trimmed_path.append(path[i])
                    i += 1
            else:
                raise ValueError(f'Error: Unknown number of pi_dim {self.pi_dim}')
        if self.verbose: print('Action selection:', path, 'trimmed path:', trimmed_path)
        return trimmed_path

def calc_threshold(P, axis):
    return torch.max(P, dim=axis).values - torch.mean(P, dim=axis)

def normalization(x, tau=1):
    '''Compute softmax values for each sets of scores in x.'''
    return x / x.sum(dim=0)

class MCTS_Params:
    def __init__(self):
        self.C = 1.0
        self.threshold = 0.5
        self.repeats = 300
        self.simulation_repeats = 1
        self.simulation_depth = 3
        self.use_habit = False
        self.use_means = True
        self.verbose = False
        self.method = 'ai'
        self.using_prior_for_exploration = False

def active_inference_mcts(model, frame, params, o_shape=(64,64,1)):
    states_explored = 0
    all_paths = [] # For debugging.
    all_paths_G = [] # For debugging.
    if frame == []:
        return [0], 0, states_explored, all_paths, all_paths_G

    # Calculate current s_t
    qs0_mean, qs0_logvar = model.model_down.encoder(frame.reshape(1, *o_shape))

    # Important to be the mean here as we repeat it model.pi_dim times!
    root = Node(s=qs0_mean[0], model=model, C=params.C, pi_dim=model.pi_dim, using_prior_for_exploration=params.using_prior_for_exploration)

    # Habit.
    root.Qpi = model.model_top.encode_s(qs0_mean)[1][0]

    if params.use_habit:
        if calc_threshold(root.Qpi, axis=0) > params.threshold:
            if params.verbose: print('Decision in phase A Qpi:', root.Qpi, calc_threshold(root.Qpi, axis=0))
            MCTS_choices = root.Qpi
            return [torch.multinomial(root.Qpi, 1).item()], 0, states_explored, all_paths, all_paths_G

    root.expand(use_means=params.use_means)

    for repeat in range(params.repeats):

        if calc_threshold(normalization(root.N), axis=0) > params.threshold:
            MCTS_choices = normalization(root.N)
            if params.verbose: print('Decision in phase B', torch.round(root.probs_for_selection(), decimals=2), torch.round(MCTS_choices, decimals=2),
                                   calc_threshold(MCTS_choices, axis=0), 'N:', root.N)
            final_path = root.action_selection(deterministic=True)
            return final_path, repeat, states_explored, all_paths, all_paths_G

        path, actions_path = root.select(deterministic=True) # Path[-1] is a leaf node!
        path[-1].expand(use_means=params.use_means)
        all_av_G = torch.zeros(params.simulation_repeats)
        for sim_repeat in range(params.simulation_repeats):
            states_explored += params.simulation_depth
            all_av_G[sim_repeat], pi0, path[-1].Qpi = model.mcts_step_simulate(path[-1].s[0], params.simulation_depth, use_means=False)
        path[-1].backpropagate([root] + path[:-1], all_av_G.mean())
        all_paths.append(actions_path)
        all_paths_G.append(all_av_G.mean().item())

    final_path = root.action_selection(deterministic=True)
    if params.verbose: print('Decision in phase C', root.N, len(all_paths), torch.round(root.Q()-torch.min(root.Q()), decimals=2), torch.round(root.C * root.Qpi * torch.sqrt(root.N.sum())/(1.0+root.N), decimals=2), 'Qpi:', torch.round(root.Qpi, decimals=2))
    return final_path, params.repeats, states_explored, all_paths, all_paths_G
