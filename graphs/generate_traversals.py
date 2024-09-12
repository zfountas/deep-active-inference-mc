import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torch.nn.functional import sigmoid
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

def generate_traversals(model, s_dim, s_sample, S_real, filenames=[], naive=False, colour=False):
    elements = 10

    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(s_dim, 3, width_ratios=[5,1,1])
    arg_max_hist_value = torch.zeros(s_dim)
    start_val = torch.zeros(s_dim)
    end_val = torch.zeros(s_dim)
    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3+1])
        hh = plt.hist(s_sample[:,s_indx].cpu().numpy())

        if naive:
            arg_max_hist_value[s_indx] = 0.0
            start_val[s_indx] = -3.0
            end_val[s_indx] = 3.0
        else:
            index_of_highest = torch.argmax(torch.tensor(hh[0]))
            arg_max_hist_value[s_indx] = (hh[1][index_of_highest]+hh[1][index_of_highest+1])/2.0
            start_val[s_indx] = (hh[1][0]+hh[1][1])/2.0
            end_val[s_indx] = (hh[1][-2]+hh[1][-1])/2.0

    start_val = torch.tensor([-5.0, -5.0, -2.0, -5.0, -1.3, -0.65, -2.0,  -2.5,  0.4, -2.5])
    arg_max_hist_value = torch.tensor([-1.5,0.0,-1.5,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    end_val = torch.tensor([4.0, 5.0, 2.0, 5.0, 4.75, 2.1, 2.0, 2.5, 3.45, 2.5])

    if len(S_real) > 0:
        correlations = torch.zeros((10,6))
        correlations_cat = torch.zeros((10,6))
        correlations_p = torch.zeros((10,6))
        labels = ['shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        for real_s_indx in range(6):
            for s_indx in range(s_dim):
                corr, p_value = spearmanr(s_sample[:,s_indx].cpu().numpy(), S_real[:,real_s_indx].cpu().numpy())
                correlations[s_indx,real_s_indx] = abs(corr)
                correlations_p[s_indx,real_s_indx] = p_value
                correlations_cat[s_indx,real_s_indx] = mutual_info_regression(s_sample[:,s_indx].cpu().numpy().reshape(-1,1), S_real[:,real_s_indx].cpu().numpy())

        for s_indx in range(s_dim):
            plt.subplot(gs[s_indx*3+2])
            sns.lineplot(data=correlations[s_indx][1:].cpu().numpy())
            if torch.max(correlations[s_indx][1:]) < 0.5:
                plt.ylim(0.0,0.5)
            sns.lineplot(data=correlations_cat[s_indx].cpu().numpy())
        plt.ylabel('Correlation')
        plt.xticks(range(len(labels)-1),labels[1:], rotation='vertical')

    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3])
        plt.ylabel(r'$s_{'+str(s_indx)+'}$')
        s = torch.zeros((elements,s_dim))
        for x in range(elements):
            for y in range(s_dim):
                s[x,y] = arg_max_hist_value[y]

        for x,s_x in enumerate(torch.linspace(start_val[s_indx],end_val[s_indx],elements)):
            s[x,s_indx] = s_x
        with torch.no_grad():
            new_img = model.model_down.decoder(s)
        if colour:
            plt.imshow(torch.hstack(new_img[:,:,:3]).cpu(), vmin=0, vmax=1)
        else:
            plt.imshow(torch.hstack(new_img[:,:,:,0]).cpu(), cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{start_val[s_indx]:.4f} <-- {arg_max_hist_value[s_indx]:.4f} --> {end_val[s_indx]:.4f}")

    fig.set_tight_layout(True)
    for filename in filenames:
        plt.savefig(filename)
    plt.close()
