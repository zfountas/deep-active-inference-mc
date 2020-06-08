import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from tensorflow import sigmoid
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_regression

def generate_traversals(model, s_dim, s_sample, S_real, filenames=[], naive=False, colour=False):
    elements = 10

    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(s_dim, 3, width_ratios=[5,1,1])
    arg_max_hist_value = np.zeros(s_dim)
    start_val = np.zeros(s_dim)
    end_val = np.zeros(s_dim)
    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3+1])
        hh = plt.hist(s_sample[:,s_indx])

        if naive:
            arg_max_hist_value[s_indx] = 0.0
            start_val[s_indx] = -3.0
            end_val[s_indx] = 3.0
        else:
            index_of_highest = np.argmax(hh[0])
            arg_max_hist_value[s_indx] = (hh[1][index_of_highest]+hh[1][index_of_highest+1])/2.0
            start_val[s_indx] = (hh[1][0]+hh[1][1])/2.0
            end_val[s_indx] = (hh[1][-2]+hh[1][-1])/2.0

    start_val = np.array([-5.0, -5.0, -2.0, -5.0, -1.3, -0.65, -2.0,  -2.5,  0.4, -2.5])
    arg_max_hist_value = np.array([-1.5,0.0,-1.5,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    #arg_max_hist_value = np.array([-1.5,0.0,-1.5,0.0,0.75,1.0,0.0,0.0,0.75,0.0]) # for s8
    end_val = np.array([4.0, 5.0, 2.0, 5.0, 4.75, 2.1, 2.0, 2.5, 3.45, 2.5])

    if len(S_real) > 0:
        correlations = np.zeros((10,6))
        correlations_cat = np.zeros((10,6))
        correlations_p = np.zeros((10,6))
        labels = ['shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        for real_s_indx in range(6):
            for s_indx in range(s_dim):
                correlations[s_indx,real_s_indx],correlations_p[s_indx,real_s_indx] = spearmanr(s_sample[:,s_indx],S_real[:,real_s_indx])
                correlations[s_indx,real_s_indx] = abs(correlations[s_indx,real_s_indx])
                correlations_cat[s_indx,real_s_indx] = mutual_info_regression(s_sample[:,s_indx].numpy().reshape(-1,1),S_real[:,real_s_indx])

        for s_indx in range(s_dim):
            plt.subplot(gs[s_indx*3+2])
            plt.plot(correlations[s_indx][1:])
            if np.max(correlations[s_indx][1:]) < 0.5:
                plt.ylim(0.0,0.5)
            plt.plot(correlations_cat[s_indx])
        plt.ylabel('Correlation')
        plt.xticks(range(len(labels)-1),labels[1:], rotation='vertical')

    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3])
        plt.ylabel(r'$s_{'+str(s_indx)+'}$')
        s = np.zeros((elements,s_dim))
        for x in range(elements):
            for y in range(s_dim):
                s[x,y] = arg_max_hist_value[y]

        for x,s_x in enumerate(np.linspace(start_val[s_indx],end_val[s_indx],elements)):
            s[x,s_indx] = s_x
        if colour:
            new_img = model.model_down.decoder(s)[:,:,:]
            plt.imshow(np.hstack(new_img), vmin=0, vmax=1)
        else:
            new_img = model.model_down.decoder(s)[:,:,:,0]
            plt.imshow(np.hstack(new_img), cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(round(start_val[s_indx],4))+' <-- '+str(round(arg_max_hist_value[s_indx],4))+' --> '+str(round(end_val[s_indx],4)))


    fig.set_tight_layout(True)
    for filename in filenames:
        plt.savefig(filename)
    #plt.show()
    plt.close()


#
