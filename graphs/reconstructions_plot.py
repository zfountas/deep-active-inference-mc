import matplotlib.pyplot as plt
import numpy as np

def reconstructions_plot(o0, o1, po1, filename, colour=False):
    if colour:
        o0 = o0[:7,:,:]
        o1 = o1[:7,:,:]
        po1 = po1[:7,:,:]
    else:
        o0 = o0[:7,:,:,0]
        o1 = o1[:7,:,:,0]
        po1 = po1[:7,:,:,0]
    fig = plt.figure(figsize=(10,5))
    plt.subplot(3,1,1)
    if colour: plt.imshow(np.hstack(o0), vmin=0, vmax=1)
    else: plt.imshow(np.hstack(o0), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o0')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,2)
    if colour: plt.imshow(np.hstack(o1)) #, vmin=0, vmax=1)
    else: plt.imshow(np.hstack(o1), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o1')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,3)
    if colour: plt.imshow(np.hstack(po1), vmin=0, vmax=1)
    else: plt.imshow(np.hstack(po1), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o1 reconstr')
    plt.xticks([])
    plt.yticks([])
    fig.set_tight_layout(True)
    plt.savefig(filename)
    plt.close()
