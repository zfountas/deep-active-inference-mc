import os, time, numpy as np, argparse, matplotlib.pyplot as plt, scipy
from sys import argv
from distutils.dir_util import copy_tree
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import custom libraries
from src.game_environment import Game
import src.util as u
import src.tfloss as loss
from src.tfmodel import ActiveInferenceModel
from src.tfutils import *
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-r', '--resume', action='store_true', help='If this is used, the script tries to load existing weights and resume training.')
parser.add_argument('-b', '--batch', type=int, default=50, help='Select batch size.')
args = parser.parse_args()

'''
a: The sum a+d show the maximum value of omega
b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
c: This moves the steepness of the sigmoid
d: This is the minimum omega (when sigmoid is zero)
'''
var_a = 1.0;         var_b = 25.0;          var_c = 5.0;         var_d = 1.5
s_dim = 10;          pi_dim = 4;            beta_s = 1.0;        beta_o = 1.0;
gamma = 0.0;         gamma_rate = 0.01;     gamma_max = 0.8;     gamma_delay = 30
deepness = 1;        samples = 1;           repeats = 5
l_rate_top = 1e-04;  l_rate_mid = 1e-04;    l_rate_down = 0.001
ROUNDS = 1000;       TEST_SIZE = 1000;      epochs = 1000

signature = 'final_model_'
signature += str(gamma_rate)+'_'+str(gamma_delay)+'_'+str(var_a)+'_'+str(args.batch)+'_'+str(s_dim)+'_'+str(repeats)
folder = 'figs_'+signature
folder_chp = folder + '/checkpoints'

try: os.mkdir(folder)
except: print('Folder already exists!!')
try: os.mkdir(folder_chp)
except: print('Folder chp creation error')

games = Game(args.batch)
game_test = Game(1)
model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=gamma, beta_s=beta_s, beta_o=beta_o, colour_channels=1, resolution=64)

stats_start = {'F':[], 'F_top':[], 'F_mid':[], 'F_down':[], 'mse_o':[], 'TC':[], 'kl_div_s':[],
   'kl_div_s_anal':[], 'omega':[], 'learning_rate':[], 'current_lr':[], 'mse_r':[],
   'omega_std':[], 'kl_div_pi':[], 'kl_div_pi_min':[], 'kl_div_pi_max':[],
   'kl_div_pi_med':[], 'kl_div_pi_std':[], 'kl_div_pi_anal':[], 'deep_mse_o':[],
   'var_beta_o':[], 'var_beta_s':[], 'var_gamma':[], 'var_a':[], 'var_b':[],
   'var_c':[], 'var_d':[], 'kl_div_s_naive':[], 'kl_div_s_naive_anal':[], 'score':[],
   'train_scores_m':[],'train_scores_std':[],'train_scores_sem':[],'train_scores_min':[],'train_scores_max':[]}

if args.resume:
    stats, optimizers = model.load_all(folder_chp)
    for k in stats_start.keys():
        if k not in stats:
            stats[k] = []
        while len(stats[k]) < len(stats['F']):
            stats[k].append(0.0)
    start_epoch = len(stats['F']) + 1
else:
    stats = stats_start
    start_epoch = 1
    optimizers = {}

if optimizers == {}:
    optimizers['top'] = tf.keras.optimizers.Adam(learning_rate=l_rate_top)
    optimizers['mid'] = tf.keras.optimizers.Adam(learning_rate=l_rate_mid)
    optimizers['down'] = tf.keras.optimizers.Adam(learning_rate=l_rate_down)

start_time = time.time()
for epoch in range(start_epoch, epochs + 1):
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
            model.model_down.gamma.assign(model.model_down.gamma+gamma_rate)

    train_scores = np.zeros(ROUNDS)
    for i in range(ROUNDS):
        # -- MAKE TRAINING DATA FOR THIS BATCH ---------------------------------
        games.randomize_environment_all()
        o0, o1, pi0, log_Ppi = u.make_batch_dsprites_active_inference(games=games, model=model, deepness=deepness, samples=samples, calc_mean=True, repeats=repeats)

        # -- TRAIN TOP LAYER ---------------------------------------------------
        qs0,_,_ = model.model_down.encoder_with_sample(o0)
        D_KL_pi = loss.train_model_top(model_top=model.model_top, s=qs0, log_Ppi=log_Ppi, optimizer=optimizers['top'])
        D_KL_pi = D_KL_pi.numpy()

        current_omega = loss.compute_omega(D_KL_pi, a=var_a, b=var_b, c=var_c, d=var_d).reshape(-1,1)

        # -- TRAIN MIDDLE LAYER ------------------------------------------------
        qs1_mean, qs1_logvar = model.model_down.encoder(o1)
        ps1_mean, ps1_logvar = loss.train_model_mid(model_mid=model.model_mid, s0=qs0, qs1_mean=qs1_mean, qs1_logvar=qs1_logvar, Ppi_sampled=pi0, omega=current_omega, optimizer=optimizers['mid'])

        # -- TRAIN DOWN LAYER --------------------------------------------------
        loss.train_model_down(model_down=model.model_down, o1=o1, ps1_mean=ps1_mean, ps1_logvar=ps1_logvar, omega=current_omega, optimizer=optimizers['down'])

    if epoch % 2 == 0:
        model.save_all(folder_chp, stats, argv[0], optimizers=optimizers)
    if epoch % 2 == 25:
        # keep the checkpoints every 25 steps
        copy_tree(folder_chp, folder_chp+'_epoch_'+str(epoch))
        os.remove(folder_chp+'_epoch_'+str(epoch)+'/optimizers.pkl')

    o0, o1, pi0, S0_real, _ = u.make_batch_dsprites_random(game=game_test, index=0, size=TEST_SIZE, repeats=repeats)
    log_Ppi = np.log(pi0+1e-15)
    s0,_,_ = model.model_down.encoder_with_sample(o0)
    F_top, kl_div_pi, kl_div_pi_anal, Qpi = loss.compute_loss_top(model_top=model.model_top, s=s0, log_Ppi=log_Ppi)
    qs1_mean, qs1_logvar = model.model_down.encoder(o1)
    qs1 = model.model_down.reparameterize(qs1_mean, qs1_logvar)
    F_mid, loss_terms_mid, ps1, ps1_mean, ps1_logvar = loss.compute_loss_mid(model_mid=model.model_mid, s0=s0, Ppi_sampled=pi0, qs1_mean=qs1_mean, qs1_logvar=qs1_logvar, omega=var_a/2.0+var_d)
    F_down, loss_terms, po1, qs1 = loss.compute_loss_down(model_down=model.model_down, o1=o1, ps1_mean=ps1_mean, ps1_logvar=ps1_logvar, omega=var_a/2.0+var_d)
    stats['F'].append(np.mean(F_down) + np.mean(F_mid) + np.mean(F_top))
    stats['F_top'].append(np.mean(F_top))
    stats['F_mid'].append(np.mean(F_mid))
    stats['F_down'].append(np.mean(F_down))
    stats['mse_o'].append(np.mean(loss_terms[0]))
    stats['kl_div_s'].append(np.mean(loss_terms[1]))
    stats['kl_div_s_anal'].append(np.mean(loss_terms[2],axis=0))
    stats['kl_div_s_naive'].append(np.mean(loss_terms[3]))
    stats['kl_div_s_naive_anal'].append(np.mean(loss_terms[4],axis=0))
    stats['omega'].append(np.mean(current_omega))
    stats['omega_std'].append(np.std(current_omega))
    stats['kl_div_pi'].append(np.mean(kl_div_pi))
    stats['kl_div_pi_min'].append(np.min(kl_div_pi))
    stats['kl_div_pi_max'].append(np.max(kl_div_pi))
    stats['kl_div_pi_med'].append(np.median(kl_div_pi))
    stats['kl_div_pi_std'].append(np.std(kl_div_pi))
    stats['kl_div_pi_anal'].append(np.mean(kl_div_pi_anal,axis=0))
    stats['var_beta_s'].append(model.model_down.beta_s.numpy())
    stats['var_gamma'].append(model.model_down.gamma.numpy())
    stats['var_beta_o'].append(model.model_down.beta_o.numpy())
    stats['var_a'].append(var_a)
    stats['var_b'].append(var_b)
    stats['var_c'].append(var_c)
    stats['var_d'].append(var_d)
    stats['TC'].append(np.mean(total_correlation(qs1.numpy())))
    stats['learning_rate'].append(optimizers['down'].lr.numpy())
    stats['current_lr'].append(optimizers['down']._decayed_lr(tf.float32).numpy())

    generate_traversals(model=model, s_dim=s_dim, s_sample=s0, S_real=S0_real,
                        filenames=[folder+'/traversals_at_epoch_{:04d}.png'.format(epoch)], colour=False)
    reconstructions_plot(o0, o1, po1.numpy(), filename=folder+'/imagination_'+signature+'_'+str(epoch)+'.png', colour=False)

    # Test how well the agent learnt the dynamics related to the reward..
    o0,o1,pi0 = u.make_batch_dsprites_random_reward_transitions(game=game_test, index=0, size=TEST_SIZE, repeats=repeats)
    po1 = model.imagine_future_from_o(o0, pi0)
    reconstructions_plot(o0, o1, po1.numpy(), filename=folder+'/reward_imagination_'+signature+'_'+str(epoch)+'.png')
    mse_reward = u.compare_reward(o1=o1,po1=po1.numpy())
    stats['mse_r'].append(mse_reward)
    stats_plot(stats, folder+'/1_result_'+signature)

    print('{0}, F: {1:.2f}, MSEo: {2:.3f}, KLs: {3:.2f}, omega: {4:.2f}+-{5:.2f}, KLpi: {6:.2f}, TC: {7:.2f}, dur. {8}s'.format(epoch,
          stats['F'][-1], stats['mse_o'][-1], stats['kl_div_s'][-1], stats['omega'][-1], stats['omega_std'][-1], stats['kl_div_pi'][-1],
          stats['TC'][-1], round(time.time()-start_time,2)))
    start_time = time.time()














#
