import tensorflow as tf
import numpy as np

log_2_pi = np.log(2.0*np.pi)

def kl_div_loss_analytically_from_logvar_and_precision(mu1, logvar1, mu2, logvar2, omega):
    return 0.5*(logvar2 - tf.math.log(omega) - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2) / omega) - 0.5

def kl_div_loss_analytically_from_logvar(mu1, logvar1, mu2, logvar2):
    return 0.5*(logvar2 - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2)) - 0.5

def kl_div_loss(mu1, var1, mu2, var2, axis=1):
    return tf.reduce_sum(kl_div_loss_analytically(mu1, var1, mu2, var2), axis)

log_2_pi_e = np.log(2.0*np.pi*np.e)
@tf.function
def entropy_normal_from_logvar(logvar):
    return 0.5*(log_2_pi_e + logvar)

def entropy_bernoulli(p, displacement = 0.00001):
    return - (1-p)*tf.math.log(displacement + 1 - p) - p*tf.math.log(displacement + p)

def log_bernoulli(x, p, displacement = 0.00001):
    return x*tf.math.log(displacement + p) + (1-x)*tf.math.log(displacement + 1 - p)

def calc_reward(o, resolution=64):
    perfect_reward = np.zeros((3,resolution,1),dtype=np.float32)
    perfect_reward[:,:int(resolution/2)] = 1.0
    return log_bernoulli(o[:,0:3,0:resolution,:], perfect_reward)

def total_correlation(data):
    Cov = np.cov(data.T)
    return 0.5*(np.log(np.diag(Cov)).sum() - np.linalg.slogdet(Cov)[1])
