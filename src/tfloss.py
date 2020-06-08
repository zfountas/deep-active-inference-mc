import tensorflow as tf
import numpy as np

from src.tfutils import *

def compute_omega(kl_pi, a, b, c, d):
    return a * ( 1.0 - 1.0/(1.0 + np.exp(- (kl_pi-b) / c)) ) + d

@tf.function
def compute_kl_div_pi(model, o0, log_Ppi):
    qs0 = model.model_down.encode_o_and_sample_s(o0)
    _, Qpi, log_Qpi = model.model_top.encode_s(qs0)

    # TERM: Eqs D_kl[Q(pi|s1,s0)||P(pi)], Categorical K-L divergence
    # --------------------------------------------------------------------------
    return tf.reduce_sum(Qpi*(log_Qpi-log_Ppi), 1)

@tf.function
def compute_loss_top(model_top, s, log_Ppi):
    _, Qpi, log_Qpi = model_top.encode_s(s)

    # TERM: Eqs D_kl[Q(pi|s0)||P(pi)], where P(pi) = U(0,5) - Categorical K-L divergence
    # ----------------------------------------------------------------------
    kl_div_pi_anal = Qpi*(log_Qpi - log_Ppi)
    kl_div_pi = tf.reduce_sum(kl_div_pi_anal, 1)

    F_top = kl_div_pi
    return F_top, kl_div_pi, kl_div_pi_anal, Qpi

@tf.function
def compute_loss_mid(model_mid, s0, Ppi_sampled, qs1_mean, qs1_logvar, omega):
    ps1, ps1_mean, ps1_logvar = model_mid.transition_with_sample(Ppi_sampled, s0)

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    F_mid = kl_div_s
    loss_terms = (kl_div_s, kl_div_s_anal)
    return F_mid, loss_terms, ps1, ps1_mean, ps1_logvar

@tf.function
def compute_loss_down(model_down, o1, ps1_mean, ps1_logvar, omega, displacement = 0.00001):
    qs1_mean, qs1_logvar = model_down.encoder(o1)
    qs1 = model_down.reparameterize(qs1_mean, qs1_logvar)
    po1 = model_down.decoder(qs1)

    # TERM: Eq[log P(o1|s1)]
    # --------------------------------------------------------------------------
    bin_cross_entr = o1 * tf.math.log(displacement + po1) + (1 - o1) * tf.math.log(displacement + 1 - po1) # Binary Cross Entropy
    logpo1_s1 = tf.reduce_sum(bin_cross_entr, axis=[1,2,3])

    # TERM: Eqpi D_kl[Q(s1)||N(0.0,1.0)]
    # --------------------------------------------------------------------------
    kl_div_s_naive_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, 0.0, 0.0, omega)
    kl_div_s_naive = tf.reduce_sum(kl_div_s_naive_anal, 1)

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    if model_down.gamma <= 0.05:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*kl_div_s_naive
    elif model_down.gamma >= 0.95:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*kl_div_s
    else:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*(model_down.gamma*kl_div_s + (1.0-model_down.gamma)*kl_div_s_naive)
    loss_terms = (-logpo1_s1, kl_div_s, kl_div_s_anal, kl_div_s_naive, kl_div_s_naive_anal)
    return F, loss_terms, po1, qs1

@tf.function
def train_model_top(model_top, s, log_Ppi, optimizer):
    s_stopped = tf.stop_gradient(s)
    log_Ppi_stopped = tf.stop_gradient(log_Ppi)
    with tf.GradientTape() as tape:
        F, kl_pi, _, _ = compute_loss_top(model_top=model_top, s=s_stopped, log_Ppi=log_Ppi_stopped)
        gradients = tape.gradient(F, model_top.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_top.trainable_variables))
    return kl_pi

@tf.function
def train_model_mid(model_mid, s0, qs1_mean, qs1_logvar, Ppi_sampled, omega, optimizer):
    s0_stopped = tf.stop_gradient(s0)
    qs1_mean_stopped = tf.stop_gradient(qs1_mean)
    qs1_logvar_stopped = tf.stop_gradient(qs1_logvar)
    Ppi_sampled_stopped = tf.stop_gradient(Ppi_sampled)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, loss_terms, ps1, ps1_mean, ps1_logvar = compute_loss_mid(model_mid=model_mid, s0=s0_stopped, Ppi_sampled=Ppi_sampled_stopped, qs1_mean=qs1_mean_stopped, qs1_logvar=qs1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_mid.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_mid.trainable_variables))
    return ps1_mean, ps1_logvar

@tf.function
def train_model_down(model_down, o1, ps1_mean, ps1_logvar, omega, optimizer):
    ps1_mean_stopped = tf.stop_gradient(ps1_mean)
    ps1_logvar_stopped = tf.stop_gradient(ps1_logvar)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, _, _, _ = compute_loss_down(model_down=model_down, o1=o1, ps1_mean=ps1_mean_stopped, ps1_logvar=ps1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_down.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_down.trainable_variables))
