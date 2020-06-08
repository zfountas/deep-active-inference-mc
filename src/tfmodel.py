import tensorflow as tf
import pickle
import numpy as np
from shutil import copyfile
from src.tfutils import *

class ModelTop(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelTop, self).__init__()
        # For activation function we used ReLU.
        # For weight initialization we used He Uniform

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.qpi_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(s_dim,)),
              tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dense(pi_dim),]) # No activation

    def encode_s(self, s0):
        logits_pi = self.qpi_net(s0)
        q_pi = tf.nn.softmax(logits_pi)
        log_q_pi = tf.math.log(q_pi+1e-20)
        return logits_pi, q_pi, log_q_pi

class ModelMid(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelMid, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.ps_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(pi_dim+s_dim,)),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(s_dim + s_dim),]) # No activation

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def transition(self, pi, s0):
        mean, logvar = tf.split(self.ps_net(tf.concat([pi,s0],1)), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar

class ModelDown(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision, colour_channels, resolution):
        super(ModelDown, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.colour_channels = colour_channels
        self.resolution = resolution
        if self.resolution == 64:
            last_strides = 2
        elif self.resolution == 32:
            last_strides = 1
        else:
            exit('Unknown resolution..')

        self.qs_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(self.resolution, self.resolution, self.colour_channels)),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(s_dim + s_dim),]) # No activation
        self.po_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(s_dim,)),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=16*16*64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Reshape(target_shape=(16, 16, 64)),
              tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(last_strides, last_strides), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=self.colour_channels, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid', kernel_initializer='he_uniform'),])

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def encoder(self, o):
        grad_check = tf.debugging.check_numerics(self.qs_net(o), 'check_numerics caught bad temptemptemptemptemp3')
        mean_s, logvar_s = tf.split(self.qs_net(o), num_or_size_splits=2, axis=1)
        return mean_s, logvar_s

    @tf.function
    def decoder(self, s):
        po = self.po_net(s)
        return po

    @tf.function
    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar


class ActiveInferenceModel:
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):

        self.tf_precision = tf.float32
        self.precision = 'float32'

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        tf.keras.backend.set_floatx(self.precision)

        if self.pi_dim > 0:
            self.model_top = ModelTop(s_dim, pi_dim, self.tf_precision, self.precision)
            self.model_mid = ModelMid(s_dim, pi_dim, self.tf_precision, self.precision)
        self.model_down = ModelDown(s_dim, pi_dim, self.tf_precision, self.precision, colour_channels, resolution)

        self.model_down.beta_s = tf.Variable(beta_s, trainable=False, name="beta_s")
        self.model_down.gamma = tf.Variable(gamma, trainable=False, name="gamma")
        self.model_down.beta_o = tf.Variable(beta_o, trainable=False, name="beta_o")
        self.pi_one_hot = tf.Variable([[1.0,0.0,0.0,0.0],
                                       [0.0,1.0,0.0,0.0],
                                       [0.0,0.0,1.0,0.0],
                                       [0.0,0.0,0.0,1.0]], trainable=False, dtype=self.tf_precision)
        self.pi_one_hot_3 = tf.Variable([[1.0,0.0,0.0],
                                         [0.0,1.0,0.0],
                                         [0.0,0.0,1.0]], trainable=False, dtype=self.tf_precision)


    def save_weights(self, folder_chp):
        self.model_down.qs_net.save_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.save_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.save_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.save_weights(folder_chp+'/checkpoint_ps')

    def load_weights(self, folder_chp):
        self.model_down.qs_net.load_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.load_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.load_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.load_weights(folder_chp+'/checkpoint_ps')

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        self.save_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','wb') as ff:
            pickle.dump(stats,ff)
        with open(folder_chp+'/optimizers.pkl','wb') as ff:
            pickle.dump(optimizers,ff)
        copyfile('src/tfmodel.py', folder_chp+'/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp+'/tfloss.py')
        if script_file != "":
            copyfile(script_file, folder_chp+'/'+script_file)

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','rb') as ff:
            stats = pickle.load(ff)
        try:
            with open(folder_chp+'/optimizers.pkl','rb') as ff:
                optimizers = pickle.load(ff)
        except:
            optimizers = {}
        if len(stats['var_beta_s'])>0: self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if len(stats['var_gamma'])>0: self.model_down.gamma.assign(stats['var_gamma'][-1])
        if len(stats['var_beta_o'])>0: self.model_down.beta_o.assign(stats['var_beta_o'][-1])
        return stats, optimizers

    def check_reward(self, o):
        if self.model_down.resolution == 64:
            return tf.reduce_mean(calc_reward(o),axis=[1,2,3]) * 10.0
        elif self.model_down.resolution == 32:
            return tf.reduce_sum(calc_reward_animalai(o), axis=[1,2,3])

    @tf.function
    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    @tf.function
    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi

    @tf.function
    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision)]
        sum_G = tf.zeros([o.shape[0]], self.tf_precision)

        # Predict s_t+1 for various policies
        if calc_mean: s0_temp = qs0_mean
        else: s0_temp = qs0

        for t in range(steps):
            G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1

    @tf.function
    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision)]
        sum_G = tf.zeros([4], self.tf_precision)

        # Predict s_t+1 for various policies
        if calc_mean: s0_temp = qs0_mean
        else: s0_temp = qs0

        for t in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp, self.pi_one_hot)
            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, self.pi_one_hot, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1

    @tf.function
    def calculate_G(self, s0, pi0, samples=10):

        term0 = tf.zeros([s0.shape[0]], self.tf_precision)
        term1 = tf.zeros([s0.shape[0]], self.tf_precision)
        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

            # E [ log P(o|pi) ]
            logpo1 = self.check_reward(po1)
            term0 += logpo1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]
            term1 += - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)
        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = tf.zeros(s0.shape[0], self.tf_precision)
        term2_2 = tf.zeros(s0.shape[0], self.tf_precision)
        for _ in range(samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[0])
            term2_1 += tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2_2 += tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])
        term2_1 /= float(samples)
        term2_2 /= float(samples)

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = - term0 + term1 + term2

        return G, [term0, term1, term2], ps1, ps1_mean, po1

    @tf.function
    def calculate_G_mean(self, s0, pi0):

        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1 = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        logpo1 = self.check_reward(po1)
        term0 = logpo1

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = - term0 + term1 + term2

        return G, [term0, term1, term2], ps1_mean, po1

    @tf.function
    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        po1 = self.model_down.decoder(ps1_traj)
        qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(po1)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar_traj) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return - term0 + term1 + term2

    #@tf.function
    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = np.zeros((depth, self.s_dim), self.precision)
        ps1 = np.zeros((depth, self.s_dim), self.precision)
        ps1_mean = np.zeros((depth, self.s_dim), self.precision)
        ps1_logvar = np.zeros((depth, self.s_dim), self.precision)
        pi0 = np.zeros((depth, self.pi_dim), self.precision)

        s0[0] = starting_s
        try:
            Qpi_t_to_return = self.model_top.encode_s(s0[0].reshape(1,-1))[1].numpy()[0]
            pi0[0, np.random.choice(self.pi_dim, p=Qpi_t_to_return)] = 1.0
        except:
            pi0[0, 0] = 1.0
            Qpi_t_to_return = pi0[0]
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[0].reshape(1,-1), s0[0].reshape(1,-1))
        ps1[0] = ps1_new[0].numpy()
        ps1_mean[0] = ps1_mean_new[0].numpy()
        ps1_logvar[0] = ps1_logvar_new[0].numpy()
        if 1 < depth:
            if use_means:
                s0[1] = ps1_mean_new[0].numpy()
            else:
                s0[1] = ps1_new[0].numpy()
        for t in range(1, depth):
            try:
                pi0[t, np.random.choice(self.pi_dim, p=self.model_top.encode_s(s0[t].reshape(1,-1))[1].numpy()[0])] = 1.0
            except:
                pi0[t, 0] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[t].reshape(1,-1), s0[t].reshape(1,-1))
            ps1[t] = ps1_new[0].numpy()
            ps1_mean[t] = ps1_mean_new[0].numpy()
            ps1_logvar[t] = ps1_logvar_new[0].numpy()
            if t+1 < depth:
                if use_means:
                    s0[t+1] = ps1_mean_new[0].numpy()
                else:
                    s0[t+1] = ps1_new[0].numpy()

        G = tf.reduce_mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).numpy()
        return G, pi0, Qpi_t_to_return
