__author__  = "Maximilian Beller"
import numpy as np

import tensorflow_probability as tfp
import tensorflow as tf

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Normal_Policy:
    def __init__(self, inp,  scope='policy', **userconfig):
        """
        Class defining a policy for the SAC-agent. Here a neural network is used to sample Mean and STD for a Multivariate Gaussian.
        """
        self._scope = scope

        self._sess = tf.get_default_session() or tf.InteractiveSession()
        self._input = inp
        self._config = {
            "hidden_layers": [256, 256], 
            "hidden_act_fct": tf.nn.relu,
            "output_act_fct_mu": None,
            "output_act_fct_log_std": tf.tanh,
            "weights_init" : tf.contrib.layers.xavier_initializer(),
            "bias_init": tf.constant_initializer(0.),
            "dim": 1
            }
        self._config.update(userconfig)

    
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._build_graph()
            
    def _build_graph(self):
        """
        Setting up the neural network
        """

        # Simple feedforward neural network
        x = self._input
        for i,l in enumerate(self._config["hidden_layers"]):
            x = tf.layers.dense(x, l, activation=self._config["hidden_act_fct"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"],  name="hidden_%s" % (i))
        

        # Getting output for mean of Gaussian
        self.mu = tf.layers.dense(x, self._config["dim"],activation=self._config["output_act_fct_mu"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"], name = "mu")
        # Getting output for log of Gaussian
        self.log_std = tf.layers.dense(x,self._config["dim"], activation=self._config["output_act_fct_log_std"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"], name = "log_std")
        

        if self._config["output_act_fct_log_std"] != tf.tanh:
            self.log_std = tf.clip_by_value(self.log_std, -20, 2, name = "log_std_clipped")
        else:
            self.log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (self.log_std + 1)
        self.std = tf.exp(self.log_std, name ="std")
        
        self.normal_dist = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.std)
        
        self.sample = self.normal_dist.sample()
        
        self.act = tf.tanh(self.sample, name="tanh")

        #self.log_prob = self.normal_dist.log_prob(self.sample)
        self.log_prob = self.gaussian_likelihood(self.sample, self.mu, self.log_std)
        _,_, self.log_prob = self.apply_squashing_func(self.mu, self.sample, self.log_prob)

        # Passing the mean of the normal through the tanh to receive valid action
        self.mu_tanh = tf.tanh(self.mu)
       
    # Copied from OpenAI implementation  
    def gaussian_likelihood(self,input_, mu_, log_std):
        pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)    
    # Copied from OpenAI implementation   => allowing gradient to flow, increased performance
    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
    # Copied from OpenAI implementation
    def apply_squashing_func(self, mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
        return mu, pi, logp_pi