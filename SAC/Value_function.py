__author__ = "Maximilian Beller"

import numpy as np
import tensorflow as tf


class ValueFunction:
    def __init__(self, inp, scope='Q_fct', **userconfig):
        """
        Class defining the value functions of the SAC-agent (Q_functions, Value function, target Value function)
        """
        self._scope = scope
        self._sess = tf.compat.v1.get_default_session()
        self._input = inp

        self._config = {
            "hidden_layers": [256, 256],
            "hidden_act_fct": tf.nn.relu,
            "output_act_fct": None,
            "weights_init": tf.compat.v1.contrib.layers.xavier_initializer(),
            "bias_init": tf.constant_initializer(0.)
        }
        self._config.update(userconfig)

        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            self._build_graph()

    def _build_graph(self):
        """
        simple neural network
        """
        x = self._input
        for i, l in enumerate(self._config["hidden_layers"]):
            x = tf.compat.v1.layers.dense(x, l, activation=self._config["hidden_act_fct"],
                                          kernel_initializer=self._config["weights_init"],
                                          bias_initializer=self._config["bias_init"], name="hidden_%s" % (i))
        self.output = tf.compat.v1.layers.dense(x, 1, activation=self._config["output_act_fct"],
                                                kernel_initializer=self._config["weights_init"],
                                                bias_initializer=self._config["bias_init"], name="output")
