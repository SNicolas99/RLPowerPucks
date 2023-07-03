__author__  = "Maximilian Beller"

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.training import training_util
from SAC.memory2 import ReplayBuffer
import progressbar

from IPython.display import clear_output
import matplotlib.pyplot as plt

import os 


class QFunction(tf.keras.Model):
    def __init__(self, q_fct_config):
        super(QFunction, self).__init__()
        self.layer1 = tf.keras.layers.Dense(q_fct_config['hidden_size'], activation='relu')
        self.layer2 = tf.keras.layers.Dense(q_fct_config['hidden_size'], activation='relu')
        self.q_value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        q_value = self.q_value(x)
        return q_value



def plot(total_rewards_per_episode=[], total_loss_V=[], total_loss_Q1=[], total_loss_Q2=[], total_loss_PI=[], winning=[], plot_type=0):
    """
    Can be used to live-plot the rewards, losses, winning rate during training instead of e.g. Tensorboard.
    total_rewards_per_episode:  array containing the rewards per episode
    total_loss_V:               array containing the losses of the Value function
    total_loss_Q1:              array containing the losses of the Q1 function
    total_loss_Q2:              array containing the losses of the Q2 function
    total_loss_PI:              array containing the losses of the policy network
    winning:                    array containing the percentage of wins so far
    plot_type:      = 0:        only plot total rewards
                    = 1:        plot total_rewards and losses
                    = 2:        plot total rewards and winning fraction
                    = 3:        only plot winning fraction
    """
    clear_output(True)
    if plot_type == 0:
        plt.plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
    elif plot_type == 1:
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0, 0].set_title("Reward")
        axes[0, 1].plot(range(len(total_loss_V)), total_loss_V)
        axes[0, 1].set_title("V loss")
      
        axes[1, 0].plot(range(len(total_loss_Q1)), total_loss_Q1, c="r")
        axes[1, 0].plot(range(len(total_loss_Q2)), total_loss_Q2, c="b")
        axes[1, 0].set_title("Q losses")
        axes[1, 1].plot(range(len(total_loss_PI)), total_loss_PI)
        axes[1, 1].set_title("PI Loss") 
    elif plot_type == 2:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0].set_title("Reward")
        axes[1].plot(range(len(winning)), winning)
        axes[1].set_title("Win fraction")
    elif plot_type == 3:
        plt.plot(range(len(winning)), winning)
        
    plt.show()



class SoftActorCritic:
    def __init__(self, o_space, a_space, env, value_fct, policy_fct, dim_act, dim_obs, q_fct_config, v_fct_config, pi_fct_config, save_path):
        self.obs_dim = dim_obs
        self.act_dim = dim_act
        self.gamma = 0.99  # Discount factor
        self.alpha = 0.2  # Temperature parameter for entropy regularization
        self.polyak = 0.995  # Polyak averaging coefficient for target networks
        self.batch_size = 64  # Batch size for updates
        self.num_train_steps = 1000  # Number of training steps per update

        # Initialize the value function, policy function, and Q-functions
        self.v_fct = value_fct(v_fct_config)
        self.v_fct_target = value_fct(v_fct_config)
        self.pi_fct = policy_fct(pi_fct_config)
        self.q1_fct = QFunction(q_fct_config)
        self.q2_fct = QFunction(q_fct_config)

        # Initialize the target networks with the same weights as the main networks
        self.v_fct_target.set_weights(self.v_fct.get_weights())

        # Define the optimizers for the Q-functions, value function, and policy function
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Initialize the replay buffer
        self.buffer = ReplayBuffer()

        # Save path for the model
        self.save_path = save_path

        # Set the observation and action spaces
        self.o_space = o_space
        self.a_space = a_space

    def update(self):
        for _ in range(self.num_train_steps):
            # Sample a batch from the replay buffer
            batch = self.buffer.sample(self.batch_size)
            obs, acts, rews, next_obs, done = batch

            with tf.GradientTape(persistent=True) as tape:
                # Update Q-functions
                q1_values = self.q1_fct(tf.concat([obs, acts], axis=-1))
                q2_values = self.q2_fct(tf.concat([obs, acts], axis=-1))

                # Compute target Q-values
                v_next = self.v_fct(next_obs)
                target_q_values = tf.minimum(q1_values, q2_values) - self.alpha * self.log_pi(next_obs) * (1 - done)
                target_q_values = rews + self.gamma * target_q_values

                # Compute Q-function losses
                q1_loss = tf.reduce_mean((q1_values - target_q_values) ** 2)
                q2_loss = tf.reduce_mean((q2_values - target_q_values) ** 2)

                # Update Q-functions
                q1_grads = tape.gradient(q1_loss, self.q1_fct.trainable_variables)
                self.q_optimizer.apply_gradients(zip(q1_grads, self.q1_fct.trainable_variables))

                q2_grads = tape.gradient(q2_loss, self.q2_fct.trainable_variables)
                self.q_optimizer.apply_gradients(zip(q2_grads, self.q2_fct.trainable_variables))

                # Update value function
                v_values = self.v_fct(obs)
                v_target = tf.minimum(q1_values, q2_values) - self.alpha * self.log_pi(obs) * (1 - done)
                v_target = v_target - self.alpha * self.log_pi(obs) * (1 - done)
                v_loss = tf.reduce_mean((v_values - v_target) ** 2)

                v_grads = tape.gradient(v_loss, self.v_fct.trainable_variables)
                self.v_optimizer.apply_gradients(zip(v_grads, self.v_fct.trainable_variables))

                # Update policy function
                pi_values = self.pi_fct(obs)
                q1_pi_values = self.q1_fct(tf.concat([obs, pi_values], axis=-1))
                policy_loss = tf.reduce_mean(self.alpha * self.log_pi(obs) - q1_pi_values)

                pi_grads = tape.gradient(policy_loss, self.pi_fct.trainable_variables)
                self.pi_optimizer.apply_gradients(zip(pi_grads, self.pi_fct.trainable_variables))

            # Update target networks
            self.update_target_networks()

    def update_target_networks(self):
        # Update target value function
        for v_targ, v_main in zip(self.v_fct_target.trainable_variables, self.v_fct.trainable_variables):
            v_targ.assign(self.polyak * v_targ + (1 - self.polyak) * v_main)

    def log_pi(self, obs):
        pi_values = self.pi_fct(obs)
        log_pi_values = tf.math.log(tf.clip_by_value(pi_values, 1e-8, 1.0))
        return log_pi_values

    def get_action(self, obs):
        obs = tf.expand_dims(obs, axis=0)
        pi_values = self.pi_fct(obs)[0]
        action = tf.random.categorical(tf.math.log(pi_values), 1)
        return action.numpy()[0]

    def save_model(self, save_path):
        self.q1_fct.save_weights(save_path + "/q1_fct")
        self.q2_fct.save_weights(save_path + "/q2_fct")
        self.v_fct.save_weights(save_path + "/v_fct")
        self.pi_fct.save_weights(save_path + "/pi_fct")
        print("Model saved.")

    def load_model(self, load_path):
        self.q1_fct.load_weights(load_path + "/q1_fct")
        self.q2_fct.load_weights(load_path + "/q2_fct")
        self.v_fct.load_weights(load_path + "/v_fct")
        self.pi_fct.load_weights(load_path + "/pi_fct")
        print("Model loaded.")
