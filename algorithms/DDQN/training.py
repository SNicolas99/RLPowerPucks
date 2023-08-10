import time
from copy import deepcopy
import random

import numpy as np
import hockey_env as h_env
import os

# This is the observation format
# 0  x pos player one
# 1  y pos player one
# 2  angle player one
# 3  x vel player one
# 4  y vel player one
# 5  angular vel player one
# 6  x player two
# 7  y player two
# 8  angle player two
# 9 y vel player two
# 10 y vel player two
# 11 angular vel player two
# 12 x pos puck
# 13 y pos puck
# 14 x vel puck
# 15 y vel puck
# Keep Puck Mode
# 16 time left player has puck
# 17 time left other player has puck

ACTIONS = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]


class Training:

    def __init__(self, logger, config, agent, log_directory) -> None:
        self.logger = logger
        self.config = config
        self.agent = agent
        self.log_directory = log_directory

    def train(self, env, logger):
        epsilon = self.config['epsilon']
        epsilon_decay = self.config['e_decay']
        epsilon_min = self.config['e_min']
        grd_upd_frq = self.config['gradient_update_frequency']
        max_episodes = self.config['num_episodes']
        print("epsilon value at start: " + str(epsilon))

        episode_count = 0
        # Counts the number of total gradient updates
        grad_updates_count = 0
        step_count = 0

        reward_total = 0
        reward_list = []
        info_list = []
        result_list = []
        exp_replay_loss = []
        e_decay_list = []
        all_eval_results = []

        if self.config['play_own_agent'] is True:
            print("play own agent")
            player2 = deepcopy(self.agent)
        else:
            print("Opponent weak: " + str(self.config['weak_opponent']))
            player2 = h_env.BasicOpponent(weak=self.config['weak_opponent'])

        # TODO: Only in for tournament training
        #opponent_list = [h_env.BasicOpponent(weak=False), h_env.BasicOpponent(weak=True)]

        start_time = time.time()
        # run through every episode
        while episode_count <= max_episodes:
            episode_count += 1
            if (episode_count % 20) == 0:
                logger.print_time_info(episode_count=episode_count, max_episodes=max_episodes, start_time=start_time)
            if (episode_count % 1000) == 0:
                self.logger.plot_win_percentage(outcomes=result_list)
                self.logger.print_win_every_1000(result_list=result_list, episode_count=episode_count)

            if (episode_count % 1000) == 0:
                all_eval_results.append(self.evaluate(env=env))

            # if (episode_count % 1000) == 0:
            #   opponent_list.append(deepcopy(self.agent))

            # Reset the environment every episode
            ob = env.reset()  # returns obs, info
            # to extract the observations
            ob = ob[0]

            # Decay epsilon every episode, leading to more "exploration"
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            e_decay_list.append(epsilon)

            ob2 = env.obs_agent_two()
            #player2 = random.choice(opponent_list)
            touched = 0
            first_time_touch = 1

            # For every step do
            for step in range(self.config['step_max']):
                a_player1 = self.agent.act(ob, eps=epsilon)
                a_player1_list = ACTIONS[a_player1]
                if self.config['training_mode'] == 'shooting':
                    a_player2 = [0, 0, 0, 0]
                else:  # in case mode = normal or defense
                    a_player2 = player2.act(ob2)
                    if not isinstance(a_player2, np.ndarray):
                        a_player2 = ACTIONS[a_player2]
                (ob_new, reward, done, player1_contact_puck, info) = env.step(np.hstack([a_player1_list, a_player2]))

                curr_reward = reward + 0.05 * info['reward_touch_puck'] + 3 * info[
                    'reward_closeness_to_puck']


                reward_total += curr_reward
                reward_list.append(curr_reward)

                info_list.append([info['winner'], info['reward_closeness_to_puck'], info['reward_touch_puck'],
                                  info['reward_puck_direction']])

                if self.config['render_training']:
                    env.render()

                self.agent.store_transition((ob, a_player1, curr_reward, ob_new, done))

                if done:
                    result_list.append(env.winner)
                    break
                if step == (self.config['step_max'] - 1):
                    result_list.append(0)

                if step_count % self.config['experience_replay_frequency'] == 0:
                    exp_replay_loss.append(self.agent.train())

                ob = ob_new

                step_count += 1

                if step_count % grd_upd_frq == 0:
                    self.agent.update_target_net()

                ob2 = env.obs_agent_two()

        total_time = time.time() - start_time

        ## Save weights
        if self.config['save_weights']:
            try:
                self.agent.save_weights(os.path.join(self.log_directory, "weights"))
            except:
                print("Could not access filepath for saving weights")

        self.call_logger(max_episodes=max_episodes, result_list=result_list, reward_total=reward_total,
                         total_time=total_time, e_decay_list=e_decay_list, reward_list=reward_list,
                         eval_results=all_eval_results)

    def call_logger(self, max_episodes, result_list, reward_total, total_time, e_decay_list, reward_list, eval_results):
        self.logger.print_and_log_main_run_data(max_episodes=max_episodes,
                                                result_list=result_list,
                                                reward_total=reward_total,
                                                total_time=total_time)
        self.logger.plot_win_percentage(result_list, final=True)
        self.logger.plot_win_percentage_seperately(result_list)
        self.logger.plot_e_decay(e_decay_list, final=True)
        self.logger.plot_reward(reward_list=reward_list, final=True)
        self.logger.store_all_eval_results(eval_results=eval_results)
        self.logger.plot_eval_results(eval_results=eval_results)

    def evaluate(self, env):
        eval_result_list = []
        ob = env.reset()  # returns obs, info
        # to extract the observations
        ob = ob[0]
        ob2 = env.obs_agent_two()
        player2 = h_env.BasicOpponent(weak=self.config['weak_opponent'])
        for episode in range(1000):
            ob = env.reset()  # returns obs, info
            # to extract the observations
            ob = ob[0]
            for step in range(self.config['step_max']):
                a_player1 = self.agent.act(ob, eps=0.0)
                a_player1_list = ACTIONS[a_player1]
                a_player2 = player2.act(ob2)
                if not isinstance(a_player2, np.ndarray):
                    a_player2 = ACTIONS[a_player2]
                (ob_new, reward, done, player1_contact_puck, info) = env.step(np.hstack([a_player1_list, a_player2]))

                if done:
                    eval_result_list.append(env.winner)
                    break

                ob = ob_new

                ob2 = env.obs_agent_two()

        win_count = 0
        lose_count = 0
        for result in eval_result_list:
            if result == 1:
                win_count += 1
            if result == -1:
                lose_count += 1

        eval_result = win_count / (win_count + lose_count)
        print("eval_result: " + str(eval_result))
        return eval_result
