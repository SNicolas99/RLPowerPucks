import time

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

class Trainer:

    def __init__(self, logger, config, agent, log_directory) -> None:
        self.logger = logger
        self.config = config
        self.agent = agent
        self.log_directory = log_directory

    def train(self, env, logger):
        # TODO assign needed arguments to variables
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
        # TODO Logger Methode, die nach dem initialisieren erstmal alle wichtigen Variablen loggt

        player2 = h_env.BasicOpponent()

        start_time = time.time()
        # run through every episode
        while episode_count <= max_episodes:
            episode_count += 1
            if (episode_count % 20) == 0:
                logger.print_time_info(episode_count=episode_count, max_episodes=max_episodes, start_time=start_time)
            if (episode_count % 1000) == 0:
                self.logger.plot_win_percentage(outcomes=result_list)
                self.logger.print_win_every_1000(result_list=result_list, episode_count=episode_count)

            # Reset the environment every episode
            ob = env.reset()  # returns obs, info
            # to extract the observations
            ob = ob[0]

            # Decay epsilon every episode, leading to more "exploration"
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            e_decay_list.append(epsilon)

            ob2 = env.obs_agent_two()

            touched = 0
            first_time_touch = 1

            # For every step do
            for i in range(self.config['step_max']):
                a1 = self.agent.act(ob, eps=epsilon)
                a1_list = ACTIONS[a1]
                if self.config['training_mode'] == 'shooting':
                    a_player2 = [0, 0, 0, 0]
                else:  # in case mode = normal or defense
                    a_player2 = player2.act(ob2)
                (ob_new, reward, done, player1_contact_puck, info) = env.step(np.hstack([a1_list, a_player2]))

                curr_reward = reward + 0.05 * info['reward_touch_puck'] + 3 * info[
                    'reward_closeness_to_puck']  ## TODO figure out a good reward function
                reward_total += curr_reward
                reward_list.append(curr_reward)

                info_list.append([info['winner'], info['reward_closeness_to_puck'], info['reward_touch_puck'],
                                  info['reward_puck_direction']])

                if self.config['render_training']:
                    env.render()

                self.agent.store_transition((ob, a1, reward, ob_new, done))

                if done:
                    result_list.append(env.winner)
                    break
                if i == (self.config['step_max']-1):
                    result_list.append(0)

                #if step_count % self.config['experience_replay_frequency'] == 0:
                 #   exp_replay_loss.append(self.agent.train())

                ob = ob_new
                step_count += 1

                if grd_upd_frq % step_count == 0:
                    self.agent.update_target_net()

                ob2 = env.obs_agent_two()

        total_time = time.time() - start_time

        ## Save weights
        if self.config['save_weights']:
            try:
                self.agent.save_weights(os.path.join(self.log_directory, "weights"))
            except:
                print("Could not access filepath for saving weights")

        self.logger.print_and_log_main_run_data(max_episodes=max_episodes,
                                                result_list=result_list,
                                                reward_total=reward_total,
                                                total_time=total_time)
        self.logger.plot_win_percentage(result_list, final=True)
        self.logger.plot_e_decay(e_decay_list, final=True)
        self.logger.plot_reward(reward_list=reward_list, final=True)
