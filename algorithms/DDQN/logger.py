
from matplotlib import pyplot as plt
import time
import os
import numpy as np


class Logger:
    def __init__(self, log_directory, config) -> None:
        self.log_directory = log_directory
        self.config = config

    def print_win_every_1000(self, result_list, episode_count):



        i = episode_count-1000
        result_list = np.array(result_list[i:episode_count])

        loss_count = np.count_nonzero(result_list == -1)
        win_count = np.count_nonzero(result_list == 1)
        print("Wins: " + str(win_count) + " | Losses: " + str(loss_count))

        print("Win/Lose: " + str(win_count/loss_count))





    def plot_win_percentage(self, outcomes, final=False):
        win_count = 0
        lose_count = 0
        win_percentages = []


        for outcome in outcomes:
            if outcome == 1:  # Win
                win_count += 1
            elif outcome == 0:  # Draw
                pass
            elif outcome == -1:  # Loss
                lose_count += 1

            # To prevent division through zero
            if lose_count == 0:
                win_percentages.append(1)
            else:
                win_percentages.append(win_count / (lose_count+win_count))



        # Filter win_percentages to keep only values at intervals of 100
        filtered_win_percentages = win_percentages[::20]
        filtered_win_percentages[0] = 0
        episodes_for_filtered_values = range(1, len(outcomes) + 1, 20)
        plt.plot(episodes_for_filtered_values, filtered_win_percentages)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Win Percentage")
        plt.title("Win Percentage vs. Number of Episodes")
        plt.grid(True)

        if final:
            plt.savefig(os.path.join(self.log_directory, 'win_percentage.png'))

            # TODO: See if this works properly
            file = open(os.path.join(self.log_directory, "hyperparams.txt"), "a")
            file.write("\n##############################\n")
            file.write("WIN PERCENTAGES\n")
            file.write("##############################\n")
            file.write(str(win_percentages))
            file.close()


        plt.show()


    def plot_win_percentage_seperately(self, result_list):
        result = []
        count_1 = 0
        count_minus1 = 0

        for i, num in enumerate(result_list, 1):
            if num == 1:
                count_1 += 1
            elif num == -1:
                count_minus1 += 1

            if i % 1000 == 0:
                if count_minus1 == 0:
                    result.append(100.0)  # If there are no -1s, append 100% for 1s.
                else:
                    percentage = (count_1 / (count_minus1+count_1)) * 100.0
                    result.append(percentage)
                count_1 = 0
                count_minus1 = 0

        x = [i * 1000 for i in range(1, len(result) + 1)]

        print(str(x))
        print(str(result))

        plt.plot(x, result, marker='o', linestyle='-', color='b')
        plt.xlabel('Episodes in 1000')
        plt.ylabel('Win/Lose Percentage')
        plt.title('Wins compared to losses')
        plt.grid(True)
        plt.show()

        plt.savefig(os.path.join(self.log_directory, 'Win_Perecentage_1000_steps.png'))





    def plot_e_decay(self, e_decay_list, final=False):
        plt.plot(range(1, len(e_decay_list) + 1), e_decay_list)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon value in the course of training")
        plt.grid(True)

        if final:
            plt.savefig(os.path.join(self.log_directory, 'e_decay.png'))

        plt.show()

    def print_and_log_main_run_data(self, result_list, max_episodes, reward_total, total_time):
        win_count = 0
        lose_count = 0
        undecided_count = 0

        for result in result_list:
            if result == 1:
                win_count += 1
            if result == -1:
                lose_count += 1
            else:
                undecided_count += 1

        print("##############################")
        print("ANALYSIS")
        print("------------------------------")
        print("total time: " + str(total_time))
        print("------------------------------")
        print("Win/Lose/Undecided: " + str(win_count) + "/" + str(lose_count) + "/" + str(
            max_episodes - win_count - lose_count))
        print("------------------------------")
        print("Win/Lose Rate: " + str(win_count / lose_count))
        print("##############################")

        file = open(os.path.join(self.log_directory, "hyperparams.txt"), "a")
        file.write("\n\n\n###########################")
        file.write("\nRUN RESULTS")
        file.write("\n###########################")
        file.write("\nTotal Time: " + str(round(total_time / 60, 3)) + " minutes")
        file.write("\nReward Total: " + str(round(reward_total, 3)))
        file.write("\nWin/Lose Rate: " + str(round(win_count / lose_count, 3)))
        file.write("\nWin/Lose/Undecided: " + str(win_count) + "/" + str(lose_count) + "/" + str(
            max_episodes - win_count - lose_count))
        file.write("\n###########################\n")
        file.close()

    def print_time_info(self, episode_count, max_episodes, start_time):
        print(str(episode_count) + "/" + str(max_episodes) + " episodes done.")
        curr_dur = time.time() - start_time
        estimated_runtime = (((max_episodes / episode_count) * curr_dur) - curr_dur) / 60
        print("Estimated remaining runtime: " + str(round(estimated_runtime, 3)) + " minutes")

    def plot_reward(self, reward_list, final=False):
        plt.plot(range(1, len(reward_list) + 1), reward_list)
        plt.xlabel("Number of Steps")
        plt.ylabel("Reward Value")
        plt.title("Reward Value in the course of training")
        plt.grid(True)

        if final:
            plt.savefig(os.path.join(self.log_directory, 'reward.png'))

        plt.show()

    # TODO adapt the method argument names if changed in the argument parser
    def save_hyperparams_to_file(self, e, e_min, e_decay, batch_size, discount, grd_upd_frq, step_max, num_episodes,
                                 training_mode, learning_rate, buffer_size, exp_rep_frq, use_target_net, use_per, use_existing_weights, weight_path):

        file = open(os.path.join(self.log_directory, "hyperparams.txt"), "w")
        file.write("\n###########################")
        file.write("\nHYPERPARAMETERS")
        file.write("\n###########################")
        file.write("\ne: " + e)
        file.write("\ne_min: " + e_min)
        file.write("\ne_decay: " + e_decay)
        file.write("\nbatch_size: " + batch_size)
        file.write("\ndiscount: " + discount)
        file.write("\ngrd_upd_frq: " + grd_upd_frq)
        file.write("\nstep_max: " + step_max)
        file.write("\nnum_episodes: " + num_episodes)
        file.write("\ntraining_mode: " + training_mode)
        file.write("\nlearning_rate: " + learning_rate)
        file.write("\nbuffer_size: " + buffer_size)
        file.write("\nexp_rep_frq: " + exp_rep_frq)
        file.write("\nuse_target_net: " + use_target_net)
        file.write("\nuse_PER: " + use_per)
        file.write("\nuse_existing_weights: " + use_existing_weights)
        if use_existing_weights:
            file.write("\nweight_path: " + weight_path)
        file.write("\n###########################\n")
        file.close()


    def store_all_eval_results(self, eval_results):
        file = open(os.path.join(self.log_directory, "hyperparams.txt"), "a")
        file.write("\n###########################")
        file.write("\nEVALUATION RESULTS")
        file.write("\n###########################\n")
        file.write(str(eval_results))
        file.write("###########################\n")
        file.close()

    def plot_eval_results(self, eval_results):


        x = [i * 1000 for i in range(1, len(eval_results) + 1)]
        x.insert(0,0)
        eval_results.insert(0,0)
        print("eval_results: " + str(eval_results))

        plt.plot(x, eval_results)
        plt.xlabel("Episodes")
        plt.ylabel("Win/Lose Percentage")
        plt.title("Evaluation: Wins compared to losses")
        plt.grid(True)
        plt.savefig(os.path.join(self.log_directory, 'final_evaluation.png'))
        plt.show()

