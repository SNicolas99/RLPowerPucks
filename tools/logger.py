import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def is_iterable(itm):
    try:
        iterator = iter(itm)
        return True
    except TypeError:
        return False
    
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Logger:

    def __init__(self, n_steps, print_every):
        self.n_steps = n_steps
        self.print_every = print_every
        
        self.ep_steps = []
        self.ep_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.ep_durations = []
        self.train_durations = []

        self.test_rewards = []
        self.winrate = []
        self.drawrate = []
        self.lossrate = []

        self.hockey = False

        self.test_interval = 100

    def log(self, steps, ep_rewards, actor_losses, critic_losses, ep_durations, train_durations):
        items = [steps, ep_rewards, actor_losses, critic_losses, ep_durations, train_durations]
        lists = [self.ep_steps, self.ep_rewards, self.actor_losses, self.critic_losses, self.ep_durations, self.train_durations]

        for itm, lst in zip(items, lists):
            if is_iterable(itm):
                lst.extend(itm)
            else:
                lst.append(itm)
    
    def log_test(self, test_reward_array):
        self.test_rewards.append(np.mean(test_reward_array, axis=0))

    def log_hockey(self, winrate, drawrate):
        self.hockey = True
        self.winrate.append(winrate)
        self.drawrate.append(drawrate)
        lossrate = 1-winrate-drawrate
        self.lossrate.append(lossrate)

    def print(self, i):
        hockey_str = ''
        if self.hockey:
            hockey_str = f"""winrate: {self.winrate[-1]:.2f}, drawrate: {self.drawrate[-1]:.2f}, lossrate: {self.lossrate[-1]:.2f}"""
        print(f"""Step {i}/{self.n_steps}:
                    test reward: {self.test_rewards[-1]:.2f}
                    mean reward: {np.mean(self.ep_rewards[-self.print_every:]):.2f}, max reward: {np.max(self.ep_rewards[-self.print_every:]):.2f}
                    Avg. step count: {np.mean(self.ep_steps[-100:]):.1f}, Avg ep duration: {np.mean(self.ep_durations[-100:]):.3f}s
                    Avg. critic loss: {np.mean(self.critic_losses[-100:]):.2f}, Avg. actor loss: {np.mean(self.actor_losses[-100:]):.2f}
                    Avg. train duration: {np.mean(self.train_durations[-100:]):.3f}s
                    {hockey_str}
            """)

    def plot(self):
        
        test_rewards = np.array(self.test_rewards)
        ep_steps = np.array(self.ep_steps)
        ep_durations = np.array(self.ep_durations)
        ep_rewards = np.array(self.ep_rewards)
        actor_losses = np.array(self.actor_losses)
        critic_losses = np.array(self.critic_losses)
        if self.hockey:
            winrate = np.array(self.winrate)
            drawrate = np.array(self.drawrate)
            lossrate = np.array(self.lossrate)
            hockey_episodes = np.arange(winrate.size) * self.test_interval

        test_episodes = np.arange(test_rewards.size) * self.test_interval
        

        if test_rewards.size == 0:
            print('No test rewards to plot')
        elif test_rewards.size < 51:
            plt.plot(test_episodes, test_rewards, label='test reward')
        else:
            plt.plot(test_episodes, savgol_filter(test_rewards, 10, 3, axis=0), label='test reward')
        plt.title('test reward')
        plt.show()

        if self.hockey:
            if winrate.size < 51:
                plt.plot(hockey_episodes, winrate, label='winrate')
                plt.plot(hockey_episodes, lossrate, label='lossrate')
            else:
                plt.plot(hockey_episodes, savgol_filter(winrate, 51, 3, axis=0), label='winrate')
                plt.plot(hockey_episodes, savgol_filter(lossrate, 51, 3, axis=0), label='lossrate')
            plt.title('hockey result')
            plt.legend()
            plt.show()

        plt.plot(savgol_filter(ep_rewards, 51, 3, axis=0), label='reward')
        plt.title('reward')
        plt.show()

        plt.plot(savgol_filter(critic_losses, 51, 3, axis=0), label='critic loss')
        plt.title('critic loss')
        plt.show()

        plt.plot(savgol_filter(actor_losses, 51, 3, axis=0), label='actor loss')
        plt.title('actor loss')
        plt.show()

        plt.plot(savgol_filter(ep_steps, 51, 3, axis=0), label='avg. ep steps')
        plt.title('avg. ep steps')
        plt.show()

        plt.plot(savgol_filter(ep_durations, 51, 3, axis=0), label='avg. ep durations')
        plt.title('avg. ep durations')
        plt.show()

        plt.plot(savgol_filter(self.train_durations, 51, 3, axis=0), label='avg. train durations')
        plt.title('avg. train durations')
        plt.show()

    def save(self, path):
        state = {
            'ep_steps': self.ep_steps,
            'ep_rewards': self.ep_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'ep_durations': self.ep_durations,
            'train_durations': self.train_durations,
            'test_rewards': self.test_rewards,
            'winrate': self.winrate,
            'drawrate': self.drawrate,
            'lossrate': self.lossrate
        }
        np.save(path, state)
    
    def load(self, path):
        state = np.load(path, allow_pickle=True).item()
        self.ep_steps = state['ep_steps']
        self.ep_rewards = state['ep_rewards']
        self.actor_losses = state['actor_losses']
        self.critic_losses = state['critic_losses']
        self.ep_durations = state['ep_durations']
        self.train_durations = state['train_durations']
        self.test_rewards = state['test_rewards']
        self.winrate = state['winrate']
        self.drawrate = state['drawrate']
        self.lossrate = state['lossrate']
