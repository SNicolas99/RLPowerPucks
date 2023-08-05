import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def is_iterable(itm):
    try:
        iterator = iter(itm)
        return True
    except TypeError:
        return False

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

    def print(self, i):
        print(f"""Step {i+1}/{self.n_steps}:
                    test reward: {self.test_rewards[-1]:.2f}
                    mean reward: {np.mean(self.ep_rewards[-self.print_every:]):.2f}, max reward: {np.max(self.ep_rewards[-self.print_every:]):.2f}
                    Avg. step number: {np.mean(self.ep_steps[-100:]):.1f}, Avg ep duration: {np.mean(self.ep_durations[-100:]):.3f}s
                    Avg. train duration: {np.mean(self.train_durations[-100:]):.3f}s
            """)

    def plot(self):
        
        test_rewards = np.array(self.test_rewards)
        ep_steps = np.array(self.ep_steps)
        ep_durations = np.array(self.ep_durations)
        ep_rewards = np.array(self.ep_rewards)
        actor_losses = np.array(self.actor_losses)
        critic_losses = np.array(self.critic_losses)

        plt.plot(savgol_filter(test_rewards, 51, 3, axis=0), label='test reward')
        plt.title('test reward')
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
            'train_durations': self.train_durations
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
