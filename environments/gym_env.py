import gymnasium as gym
from laserhockey.laserhockey.laser_hockey_env import LaserHockeyEnv
from algorithms.DDQN.hockey_env import HockeyEnv

class GymEnvironment:
    """
    Wrapper class for gym environments
    """
    def __init__(self, env_name: str = 'laserhockey', device:str='cpu', **kwargs):
        """
        Initialize the environment.

        :param env_name: The name of the environment to use. (default: 'laserhockey')
        :param device: The device to use for training. (default: 'cpu')
        """
        if env_name == 'laserhockey':
            self.env = LaserHockeyEnv(**kwargs)
        elif env_name == 'hockey':
            self.env = HockeyEnv(**kwargs)
        else:
            self.env = gym.make(env_name, **kwargs)

        self.device = device

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # get the dimensions of the observation and action spaces
        self.obs_dim = self.env.observation_space.shape[0]
        # if discrete action space, use the number of actions as action_dim
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        # if continuous action space, use the length of the action space as action_dim
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
        else:
            raise NotImplementedError


    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        return self.env.step(action)

    def render(self):
        try:
            self.env.render(mode='human')
        except:
            self.env.render()
    def close(self):
        self.env.close()

    def __call__(self, *args, **kwargs):
        pass
    # todo: implement this