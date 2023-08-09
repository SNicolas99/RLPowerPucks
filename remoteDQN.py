import numpy as np

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

from algorithms.DDQN.agent import DQNAgent
from laserhockey.hockey_env import HockeyEnv

env = HockeyEnv()


class RemoteDQNAgent(DQNAgent, RemoteControllerInterface):

    def __init__(self, keep_mode=True):
        DQNAgent.__init__(self,observation_space=env.observation_space,action_space=env.discrete_action_space, tournament_weights="/Users/lucaf/RLPowerPucks/algorithms/DDQN/results/20230809-091133_10000_normal", tournament_call=True)
        RemoteControllerInterface.__init__(self, identifier='DuelingAndPrioritizedDQN')

    def remote_act(self,
                   obs : np.ndarray,
                   ) -> np.ndarray:

        #x = self.act(obs)
        #print(str(env.discrete_to_continous_action(x)))
        return np.asarray(env.discrete_to_continous_action(self.act(obs)))


if __name__ == '__main__':
    controller = RemoteDQNAgent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='RL Power Pucks',
                    password='EeB6eo1foo',
                    controller=controller,
                    output_path='logs/basic_opponents', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
