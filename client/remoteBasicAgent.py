import numpy as np
import torch

from laserhockey.hockey_env import BasicOpponent
import sys
sys.path.append('..')
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
from algorithms.td3_code_tmp.td3_lecture.TD3 import TD3Agent
from environments.hockey_wrapper import HockeyWrapper

class RemoteBasicAgent(TD3Agent, RemoteControllerInterface):

    def __init__(self, keep_mode=True):
        env = HockeyWrapper()
        TD3Agent.__init__(self, env.observation_space, env.action_space)
        agent_state = torch.load("../algorithms/td3_code_tmp/td3_lecture/checkpoint_hockey_add.pth")
        self.restore_state(agent_state)

        RemoteControllerInterface.__init__(self, identifier='TD3++')

    def remote_act(self,
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs, eps=0.)


if __name__ == '__main__':
    controller = RemoteBasicAgent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='RL Power Pucks',
                    password='EeB6eo1foo',
                    controller=controller,
                    output_path='logs/basic_opponents', # rollout buffer with finished games will be saved in here
                    interactive=True,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
