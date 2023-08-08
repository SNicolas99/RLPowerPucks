from typing import Literal
import laserhockey.hockey_env as h_env
import numpy as np
from gymnasium.spaces import Box

mode_to_hockey_mode = {
    "defense": h_env.HockeyEnv.TRAIN_DEFENSE,
    "attack": h_env.HockeyEnv.TRAIN_SHOOTING,
    "normal": h_env.HockeyEnv.NORMAL,
    "bootcamp": h_env.HockeyEnv.TRAIN_DEFENSE,
}


class HockeyWrapper:
    def __init__(
        self,
        mode: Literal["defense", "attack", "normal", "bootcamp"] = "bootcamp",
        opponent="weak",  # weak, strong, mixed, or agent object
        render_mode=None,
        opponent_list=None,
        add_opponents=False,
    ):

        self.add_opponents = add_opponents
        self.mix_opponents = opponent == "mixed"
        self.opponent_list = opponent_list

        if render_mode is not None:
            self.env = h_env.HockeyEnv(
                mode=mode_to_hockey_mode[mode], render_mode=render_mode
            )
        else:
            self.env = h_env.HockeyEnv(mode=mode_to_hockey_mode[mode])

        if self.mix_opponents:
            if self.opponent_list is None:
                self.opponent_list = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]
            self.opponent = self.opponent_list[np.random.randint(len(self.opponent_list))]
        elif opponent == "weak":
            self.opponent = h_env.BasicOpponent(weak=True)
        elif opponent == "strong":
            self.opponent = h_env.BasicOpponent(weak=False)
        else:
            self.opponent = opponent

        has = self.env.action_space
        new_low = has.low[:4]
        new_high = has.high[:4]
        self.action_space = Box(new_low, new_high, shape=(int(has.shape[0] / 2),), dtype=np.float32)
        
        self.observation_space = self.env.observation_space

        self.mode = mode
        self.episodes = 0
        self.shooting_start = 5000
        self.normal_start = 10000
        self.ishockey = True

    def add_opponent(self, opponent):
        self.opponent_list.append(opponent)

    def get_opponent_action(self):
        opponent_state = self.env.obs_agent_two()
        return self.opponent.act(opponent_state)

    def update_env(self):
        if self.mode == "bootcamp":
            if self.episodes == self.normal_start:
                self.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
            elif self.episodes > self.shooting_start:
                if self.episodes % 2 == 0:
                    self.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
                else:
                    self.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        if self.mix_opponents:
            self.opponent = self.opponent_list[np.random.randint(len(self.opponent_list))]

    def reset(self):
        self.episodes += 1
        self.update_env()
        return self.env.reset()

    def step(self, action):
        opponent_action = self.get_opponent_action()
        return self.env.step(np.hstack([action, opponent_action]))

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()