from typing import Literal
import laserhockey.hockey_env as h_env
import numpy as np

mode_to_hockey_mode = {
    "defense": h_env.HockeyEnv.TRAIN_DEFENSE,
    "attack": h_env.HockeyEnv.TRAIN_SHOOTING,
    "normal": h_env.HockeyEnv.NORMAL,
}


class HockeyWrapper:
    def __init__(
        self,
        mode: Literal["defense", "attack", "normal"] = "normal",
        opponent="weak",  # weak, strong, or agent object
        render_mode=None,
    ):
        if render_mode is not None:
            self.env = h_env.HockeyEnv(
                mode=mode_to_hockey_mode[mode], render_mode=render_mode
            )
        else:
            self.env = h_env.HockeyEnv(mode=mode_to_hockey_mode[mode])

        if opponent == "weak":
            self.opponent = h_env.BasicOpponent(weak=True)
        elif opponent == "strong":
            self.opponent = h_env.BasicOpponent(weak=False)
        else:
            self.opponent = opponent

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_opponent_action(self):
        opponent_state = self.env.obs_agent_two()
        return self.opponent.act(opponent_state)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        opponent_action = self.get_opponent_action()
        return self.env.step(np.hstack([action, opponent_action]))

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
