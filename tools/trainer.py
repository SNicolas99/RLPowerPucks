import numpy as np
from time import perf_counter
from .logger import Logger


class Trainer:
    def __init__(self):
        self.logger = None
        self.current_step = 0

    @staticmethod
    def run(
        env,
        agent,
        n_episodes=100,
        noise=0,
        store_transitions=True,
        render=False,
        hockey=False,
    ):
        rewards = []
        observations = []
        actions = []
        steps = []
        hockey_results = []
        for ep in range(1, n_episodes + 1):
            ep_reward = 0
            state, _info = env.reset()
            for t in range(2000):
                action = agent.act(state, noise)
                (next_state, reward, done, _trunc, _info) = env.step(action)
                if render:
                    env.render()

                done = _trunc or done

                if store_transitions:
                    transition = (state, action, reward, next_state, done)
                    agent.store_transition(transition)

                state = next_state

                observations.append(state)
                actions.append(action)
                ep_reward += reward

                if done or _trunc:
                    break
            steps.append(t + 1)
            rewards.append(ep_reward)
            ep_reward = 0

            if hockey:
                result = _info["winner"]
                hockey_results.append(result)

        observations = np.asarray(observations)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        steps = np.asarray(steps, dtype=np.float32)
        hockey_results = np.asarray(hockey_results, dtype=np.float32)
        return steps, rewards, observations, actions, hockey_results

    def train(
        self,
        env,
        agent,
        n_episodes=1000,
        train_every=1,
        test_every=100,
        n_test_episodes=20,
        noise=0.2,
    ):
        hockey = hasattr(env, "ishockey")

        n_steps = n_episodes // train_every
        ep_per_step = train_every
        self.logger = Logger(n_steps=n_steps, print_every=test_every)

        n_steps = n_steps + self.current_step

        try:
            for i in range(self.current_step, n_steps):
                start = perf_counter()
                steps, rewards, observations, actions, results = Trainer.run(
                    env, agent, n_episodes=ep_per_step, noise=noise
                )
                ep_duration = (perf_counter() - start) / ep_per_step

                start = perf_counter()

                fit_losses, actor_losses = agent.train()
                critic_loss = np.mean(fit_losses, axis=0)
                if len(actor_losses) > 0:
                    actor_loss = np.mean(actor_losses, axis=0)
                else:
                    actor_loss = []

                train_duration = perf_counter() - start

                self.logger.log(
                    steps, rewards, actor_loss, critic_loss, ep_duration, train_duration
                )

                if i > 0 and i % test_every == 0:
                    (
                        test_steps,
                        test_rewards,
                        test_observations,
                        test_actions,
                        test_results,
                    ) = self.run(
                        env,
                        agent,
                        n_episodes=n_test_episodes,
                        noise=0.0,
                        store_transitions=False,
                        hockey=hockey,
                    )
                    self.logger.log_test(test_rewards)
                    winrate = np.mean(test_results == 1)
                    drawrate = np.mean(test_results == 0)
                    self.logger.log_hockey(winrate, drawrate)
                    self.logger.print(i)
        except KeyboardInterrupt:
            print("Interrupted")
        return self.logger.ep_rewards
