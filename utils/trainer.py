import numpy as np
from time import perf_counter
from .logger import Logger

class Trainer:
    
    @staticmethod
    def run(env, agent, n_episodes=100, noise=0, store_transitions=True):
        rewards = []
        observations = []
        actions = []
        steps = []
        for ep in range(1, n_episodes+1):
            ep_reward = 0
            state, _info = env.reset()
            for t in range(2000):
                action = agent.act(state, noise)
                (next_state, reward, done, _trunc, _info) = env.step(action)

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
            steps.append(t)
            rewards.append(ep_reward)
            ep_reward = 0
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        steps = np.asarray(steps)
        return steps, rewards, observations, actions

    def train(self, env, agent, n_episodes=1000, train_every=1, test_every=100, n_test_episodes=20, noise=0.2):
        n_steps = n_episodes // train_every
        ep_per_step = train_every
        self.logger = Logger(n_steps=n_steps, print_every=test_every)
        try:
            for i in range(n_steps):
                start = perf_counter()
                steps, rewards, observations, actions = Trainer.run(env, agent, n_episodes=ep_per_step, noise=noise)
                ep_duration = (perf_counter() - start) / ep_per_step
                
                start = perf_counter()
                losses_train = np.mean(agent.train(), axis=0)
                train_duration = perf_counter() - start

                critic_loss = losses_train[0]
                actor_loss = losses_train[1]

                self.logger.log(steps, rewards, actor_loss, critic_loss, ep_duration, train_duration)

                if i > 0 and i % test_every == 0:
                    test_steps, test_rewards, test_observations, test_actions = self.run(env, agent, n_episodes=n_test_episodes, noise=0., store_transitions=False)
                    self.logger.log_test(test_rewards)
                    self.logger.print(i)
        except KeyboardInterrupt:
            print("Interrupted")
        return self.logger.ep_rewards