from environments import GymEnvironment
from algorithms import BaseAlgorithm # SAC, DDQN, DDPG
from algorithms import SAC
from laserhockey import LaserHockeyEnv, HockeyEnv

def train(env: GymEnvironment, agent: BaseAlgorithm,
          n_episodes: int = 1000, max_timesteps: int = 1000,
          save_model: bool = False, test_model: bool = False):
    """
    Train the agent on the given environment.

    :param env: The environment to train on.
    :param agent: The agent to train.
    :param n_episodes: The number of episodes to train for. (default: 1000)
    :param save_model: Whether to save the model after training. (default: False)
    :param test_model: If True, the model will be tested after training. (default: False)
    """
    # Initialize the environment
    state = env.reset()

    # Train the agent
    for episode in range(n_episodes):
        if episode % 10000 == 0:
            print(f'Episode {episode + 1}/{n_episodes}')
        # Train for max_timesteps
        for t in range(max_timesteps):# Choose an action
            action = agent.choose_action(state)

            # Perform the action
            next_state, reward, done, truncated, info = env.step(action)

            # Store the transition
            agent.store_transition({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            # Update the state
            state = next_state

            # Update the agent
            agent.learn()

            # Render the environment
            env.render()

            # Check if the episode is done
            if done:
                break
        if reward > -100:
            print(f'Episode {episode + 1}/{n_episodes} finished in {t + 1} timesteps')
            print(f'Episode reward: {reward}')
            print()


        # Reset the environment
        state = env.reset()
        done = False


    # Close the environment
    env.close()

if __name__ == '__main__':
    pass
