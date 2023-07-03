import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# Import the SAC class
from SAC_advanced import SAC

# Define the number of training steps
num_steps = 10000

# Create the Pendulum environment
env = gym.make('Pendulum-v0')

# Create the SAC agent
agent = SAC(env)

# Create a replay buffer
# replay_buffer = ReplayBuffer()
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=1000000)

# Create a tensorboard writer for logging
writer = SummaryWriter()

# Initialize the episode and step counters
episode = 0
step = 0

# Run the training loop
while step < num_steps:
    episode += 1
    episode_reward = 0
    done = False

    # Reset the environment
    state = env.reset()

    # Episode loop
    while not done:
        step += 1

        # Choose an action
        action, log_prob, _ = agent.get_action(torch.FloatTensor(state))

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action.detach().numpy())

        # Store the transition in the replay buffer
        replay_buffer.add(state, action.detach().numpy(), reward, next_state, done)

        # Update the agent
        agent.update(replay_buffer)

        # Update the episode reward
        episode_reward += reward

        # Update the state
        state = next_state

        # Render the environment (optional)
        env.render()

        # Break if the maximum number of steps is reached
        if step >= num_steps:
            break

    # Log the episode reward
    writer.add_scalar('Episode Reward', episode_reward, episode)

    # Print the episode reward
    print(f'Episode: {episode}\tEpisode Reward: {episode_reward}')

# Close the tensorboard writer
writer.close()

# Test the learned policy
state = env.reset()
done = False
total_reward = 0

while not done:
    action, _, _ = agent.get_action(torch.FloatTensor(state))
    next_state, reward, done, _ = env.step(action.detach().numpy())
    total_reward += reward
    state = next_state
    env.render()

# Print the total reward
print(f'Test Total Reward: {total_reward}')
