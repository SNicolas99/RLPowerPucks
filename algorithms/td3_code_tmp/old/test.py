# %%
import torch
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from IPython.display import clear_output
from ddpg import DDPG
from td3 import TD3

# %%
env_string = "LunarLander-v2"

# %%
def play(env, policy, steps=1000, render_every=4):
    state, info = env.reset()
    done = False
    step = 0
    rewards = []
    while not done and step < steps:
        if step % render_every == 0:
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
        action = policy(torch.tensor(state, dtype=torch.float32))
        state, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        step += 1
    env.close()
    return np.sum(rewards)

# %%
env = gym.make(env_string, render_mode="rgb_array", continuous=True)

# %%
agent = TD3(env, epochs=4500, steps_max=800, action_noise_scale=0.1, batch_size=64, iter_fit=32, gamma=0.99, buffer_size=100000, tau_critic=0.005, tau_actor=0.005)

# %%
torch.cuda.empty_cache()

# %%
ep_rewards, test_rewards = agent.train()

# %%
from torchviz import make_dot

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32)
action = agent.policy(state)

grad_plot = make_dot(action, params=dict(agent.policy.named_parameters()), show_attrs=True, show_saved=True)

#save grad plot
grad_plot.format = "png"
grad_plot.render("grad_plot")


# # %%
# # plot
# plt.figure(figsize=(10, 5))
# plt.plot(ep_rewards, label="train")
# plt.xlabel("epochs")
# plt.ylabel("reward")

# test_epochs = np.arange(0, len(test_rewards)) * 10
# plt.scatter(test_epochs, test_rewards, label="test", color="red", marker="x", zorder=100, s=10)

# # %%
# # env = gym.make(env_string, render_mode="rgb_array")

# # %%
# policy = lambda x: agent.get_action(x, noise=False)
# play(env, policy, steps=800, render_every=4)

# # %%
# # sample from buffer
# state, action, reward, next_state, done = agent.replay_buffer.sample(10)
# state = torch.tensor(state, dtype=torch.float32)

# agent.update_policy(state)

# # %%
# # noise_policy = lambda x: agent.get_action(x, noise=True)

# # %%
# # play(env, noise_policy, steps=1000, render_every=4)

# # %%
# max(test_rewards)

# # %%
# # test for some values
# state, info = env.reset()
# action = agent.get_action(torch.tensor(state, dtype=torch.float32), noise=False)
# qval = agent.get_q1(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
# # print

# print("state: ", state)
# print("action: ", action)
# print("qval: ", qval)

# # %%
# # %matplotlib inline
# # q = agent.q
# # # plot q function over state space
# # # sample states
# # n = 100
# # x = np.linspace(-1.2, 0.6, n)
# # y = np.linspace(-0.07, 0.07, n)
# # xx, yy = np.meshgrid(x, y)
# # states = np.stack([xx, yy], axis=-1).reshape(-1, 2)

# # # concat 0 action
# # states = np.concatenate([states, np.zeros((states.shape[0], 1))], axis=-1)

# # # compute q values

# # q_values = q(torch.tensor(states, dtype=torch.float32)).detach().numpy().reshape(n, n)

# # # plot 3d surf

# # from mpl_toolkits.mplot3d import Axes3D

# # fig = plt.figure(figsize=(10, 10))

# # ax = fig.add_subplot(111, projection="3d")
# # ax.plot_surface(xx, yy, q_values, cmap="viridis")

# # plt.xlabel("position")
# # plt.ylabel("velocity")
# # plt.title("Q function")


