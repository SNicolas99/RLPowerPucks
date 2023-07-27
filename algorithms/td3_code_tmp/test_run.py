# %%
import torch
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from td3 import TD3
import laserhockey.hockey_env as h_env
import laserhockey.laser_hockey_env as lh

# %%
env_string = "HockeyEnv"

# %%
if env_string == "LunarLander-v2":
    env = gym.make(env_string, continuous=True)
elif env_string == "HockeyEnv":
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    hockey_opponent = h_env.BasicOpponent(weak=True)
elif env_string == "LaserHockeyEnv":
    env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.NORMAL)
    hockey_opponent = lh.BasicOpponent()
else:
    env = gym.make(env_string)

# %%
agent = TD3(
    env,
    env_string=env_string,
    epochs=30000,
    steps_max=1000,
    action_noise_scale=0.1,
    prioritized_replay=False,
    gradient_steps=200,
    hockey_opponent=hockey_opponent,
    noise_mode="gaussian",
    h=128,
    gamma=0.995,
    batch_size=32,
)

# %%
# agent.total_steps = 2000
# agent.test(render_mode="human", noise_enabled=True)

# %%
# agent.env.close()

# %%
(
    episode_rewards,
    test_rewards,
    episode_steps,
    ep_durations,
    upd_durations,
    ep_actor_losses,
    ep_critic_losses,
) = agent.train()

# %%
# plot all
# plt.figure(figsize=(25, 15))
# plt.subplot(3, 3, 1)
# plt.plot(episode_rewards)
# plt.title("Episode rewards")
# plt.subplot(3, 3, 2)
# plt.plot(np.arange(len(test_rewards))*10, test_rewards)
# # plt.scatter(np.arange(len(test_rewards))*10, test_rewards, s=2, alpha=0.2)
# plt.title("Test rewards")
# plt.subplot(3, 3, 3)
# plt.plot(episode_steps)
# plt.title("Episode steps")
# plt.subplot(3, 3, 4)
# plt.plot(ep_durations)
# plt.title("Episode durations")
# plt.subplot(3, 3, 5)
# plt.plot(upd_durations)
# plt.title("Update durations")
# plt.subplot(3, 3, 6)
# plt.plot(ep_actor_losses)
# plt.title("Episode actor loss")
# plt.subplot(3, 3, 7)
# plt.plot(ep_critic_losses)
# plt.title("Episode critic loss")
# plt.show()

# %%
# save policy and q function
torch.save(agent.policy.state_dict(), "policy.pt")
torch.save(agent.q1.state_dict(), "q1.pt")
torch.save(agent.q2.state_dict(), "q2.pt")

# # %%
# # from feedforward import FeedforwardNetwork
# # pol_test = FeedforwardNetwork(18,4, h=64)
# # pol_test.load_state_dict(torch.load("policy.pt"))

# # %%
# # # plot
# # plt.figure(figsize=(10, 5))
# # plt.plot(ep_rewards, label="train")
# # plt.xlabel("epochs")
# # plt.ylabel("reward")

# # test_epochs = np.arange(0, len(test_rewards)) * 10
# # plt.scatter(test_epochs, test_rewards, label="test", color="red", marker="x", zorder=100, s=10)

# # %%
# # env = gym.make(env_string, render_mode="rgb_array")

# # %%
# from tqdm import tqdm

# # %%
# wins = 0
# draws = 0
# for i in range(100):
#     agent.test(render_mode=None, noise_enabled=False)
#     if agent.env.winner == 1:
#         wins += 1
#     elif agent.env.winner == 0:
#         draws += 1

# print("wins: ", wins)
# print("draws: ", draws)
# print("losses: ", 100 - wins - draws)

# # %%
# agent.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

# # %%
# agent.test(noise_enabled=False, render_mode="human")
# agent.env.reset()

# # %%
# class MyHockeyOpponent:
#     def __init__(self):
#         pass

#     def act(self, obs):
#         return agent.get_action(obs, noise=False)
#         # return np.array([0, 0, 0.5, 0])
    
# agent.hockey_opponent = MyHockeyOpponent()

# # %%
# agent.hockey_opponent = h_env.BasicOpponent(weak=False)
# agent.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

# # %%
# for i in range(15):
#     agent.test(noise_enabled=False, render_mode="human")

# # %%
# # # test normal
# # env_normal = h_env.HockeyEnv(mode=0)
# # agent.env = env_normal
# # agent.test(noise_enabled=False, render_mode="human")
# # agent.env = env

# # %%
# # sample from buffer
# # state, action, reward, next_state, done = agent.replay_buffer.sample(10)
# # state = torch.tensor(state, dtype=torch.float32)
# # agent.update_policy(state)

# # %%
# # combined = torch.cat((state, action), dim=-1)
# # qval = agent.q1(combined)
# # qval

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
# qval = agent.get_q1(
#     torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)
# )
# # print

# print("state: ", state)
# print("action: ", action)
# print("qval: ", qval)

# # %%
# %matplotlib inline
# q = agent.q1
# # plot q function over state space
# # sample states
# n = 100
# x = np.linspace(-1.2, 0.6, n)
# y = 0.0
# actions = np.linspace(-1.0, 1.0, n)

# states = np.zeros((n, 2))
# for i in range(n):
#     states[i, :] = np.array([x[i], y])

# # compute q values
# q_values = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         q_values[i, j] = q(torch.hstack((torch.tensor(states[i], dtype=torch.float32), torch.tensor(actions[j], dtype=torch.float32)))).detach().numpy()

# policy_actions = np.zeros((n))
# for i in range(n):
#     policy_actions[i] = agent.get_action(torch.tensor(states[i], dtype=torch.float32), noise=False)

# # plot 2d
# plt.figure(figsize=(10, 10))
# plt.imshow(q_values, extent=[-1.2, 0.6, -1.0, 1.0], aspect="auto")
# plt.colorbar()
# plt.scatter(states[:, 0], policy_actions, color="red", marker="x", zorder=100, s=10)

# plt.xlabel("position")
# plt.ylabel("action")
# plt.title("Q function")

# # %%
# state = np.array([-0.9, 0])
# state = torch.tensor(state, dtype=torch.float32)
# print(agent.get_action(state, noise=False))


# agent.update_policy(state)

# # %%
# # from torchviz import make_dot

# # # TODO: find performance bottleneck in torch

# # state, info = env.reset()
# # state = torch.tensor(state, dtype=torch.float32)
# # action = agent.policy(state)

# # combined = torch.cat((state, action), dim=-1)

# # qval = agent.q1(combined)

# # grad_plot = make_dot(qval, show_attrs=True, show_saved=True)

# # #save grad plot
# # grad_plot.format = "png"
# # grad_plot.render("grad_plot")


