import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from replay_buffer import ReplayBufferPrioritized as ReplayBuffer
from ornstein_uhlenbeck import OUActionNoise
from scaler import Scaler
from feedforward import FeedforwardNetwork
from time import perf_counter
import gymnasium as gym
from matplotlib import pyplot as plt

def to_torch(x, device="cpu"):
    # check if x is torch tensor
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.float32)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)

def weighted_smooth_l1_loss(x, y, weights):
    # x, y: torch tensors of shape (batch_size, ...)
    # weights: torch tensor of shape (batch_size
    # returns: weighted smooth l1 loss
    if weights is None:
        weights = torch.ones_like(x)
    diff = x - y
    loss = torch.where(
        torch.abs(diff) < 1,
        0.5 * weights * diff ** 2,
        (torch.abs(diff) - 0.5)*weights,
    )
    return loss.mean()

class TD3:
    def __init__(
        self,
        env,
        env_string,
        gamma=0.99,
        tau_actor=0.005,
        tau_critic=0.005,
        zeta=0.97, # for setting beta
        lr_q=1e-3,
        lr_pol=1e-3,
        wd_q=0.0001,
        wd_pol=0.0001,
        batch_size=100,
        steps_max=1000,
        start_steps=10,
        gradient_steps=-1,
        update_every=1,
        policy_update_freq=2,
        epochs=300,
        buffer_size=10000,
        action_noise_scale=0.1,
        target_action_noise_scale=0.2,
        target_action_noise_clip=0.5,
        h=256,
        test_interval=10,
        play_interval=1000,
        prioritized_replay=True,
        noise_mode="ornstein-uhlenbeck",
        hockey_opponent=None,
    ):
        self.device = "cpu"
        self.debug = False
        self.env = env
        self.env_string = env_string
        self.gamma = gamma
        self.rho_actor = 1 - tau_actor
        self.rho_critic = 1 - tau_critic
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.noise_mode = noise_mode
        self.action_noise_scale = action_noise_scale
        self.target_action_noise_scale = target_action_noise_scale
        self.target_action_noise_clip = target_action_noise_clip
        self.steps_max = steps_max
        self.gradient_steps = gradient_steps
        self.iter_fit = gradient_steps
        self.update_every = update_every
        self.policy_update_freq = policy_update_freq
        self.epochs = epochs
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = 1e-6
        self.zeta = zeta
        self.beta = 1.0 - zeta / 2.0
        self.test_interval = test_interval
        self.play_interval = play_interval
        self.hockey_opponent = hockey_opponent

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, prioritized_replay=prioritized_replay)

        n_obs = env.observation_space.shape[0]
        action_space_shape = env.action_space.shape
        if env_string == "HockeyEnv":
            action_space_shape = (int(env.action_space.shape[0] / 2), )
        n_act = action_space_shape[0]

        q1 = FeedforwardNetwork(n_obs + n_act, 1, act_out=nn.Identity(), h=h)
        q2 = FeedforwardNetwork(n_obs + n_act, 1, act_out=nn.Identity(), h=h)
        pol = FeedforwardNetwork(n_obs, n_act, h=h)

        self.q1 = q1
        self.q2 = q2
        self.policy = pol

        # define target networks
        self.target_q1 = q1.copy()
        self.target_q2 = q2.copy()
        self.target_policy = pol.copy()

        # disable gradients for target networks
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False
        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=lr_q, eps=0.000001, weight_decay=wd_q
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=lr_q, eps=0.000001, weight_decay=wd_q
        )

        self.q_loss = weighted_smooth_l1_loss
        self.pol_loss = lambda x: -((x.mean()) ** 2)

        self.pol_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr_pol, eps=0.000001, weight_decay=wd_pol
        )

        # define scaler
        self.scaler = Scaler(self.env, hockey=env_string=="HockeyEnv")

        # define noise generator
        if self.noise_mode == "ornstein-uhlenbeck":
            self.noise_generator = OUActionNoise(
                mean=np.zeros(action_space_shape),
                std_deviation=self.action_noise_scale
                * np.ones(action_space_shape),
            )
        elif self.noise_mode == "gaussian":
            self.noise_generator = lambda: np.random.normal(
                0, self.action_noise_scale, action_space_shape
            )
        else:
            raise ValueError("Unknown noise mode")

        self.train_iter = 0
        self.total_steps = 0

    def reset(self):
        if self.noise_mode == "ornstein-uhlenbeck":
            self.noise_generator.reset()

    def update_targets(self, rho=None):
        if rho is None:
            self.update_target(self.q1, self.target_q1, rho=self.rho_critic)
            self.update_target(self.q2, self.target_q2, rho=self.rho_critic)
            self.update_target(self.policy, self.target_policy, rho=self.rho_actor)
        else:
            self.update_target(self.q1, self.target_q1, rho=rho)
            self.update_target(self.q2, self.target_q2, rho=rho)
            self.update_target(self.policy, self.target_policy, rho=rho)

    def get_q1(self, state, action):
        unscaled_state = self.scaler.unscale_state(state)
        unscaled_action = self.scaler.unscale_action(action)

        combined = torch.hstack((unscaled_state, unscaled_action))
        us_res = self.q1(combined)
        res = torch.squeeze(us_res)

        return res

    def get_q2(self, state, action):
        unscaled_state = self.scaler.unscale_state(state)
        unscaled_action = self.scaler.unscale_action(action)

        combined = torch.hstack((unscaled_state, unscaled_action))
        us_res = self.q2(combined)
        res = torch.squeeze(us_res)

        return res

    def get_q1_target(self, state, action):
        unscaled_state = self.scaler.unscale_state(state)
        unscaled_action = self.scaler.unscale_action(action)
        return torch.squeeze(
            self.target_q1(torch.hstack((unscaled_state, unscaled_action)))
        )

    def get_q2_target(self, state, action):
        unscaled_state = self.scaler.unscale_state(state)
        unscaled_action = self.scaler.unscale_action(action)
        return torch.squeeze(
            self.target_q2(torch.hstack((unscaled_state, unscaled_action)))
        )

    def get_policy_action(self, state):
        unscaled_state = self.scaler.unscale_state(state)
        return self.policy(unscaled_state)

    def get_target_policy_action(self, state):
        unscaled_state = self.scaler.unscale_state(state)
        target_action = self.target_policy(unscaled_state)
        # add normal noise
        noise = to_torch(
            torch.normal(0, self.target_action_noise_scale, size=target_action.shape),
            device=self.device,
        )
        clamped_noise = torch.clamp(
            noise, -self.target_action_noise_clip, self.target_action_noise_clip
        )
        return torch.clamp(
            target_action + clamped_noise,
            self.scaler.action_low,
            self.scaler.action_high,
        )

    def update_q(self, state, action, reward, next_state, done):
        # torch.cuda.empty_cache()
        # print time
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)

        debug = False
        start = 0
        if debug:
            start = perf_counter()
        # compute target
        act_target = self.get_target_policy_action(next_state)
        if debug:
            print(f"act target took: {perf_counter() - start}")

        with torch.no_grad():
            q1_target_next = self.get_q1_target(next_state, act_target).detach()
            q2_target_next = self.get_q2_target(next_state, act_target).detach()

        if debug:
            print(f"q1, q2 targets took: {perf_counter() - start}")

        q_target_next = torch.minimum(q1_target_next, q2_target_next)

        if debug:
            print(f"q target min took: {perf_counter() - start}")

        target = torch.squeeze(
            reward + self.gamma * (1 - done) * q_target_next
        ).detach()

        if debug:
            print(f"q target calc took: {perf_counter() - start}")

        # compute predictions
        pred1 = self.get_q1(state, action)
        pred2 = self.get_q2(state, action)

        if debug:
            print(f"q1, q2 preds took: {(perf_counter() - start)*self.iter_fit}")

        if self.prioritized_replay:
            # calculate importance sampling weights
            probs = self.replay_buffer.get_last_probs()
            weights = (1 / (probs * self.replay_buffer.size)) ** self.beta
            weights = to_torch(weights / np.max(weights))
        else:
            weights = None

        # compute losses & backpropagate
        loss1 = self.q_loss(pred1, target, weights=weights)
        loss2 = self.q_loss(pred2, target, weights=weights)

        if debug:
            print(f"q1, q2 loss took: {(perf_counter() - start)*self.iter_fit}")

        loss1.backward(inputs=list(self.q1.parameters()), retain_graph=False)
        loss2.backward(inputs=list(self.q2.parameters()), retain_graph=False)

        if debug:
            print(f"q1, q2 backprop took: {(perf_counter() - start)*self.iter_fit}")

        self.q1_optimizer.step()
        self.q2_optimizer.step()

        if debug:
            print(f"q1, q2 opt step took: {(perf_counter() - start)*self.iter_fit}")

        res = (loss1.item() + loss2.item()) / 2

        # set replay buffer priorities
        if self.prioritized_replay:
            td_error1 = torch.abs(pred1 - target).detach().cpu().numpy()
            td_error2 = torch.abs(pred2 - target).detach().cpu().numpy()
            td_error = (td_error1 + td_error2) / 2
            priorities = (np.abs(td_error)) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(priorities=priorities)

        return res

    def update_policy(self, state):
        self.pol_optimizer.zero_grad(set_to_none=True)
        # compute loss
        pol_action = self.get_policy_action(state)

        q_pol = self.get_q1(state, pol_action)

        loss = -q_pol.mean()
        # backpropagate
        loss.backward(inputs=list(self.policy.parameters()), retain_graph=False)
        self.pol_optimizer.step()

        return loss.item()

    def update_step(self, inds=None):
        self.train_iter += 1

        # start = perf_counter()
        # sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(
            inds=inds, batch_size=self.batch_size
        )
        # sample_time = perf_counter() - start
        # print(f"sample took: {sample_time*self.iter_fit}, buffer size: {self.replay_buffer.size}")

        state = to_torch(state, device=self.device)
        action = to_torch(action, device=self.device)
        reward = to_torch(reward, device=self.device)
        next_state = to_torch(next_state, device=self.device)
        done = to_torch(done, device=self.device)

        # print(f"state: {state.shape}, action: {action.shape}, reward: {reward.shape}, next_state: {next_state.shape}, done: {done.shape}")
        # # print dtypes
        # print(f"state: {state.dtype}, action: {action.dtype}, reward: {reward.dtype}, next_state: {next_state.dtype}, done: {done.dtype}")
        # print(reward)

        # start = perf_counter()
        # update q network
        critic_loss = self.update_q(state, action, reward, next_state, done)
        # end = perf_counter()
        # q_time = end - start
        # print(f"Q time: {q_time*self.iter_fit:.3f}")

        actor_loss = None
        if self.train_iter % self.policy_update_freq == 0:
            # start = perf_counter()
            # update policy network
            actor_loss = self.update_policy(state)
            # end = perf_counter()
            # pol_time = end - start
            # print(f"Policy time: {pol_time*self.iter_fit:.3f}")

        self.update_targets()

        return actor_loss, critic_loss

    def get_action(self, state, noise=True):
        if self.total_steps < self.start_steps:
            action = self.env.action_space.sample()
            if self.env_string == "HockeyEnv":
                action = action[:4]
        else:
            state = to_torch(state)
            action = self.get_policy_action(state).detach()
            if noise:
                noise = self.noise_generator()
                action = action + noise
                if self.scaler.action_scaling:
                    action = torch.clamp(action, -1, 1)
            # scale from [-1, 1] to action space scales
            action = self.scaler.scale_action(action).numpy()
        if self.env_string == "HockeyEnv" and np.sum(action.shape) > 4:
            print("Action too big !!!!!!!!!!!!!")
        return action

    def train(self):

        hockey_opponent = self.hockey_opponent

        if (self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv") and hockey_opponent is None:
            raise ValueError("Hockey opponent must be specified")

        self.total_steps = 0

        episode_rewards = []
        test_rewards = []
        episode_steps = []
        ep_durations = []
        upd_durations = []
        ep_actor_losses = []
        ep_actor_losses_first = []
        ep_actor_losses_last = []

        ep_critic_losses = []
        ep_critic_losses_first = []
        ep_critic_losses_last = []

        test_interval = self.test_interval
        play_interval = self.play_interval
        try:
            # for epoch in tqdm(range(self.epochs)):
            for epoch in range(self.epochs):
                if epoch > 0 and epoch % test_interval == 0:
                    test_reward = np.mean([self.test(noise_enabled=False, render_mode=None) for _ in range(10)])
                    test_rewards.append(test_reward)
                    # print reward without breaking tqdm
                    print(
                        f"""Epoch {epoch}:
                                rewards: train: {episode_rewards[-1]:.2f}, test: {test_reward:.2f}
                                            max train reward: {np.max(episode_rewards[-test_interval:]):.2f}
                                avg_steps: {np.mean(episode_steps[-test_interval:]):.2f}
                                losses:
                                    actor: mean: {np.mean(ep_actor_losses[-test_interval:]):.2f}, first: {np.mean(ep_actor_losses_first[-test_interval:]):.2f}, last: {np.mean(ep_actor_losses_last[-test_interval:]):.2f}
                                    critic: mean: {np.mean(ep_critic_losses[-test_interval:]):.2f}, first: {np.mean(ep_critic_losses_first[-test_interval:]):.2f}, last: {np.mean(ep_critic_losses_last[-test_interval:]):.2f}
                                avg_duration: {np.mean(ep_durations[-test_interval:]):.2f}, upd_duration: {np.mean(upd_durations[-test_interval:]):.2f}
                    """
                    )
                if epoch > 0 and epoch % (play_interval) == 0:
                    self.test(noise_enabled=False, render_mode="human")

                start_ep = perf_counter()

                state, info = self.env.reset()
                if self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv":
                    state_opponent = self.env.obs_agent_two()
                self.reset()
                done = False
                steps = 0
                rewards = []

                # start = perf_counter()
                while not done and steps < self.steps_max:
                    # get action
                    # start_act = perf_counter()
                    action = self.get_action(state)
                    # print(f"get action took: {(perf_counter() - start_act)*self.steps_max}")
                    # take action, save transition
                    if self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv":
                        action_opponent = hockey_opponent.act(state_opponent)
                        next_state, reward, terminated, truncated, _ = self.env.step(np.hstack((action, action_opponent)))
                    else:
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    if self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv":
                        state_opponent = self.env.obs_agent_two()

                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                    self.total_steps += 1
                    rewards.append(reward)

                # print(f"Episode w/o updt took: {perf_counter() - start}")

                ep_duration = perf_counter() - start_ep
                ep_durations.append(ep_duration)

                episode_steps.append(steps)
                ep_reward = np.sum(rewards)
                episode_rewards.append(ep_reward)

                self.beta = 1.0 - self.zeta**epoch / 2

                if (
                    self.total_steps > self.batch_size
                    and epoch % self.update_every == 0
                ):
                    actor_losses = []
                    critic_losses = []
                    start_up = perf_counter()
                    
                    if self.gradient_steps < 0:
                        self.iter_fit = steps

                    # sampling indices beforehand is faster, but limits prioritized replay until buffer has iter_fit*batch_size samples
                    if not self.prioritized_replay:
                        inds_all = self.replay_buffer.sample_inds(
                            (self.iter_fit, self.batch_size)
                        )
                    else:
                        inds_all = None

                    for i in range(self.iter_fit):
                        if inds_all is not None:
                            inds = inds_all[i]
                        else:
                            inds = self.replay_buffer.sample_inds((self.batch_size))
                        # inds = sample_inds[i]

                        actor_loss, critic_loss = self.update_step(inds=inds)
                        if actor_loss is not None:
                            actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                    # print(f"Update step took: {perf_counter() - start_up}")
                    upd_duration = perf_counter() - start_up
                    upd_durations.append(upd_duration)

                    actor_loss_mean = np.mean(actor_losses)
                    critic_loss_mean = np.mean(critic_losses)

                    actor_loss_first = actor_losses[0]
                    critic_loss_first = critic_losses[0]

                    actor_loss_last = actor_losses[-1]
                    critic_loss_last = critic_losses[-1]

                    ep_actor_losses_first.append(actor_loss_first)
                    ep_actor_losses_last.append(actor_loss_last)

                    ep_critic_losses_first.append(critic_loss_first)
                    ep_critic_losses_last.append(critic_loss_last)

                    ep_actor_losses.append(actor_loss_mean)
                    ep_critic_losses.append(critic_loss_mean)

                    # print(
                    #     f"Epoch {epoch}: ep reward: {episode_rewards[-1]}, actor loss: {actor_loss_mean:.2f}, critic loss: {critic_loss_mean:.2f}, duration: {ep_duration:.2f}"
                    # )

        except KeyboardInterrupt:
            return (
                episode_rewards,
                test_rewards,
                episode_steps,
                ep_durations,
                upd_durations,
                ep_actor_losses,
                ep_critic_losses,
            )
        return (
            episode_rewards,
            test_rewards,
            episode_steps,
            ep_durations,
            upd_durations,
            ep_actor_losses,
            ep_critic_losses,
        )

    def test(self, noise_enabled=False, render_mode=None):
        hockey_opponent = self.hockey_opponent
        # set render mode
        env = None
        if self.env_string == "LunarLander-v2":
            env = gym.make(self.env_string, continuous=True, render_mode=render_mode)
        elif self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv":
            if hockey_opponent is None:
                raise ValueError("Hockey opponent must be specified")
            return self.hockey_test(render_mode=render_mode, noise_enabled=noise_enabled)
        else:
            env = gym.make(self.env_string, render_mode=render_mode)

        state, info = env.reset()
        done = False
        step = 0
        rewards = []
        while not done and step < self.steps_max:
            action = self.get_action(state, noise=noise_enabled)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            step += 1
            if render_mode == "human" and self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv":
                env.render(mode="human")
        if render_mode == "human" and not (self.env_string == "HockeyEnv" or self.env_string == "LaserHockeyEnv"):
            # close window
            env.close()
        return np.sum(rewards)

    def hockey_test(self, noise_enabled=False, render_mode=None):
        hockey_opponent = self.hockey_opponent
        env = self.env

        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        done = False
        step = 0
        rewards = []
        while not done and step < self.steps_max:
            action = self.get_action(state, noise=noise_enabled)
            action_opponent = hockey_opponent.act(obs_agent2)
            state, reward, terminated, truncated, _ = env.step(np.hstack((action, action_opponent)))
            done = terminated or truncated
            obs_agent2 = env.obs_agent_two()
            rewards.append(reward)
            step += 1
            if render_mode == "human":
                env.render(mode="human")
        return np.sum(rewards)


    @staticmethod
    def update_target(net, target, rho=0.995):
        # get state dicts
        target_state_dict = target.state_dict()
        net_state_dict = net.state_dict()
        # update target state dict
        for key in target_state_dict.keys():
            target_state_dict[key] = (
                rho * target_state_dict[key] + (1 - rho) * net_state_dict[key]
            )
        # load target state dict
        target.load_state_dict(target_state_dict)
