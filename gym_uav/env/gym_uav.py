import gym
import math

import numpy as np

from gym_uav.env.utils import RawEnv, GameStateType

from gym_uav.env.config import environ_config


class GymUav(gym.Env):
    def __init__(self, size_len=100.0, obs_interval=20.0,
                 obs_percentage=0.2, obs_radius=7.0, agent_speed=2.0,
                 compute_per_second=10, minimum_dist_to_destination=10,
                 agent_initial_ori=0.25, alpha=8.0, sigma=2.0, gamma=2.0,
                 r_free=0.1, r_step=-0.6, sensor_max_dist=25.0, reward_params=None):
        if reward_params is not None:
            self.reward_params = reward_params
        else:
            self.reward_params = environ_config.reward_params
        beta = self.reward_params[2]
        self.reward_params = (*self.reward_params[0:2], self.reward_params[3], 1)
        # self.reward_params = (*self.reward_params[0:2], self.reward_params[3], 1, self.reward_params[4])
        self.raw_env = RawEnv(
            size_len, obs_interval, obs_percentage, obs_radius, agent_speed,
            compute_per_second, minimum_dist_to_destination, agent_initial_ori,
            alpha, beta, sigma, gamma, r_free, r_step, sensor_max_dist
        )

        # [theta, speed]
        self.action_space = gym.spaces.Box(
            np.array([-0.25, 0.5], dtype=np.float32),
            np.array([0.25, 3.5], dtype=np.float32)
        )
        # self.action_space = gym.spaces.Box(np.array([-0.25], dtype=np.float32), np.array([0.25], dtype=np.float32))
        # [d0 -> d8, d9, theta, omega]
        self.observation_space = gym.spaces.Box(
            np.array([0.] * 9 + [-math.inf, 0, -1], dtype=np.float32),
            np.array([sensor_max_dist] * 9 + [math.inf, 2, 1], dtype=np.float32)
        )
        self._last_obs = None

    def seed(self, seed=None):
        pass

    def step(self, action):
        self.raw_env.act(action)

        done = self.raw_env.is_ended()
        if done == GameStateType.CollideObs or done == GameStateType.CollideWall:
            observation = [0] * 12
        else:
            observation = self.raw_env.observe()

        # for non-human render
        self._last_obs = observation

        # r_tran, r_bar, r_free, r_step
        raw_reward = self.raw_env.raw_reward()
        # print('tans: %.4f; bar: %.4f; free: %.4f; step: %.2f' % raw_reward)
        reward = 0
        for x, y in zip(raw_reward, self.reward_params):
            reward += x * y

        self.raw_env.update_info(done)

        # if done.value == 1:
        #     reward = 1e9
        # elif done:
        #     reward = -1e6

        return observation, reward, done.__bool__(), self.raw_env.info

    def reset(self, obs_map=None):
        self.raw_env.info.clear()
        self.raw_env.reset(obs_map)
        return self.raw_env.observe()

    def render(self, mode='human'):
        if mode == 'human':
            self.raw_env.render()
        else:
            print(f'agent_pos: {self.raw_env.agent_current_pos}')
            print(f'agent_ori: {self.raw_env.agent_orientation}')
            print(f'obs: {self._last_obs}')
