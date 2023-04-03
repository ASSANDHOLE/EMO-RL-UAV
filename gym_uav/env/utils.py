import math
import random
from enum import Enum
from typing import Any

import numpy
import numpy as np

from gym_uav.env.config import NamedDict
from gym_uav.env.c_functions import c_dijkstra, c_obs_dist_circle

import os
if 'DISPLAY' in os.environ:
    try:
        import pygame
        pygame.init()
    except ImportError:
        pygame = None
else:
    pygame = None


class GameStateType(Enum):
    Progress = 0
    ReachDest = 1
    CollideObs = 2
    CollideWall = 3
    TimeLimitReached = 4

    def __bool__(self):
        return self != self.Progress


def l2_norm(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def gen_obs_coordinate(size, interval):
    amount = int(size / interval) - 1
    res = numpy.zeros((2, amount, amount))
    base = interval
    for i in range(amount):
        for j in range(amount):
            res[0][i][j] = base + i * interval
            res[1][i][j] = base + j * interval
    return np.array(res, dtype=np.float32)


def gen_start_and_ending_coordinate(size, interval):
    amount = int(size / interval)
    res = numpy.zeros((2, amount, amount))
    base = interval / 2
    for i in range(amount):
        for j in range(amount):
            res[0][i][j] = base + i * interval
            res[1][i][j] = base + j * interval
    return np.array(res, dtype=np.float32)


def gen_obs_map(obs_coordinate_size, percent):
    start = int(obs_coordinate_size * obs_coordinate_size * (1 - percent))
    target = np.random.random(obs_coordinate_size * obs_coordinate_size)
    obs_coordinate = target.argsort()[start:]
    target = np.zeros(obs_coordinate_size * obs_coordinate_size, dtype=bool)
    for i in obs_coordinate:
        target[i] = True
    return target.reshape((obs_coordinate_size, obs_coordinate_size))


class RawEnv:
    """The Class to handle UAV test environment.
        * * * * (n,n)
        * * * * *
        * * * * *
        * * * * *
    (0,0) * * * *

    agent_orientation -- orientation form agent to destination, regularized from (0, 2*pi) to (0, 2)

    ---------------
    agent->dest ori=0
    ---------------
           dest
    agent ^     ori=1/4
    ---------------
    """

    def __init__(self, size_len, obs_interval, obs_percentage,
                 obs_radius, agent_initial_speed, compute_per_second,
                 minimum_dist_to_destination, agent_initial_ori,
                 alpha, beta, sigma, gamma, r_free, r_step,
                 sensor_max_dist):
        # assertions
        assert obs_interval > 2 * obs_radius
        assert type(compute_per_second) == int
        assert minimum_dist_to_destination > 0
        # params
        self.size_len = size_len
        self.obs_interval = obs_interval
        self.obs_radius = obs_radius
        self.obs_percentage = obs_percentage
        # shape = (2, n_obs, n_obs), obs_coordinate[0] is the x coordinate of obstacles, [1] is y
        self.obs_coordinate = gen_obs_coordinate(size_len, obs_interval)
        self.start_and_ending_coordinate = gen_start_and_ending_coordinate(size_len, obs_interval)
        # shape = (n_obs, n_obs), dtype=bool. if obs_map[a][b], there is a obstacle at obs_coordinate[:, a, b]
        self.obs_map = gen_obs_map(self.obs_coordinate.shape[1], self.obs_percentage)
        self.agent_speed = agent_initial_speed
        self.agent_initial_speed = agent_initial_speed
        self.agent_speed_list = []
        self.agent_path_len = 0
        self.compute_per_second = compute_per_second
        self.dist_per_move = self.agent_speed / self.compute_per_second
        self.minimum_dist_to_destination = minimum_dist_to_destination
        self.agent_initial_ori = agent_initial_ori
        self.agent_orientation = self.agent_initial_ori
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.r_free = r_free
        self.r_step = r_step
        self.sensor_max_dist = sensor_max_dist
        self.agent_current_pos = [10.0, 10.0]
        self.destination_pos = (size_len, size_len)
        self._min_dist_with_obs = self.obs_interval
        self._closest_obs_or_wall = self._min_dist_with_obs
        self._last_dist_to_dest = l2_norm(self.agent_current_pos, self.destination_pos)
        self.info = NamedDict({})
        self._init_info()
        self._time_step = 0
        self._max_time = 0
        self._shortest_path = 0
        self.screen = None
        self._d4 = 0
        self._last_r_bar = 0
        self._last_speed = agent_initial_speed

    def set_max_time(self, rate=6):
        # only after '_generate_shortest_path' called
        self._max_time = int(self._shortest_path / self.agent_speed * self.compute_per_second * rate)
        #self._max_time = int(self._shortest_path / ((0.5+3.5)/2) * self.compute_per_second * rate)

    def act(self, action):
        """Act depend on action for time=1/self.compute_per_second s
        :param action: steering signal, range from -1/4 to 1/4 represent -pi/4 to pi/4
        :return: None
        """
        try:
            action_ori = action[0]
            action_speed = action[1]
        except TypeError:
            pass
        self._time_step += 1
        raw_ori = self.agent_orientation
        raw_ori += action_ori
        if raw_ori > 2:
            raw_ori -= 2
        elif raw_ori < 0:
            raw_ori += 2
        self.agent_orientation = raw_ori
        # self.agent_current_pos[0] += math.cos(self.agent_orientation * math.pi) * self.dist_per_move
        # self.agent_current_pos[1] += math.sin(self.agent_orientation * math.pi) * self.dist_per_move
        self.agent_speed = action_speed
        self.agent_speed_list.append(action_speed)
        self.dist_per_move = self.agent_speed / self.compute_per_second
        self.agent_path_len += self.dist_per_move
        self.agent_current_pos[0] += math.cos(self.agent_orientation * math.pi) * self.dist_per_move
        self.agent_current_pos[1] += math.sin(self.agent_orientation * math.pi) * self.dist_per_move

    def _collide_with_obs(self):
        x_dist = (self.obs_coordinate[0] - self.agent_current_pos[0]) ** 2
        y_dist = (self.obs_coordinate[1] - self.agent_current_pos[1]) ** 2
        final = x_dist + y_dist
        final = final[self.obs_map]
        self._min_dist_with_obs = np.sqrt(np.min(final)) - self.obs_radius
        square_radius = self.obs_radius ** 2
        return np.sum(final <= square_radius) > 0

    def is_ended(self):
        """Check the game state
        :return: One of the type in enum class `GameStateType`
        """
        if self._time_step > self._max_time:
            return GameStateType.TimeLimitReached
        if not (0 < self.agent_current_pos[0] < self.size_len and
                0 < self.agent_current_pos[1] < self.size_len):
            return GameStateType.CollideWall

        if self._collide_with_obs():
            return GameStateType.CollideObs

        if l2_norm(self.destination_pos, self.agent_current_pos) <= self.minimum_dist_to_destination:
            return GameStateType.ReachDest

        return GameStateType.Progress

    def _obs_in_range(self, _range):
        obs = []
        crds = self.obs_coordinate
        cpos = self.agent_current_pos
        for i in range(crds.shape[1]):
            for j in range(crds.shape[1]):
                if not self.obs_map[i][j]:
                    continue
                dist = ((crds[0][i][j] - cpos[0]) ** 2 + (crds[1][i][j] - cpos[1]) ** 2) ** (1 / 2)
                if dist < _range:
                    obs.append([i, j])
        return obs

    def _obs_sensor_observe(self, angle):
        """
        In industrial situations, obstacles are often represented as LiDAR point clouds,
        i.e., obstacles are known to the system with coordinates.
        So, here, we make obstacles' positions(centers and radia), known to the system.
        Initialize dist as max_dist
        :param angle: range from -1, 1 meaning (-pi, pi)
        :return:
        """
        sensor_angle = self.agent_orientation + angle / 2
        dist = self.sensor_max_dist
        # Hit wall situation
        cpos = self.agent_current_pos
        end = [cpos[0] + math.cos(sensor_angle * math.pi) * self.sensor_max_dist,
               cpos[1] + math.sin(sensor_angle * math.pi) * self.sensor_max_dist]
        if end[0] < 0 or end[0] > self.size_len or end[1] < 0 or end[1] > self.size_len:
            if end[0] < 0:
                end[1] = end[1] - end[0] * (cpos[1] - end[1]) / (cpos[0] - end[0] + 1e-8)
                end[0] = 0
            elif end[0] > self.size_len:
                end[1] = end[1] - (end[0] - self.size_len) * (cpos[1] - end[1]) / (cpos[0] - end[0] + 1e-8)
                end[0] = self.size_len
            if end[1] < 0:
                end[0] = end[0] - end[1] * (cpos[0] - end[0]) / (cpos[1] - end[1] + 1e-8)
                end[1] = 0
            elif end[1] > self.size_len:
                end[0] = end[0] - (end[1] - self.size_len) * (cpos[0] - end[0]) / (cpos[1] - end[1] + 1e-8)
                end[1] = self.size_len
            dist = ((end[0] - cpos[0]) ** 2 + (end[1] - cpos[1]) ** 2) ** (1 / 2)
        #  Hit obs situation
        obs = self._obs_in_range(dist + self.obs_radius)

        obs_x = []
        obs_y = []
        for ob in obs:
            obs_x.append(self.obs_coordinate[0][ob[0]][ob[1]])
            obs_y.append(self.obs_coordinate[1][ob[0]][ob[1]])

        dist = c_obs_dist_circle(obs_x, obs_y, cpos[0], cpos[1], end[0], end[1], self.obs_radius, len(obs_x))
        return dist

    def observe(self) -> tuple[Any, float, float, float]:
        obs_sensor = []
        base_angle = -1
        for i in range(9):
            obs_sensor.append(self._obs_sensor_observe(base_angle))
            base_angle += 0.25
        d9 = l2_norm(self.agent_current_pos, self.destination_pos)
        theta = self.agent_orientation
        omega = math.atan2(self.destination_pos[1] - self.agent_current_pos[1],
                           self.destination_pos[0] - self.agent_current_pos[0]) / math.pi
        omega = 2 + omega if omega < 0 else omega
        omega -= self.agent_orientation
        self._d4 = obs_sensor[4]
        return *obs_sensor, d9, theta, omega

    def raw_reward(self) -> tuple[float, float, float, float]:
        r_dist = self.sigma * (self._last_dist_to_dest - l2_norm(self.destination_pos, self.agent_current_pos))
        closest_obs_or_wall = min(
            *self.agent_current_pos,
            self.size_len - self.agent_current_pos[0],
            self.size_len - self.agent_current_pos[1],
            self._min_dist_with_obs
        )
        r_bar = -self.alpha * math.exp(-self.beta * closest_obs_or_wall)
        self._last_r_bar = r_bar
        self._last_dist_to_dest = l2_norm(self.destination_pos, self.agent_current_pos)
        free = self._d4 > 0.95 * self.sensor_max_dist
        self._closest_obs_or_wall = closest_obs_or_wall
        # r_speed = self.gamma * (self.agent_speed - 0.5)
        self._last_speed = self.agent_speed
        return r_dist, r_bar, self.r_free * int(free), self.r_step  # , r_speed

    def _eval_start_end(self, start, end):
        return l2_norm(start, end) > self.size_len / 2

    def _select_start_end(self):
        length = self.start_and_ending_coordinate.shape[1]
        res = random.randrange(0, length * length)
        i, j = res // length, res % length
        start = self.start_and_ending_coordinate[:, i, j]
        res = random.randrange(0, length * length)
        i, j = res // length, res % length
        end = self.start_and_ending_coordinate[:, i, j]
        while not self._eval_start_end(start, end):
            res = random.randrange(0, length * length)
            i, j = res // length, res % length
            end = self.start_and_ending_coordinate[:, i, j]
        return start, end

    def _init_info(self):
        info = self.info
        info.done = GameStateType.Progress
        info.risk_factor = 0
        info.total_step = 0
        info.length_shortest_ratio = float('inf')
        info.speed_maximun_ratio = float('inf')

    def update_info(self, done: GameStateType) -> None:
        info = self.info
        info.done = done
        info.risk_factor = max(info.risk_factor, 1 / self._closest_obs_or_wall ** 2)
        info.total_step += 1
        if done.value == 1:
            info.length_shortest_ratio = self.agent_path_len / self._shortest_path
            info.speed_maximun_ratio = 3.5 / np.mean(self.agent_speed_list)

    def _transfer_to_2d_array(self, start):
        step = self.agent_speed / self.compute_per_second / 2
        arr_len = int(self.size_len // step)
        arr = np.zeros((arr_len, arr_len), dtype=np.uint8)
        obs_list = []
        len_obs = len(self.obs_map)
        for i in range(len_obs):
            for j in range(len_obs):
                if self.obs_map[i][j]:
                    obs_list.append(tuple(self.obs_coordinate[:, i, j]))
        radius = self.obs_radius / step
        x, y = np.ogrid[:arr_len, :arr_len]
        for obs in obs_list:
            center = (obs[0] / step, obs[1] / step)
            dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            mask = dist_from_center <= radius
            arr[mask] = 1

        center = (self.destination_pos[0] // step, self.destination_pos[1] // step)
        arr[int(center[0])][int(center[1])] = 2
        _start = (int(start[0] // step), int(start[1] // step))
        return _start, step, arr

    def _generate_shortest_path(self, start):
        """
        :param start: (float, float) real start coordinate (x, y)
        :return: None
        """
        start, step, arr = self._transfer_to_2d_array(start)
        # shortest_len = self._bfs_shortest_len(*res)
        # self._shortest_path = shortest_len
        self._shortest_path = c_dijkstra(arr, start) * step

    def reset(self, obs_map=None) -> None:
        self._time_step = 0
        if obs_map is not None:
            self.obs_map = obs_map
        else:
            self.obs_map = gen_obs_map(self.obs_coordinate.shape[1], self.obs_percentage)
        start_end = self._select_start_end()
        self.agent_current_pos = [*start_end[0]]
        self.destination_pos = start_end[1]
        self._last_dist_to_dest = l2_norm(self.destination_pos, self.agent_current_pos)
        self.agent_orientation = self.agent_initial_ori
        self._init_info()
        self._generate_shortest_path(start_end[0])
        self.set_max_time()
        self.agent_speed = self.agent_initial_speed
        self.agent_path_len = 0
        self.agent_speed_list = []
        self.dist_per_move = self.agent_speed / self.compute_per_second
        self._last_speed = self.agent_speed

    def render(self) -> None:
        if pygame is None:
            print('PyGame or Display is not available')
            return
        ratio = 1000 / self.size_len
        if self.screen is None:
            self.screen = pygame.display.set_mode([1000, 1000])
        self.screen.fill((255, 255, 255))
        lim = self.obs_coordinate.shape[1]
        for i in range(lim):
            for j in range(lim):
                if self.obs_map[i][j]:
                    pygame.draw.circle(
                        self.screen, (0, 0, 255),
                        (self.obs_coordinate[0][i][j] * ratio,
                         self.obs_coordinate[1][i][j] * ratio),
                        self.obs_radius * ratio)
        pygame.draw.circle(self.screen, (0, 0, 0),
                           (self.agent_current_pos[0] * ratio, self.agent_current_pos[1] * ratio),
                           5)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (self.destination_pos[0] * ratio, self.destination_pos[1] * ratio), 5)
        pygame.display.flip()

