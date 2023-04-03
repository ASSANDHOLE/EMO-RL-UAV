import gc
import os
import time
from copy import deepcopy
from multiprocessing import Pool, Manager

import gym
import numpy as np
import torch

from uav_problem.scheduler import GpuResourceScheduler

import gym_uav  # Don't remove this line, it is used to register the env
from ddpg_pytorch.ddpg import DDPG
from ddpg_pytorch.utils.noise import OrnsteinUhlenbeckActionNoise
from ddpg_pytorch.utils.replay_memory import ReplayMemory, Transition
from ddpg_pytorch.wrappers.normalized_actions import NormalizedActions

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def multiprocessing_one_generation(num_proc, params, time_step, eval_obs_map, available_devices='all'):
    """
    info: NamedDict({'done': GameStateType, 'risk_factor': float, 'total_step': int, 'length_shortest_ratio': float})
    :param num_proc: numbers of processors. i.e. Pool(number_proc)
    :param params: list of param, param -> (trans, bar_alpha, bar_beta, free, (step == 1))
    :param time_step: train for how much steps
    :param eval_obs_map: obs map used to evaluate, eval_epoch = len(eval_obs_map)
    :param available_devices: string 'all' or list of int. 'all' for all gpus, list of ints for cuda:<int>
    :return: list of [( (*param), [#eval_epoch info] ),]
    """
    assert num_proc >= 1
    assert type(num_proc) is int
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if available_devices == 'all':
            gpu_count = torch.cuda.device_count()
            available_devices = [i for i in range(gpu_count)]
    with Manager() as man:
        lock = man.Lock()
        param_list = []
        gpu_scheduler = GpuResourceScheduler(available_devices, lock, 5)
        for param in params:
            param_list.append((param, time_step, eval_obs_map, gpu_scheduler))
        if num_proc > 1:
            # ensure process is killed once finished to free GPU memory
            with Pool(processes=num_proc, maxtasksperchild=1) as p:
                res = p.starmap(_train_and_get_info, param_list)
        else:
            res = [_train_and_get_info(*param_list[0])]

    gpu_scheduler.delete()

    ret = []
    for re, pa in zip(res, params):
        ret.append((pa, re))

    return ret


def _train_and_get_info(param, time_steps, eval_obs_map, gpu_scheduler: GpuResourceScheduler):
    checkpoints_dir = None
    with gpu_scheduler.context_assign_id() as gpu_idx:
        device = torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = gym.make(
            'Multi-Objective-Uav-v0',
            size_len=30, obs_percentage=0.4,
            obs_interval=6, obs_radius=2,
            minimum_dist_to_destination=1,
            sensor_max_dist=12, reward_params=param
        )
        env = NormalizedActions(env)

        hidden_size = (400, 300)
        agent = DDPG(0.99,
                     0.001,
                     hidden_size,
                     env.observation_space.shape[0],
                     env.action_space,
                     device=device,
                     checkpoint_dir=checkpoints_dir
                     )
        nb_actions = env.action_space.shape[-1]
        ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                sigma=0.2 * np.ones(nb_actions))
        memory = ReplayMemory(int(1e5))
        timestep = 0
        info_list = []
        epoch_rewards = []
        while timestep <= time_steps:
            # print(timestep)
            ou_noise.reset()
            epoch_return = 0
            state = torch.Tensor([env.reset()]).to(device)
            while True:
                action = agent.calc_action(state, ou_noise)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                timestep += 1
                epoch_return += reward

                mask = torch.Tensor([done]).to(device)
                reward = torch.Tensor([reward]).to(device)
                next_state = torch.Tensor([next_state]).to(device)

                memory.push(state, action, mask, next_state, reward)

                state = next_state

                epoch_value_loss = 0
                epoch_policy_loss = 0

                if len(memory) > 64:
                    transitions = memory.sample(64)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = agent.update_params(batch)

                    epoch_value_loss += value_loss
                    epoch_policy_loss += policy_loss
                if done:
                    break
            epoch_rewards.append(epoch_return)

        agent.set_eval()

        for i in range(len(eval_obs_map)):
            state = torch.Tensor([env.reset(obs_map=eval_obs_map[i])]).to(device)
            episode_return = 0
            while True:
                action = agent.calc_action(state, action_noise=None)
                agent.critic(state, action)
                next_state, reward, done, info = env.step(action.cpu().numpy()[0])
                episode_return += reward
                state = torch.Tensor([next_state]).to(device)
                if done:
                    info_list.append(deepcopy(info))
                    break
        env.close()

        # return resources and free memory
        del agent
        del env
        # torch.cuda.empty_cache()
        gc.collect()

    return info_list, epoch_rewards
