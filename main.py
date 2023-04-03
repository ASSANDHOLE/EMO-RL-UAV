import sys
from multiprocessing import set_start_method
import pickle

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_termination

from gym_uav import NamedDict
from uav_problem import set_problem_config
from uav_problem import UavProblem

# TODO: set the config here
# time_step: the total time steps for each individual training
# eval_time: the number of maps for each individual evaluation
# pop_size: the population size and offspring size for each generation
# available_devices: the devices used for training,
#  'all' for all available devices,
#  otherwise a list of device ids, e.g., [0, 1, 3]
config = NamedDict({
    'num_proc': 32, 'time_step': 200000, 'eval_time': 200,
    'available_devices': 'all', 'pop_size': 32  # offspring_size=pop_size
})

set_problem_config(config)


def run():
    def dump_fn(x):
        try:
            with open('./eval_his.pkl', 'wb') as pkl:
                pickle.dump(x, pkl)
        except:
            try:
                with open('./eval_his_back.pkl', 'wb') as pkl:
                    pickle.dump(x, pkl)
                    print('dumped to backup file: eval_his_back.pkl')
            except:
                print('dump eval_history failed')

    def dump_reward_fn(x):
        try:
            with open('./reward_his.pkl', 'wb') as pkl:
                pickle.dump(x, pkl)
        except:
            try:
                with open('./reward_his_back.pkl', 'wb') as pkl:
                    pickle.dump(x, pkl)
                    print('dumped to backup file: reward_his_back.pkl')
            except:
                print('dump reward_his failed')

    problem = UavProblem(save_eval=True, dump_fn=dump_fn, dump_record_fn=dump_reward_fn)
    # TODO: set the initial sampling here
    init_arr = np.array([
        # [dist, bar, (beta), free], (step == 1 (normalized thus omitted))
        # [ 1.0,  1.0, 25.0,  1.0],  # baseline
        [14.0, 14.0, 22.0, 22.0],
        [ 7.0,  6.0,  5.0,  4.0],
        [ 8.0,  1.0,  2.0,  5.0],
        [25.0,  3.0,  2.0,  1.0],
        [ 1.0,  1.2, 22.0,  1.0],
        [ 1.0,  1.0,  1.0,  2.0],
        [ 0.5,  1.5,  2.5,  3.5]
    ])
    algorithm = NSGA2(
        pop_size=int(config.pop_size),
        n_offsprings=int(config.pop_size),
        sampling=init_arr,
        # TODO: set the crossover and mutation here
        crossover=get_crossover('real_sbx', prob=0.75, eta=10),
        mutation=get_mutation('real_pm', prob=0.5, eta=10),
        eliminate_duplicates=True
    )
    # TODO: set the termination here
    termination = get_termination('n_gen', 100)
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=True,
        verbose=True
    )
    for sub in res.history:
        sub.problem = None
    res.problem = None
    try:
        with open('./res_mo.pkl', 'wb') as f:
            pickle.dump(res, f)
        print('dump final res to ./res_mo.pkl')
    except:
        try:
            with open('./res_mo_back.pkl', 'wb') as f:
                pickle.dump(res, f)
            print('dump final res to ./res_mo_back.pkl')
        except:
            print('dump final res failed')

    try:
        with open('./eval_his.pkl', 'wb') as f:
            pickle.dump(problem.eval_res, f)
        print('dump final eval_history to ./eval_his.pkl')
    except:
        try:
            with open('./eval_his_back.pkl', 'wb') as f:
                pickle.dump(problem.eval_res, f)
            print('dump final eval_history to ./eval_his_back.pkl')
        except:
            print('dump final eval_history failed')

    return res, problem

 
if __name__ == '__main__':
    if sys.platform == 'linux':
        set_start_method('spawn')
    run()
