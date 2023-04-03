import numpy as np

from pymoo.core.problem import Problem

from uav_problem.eval_iteration import multiprocessing_one_generation
from uav_problem.problem_config import get_config
from gym_uav import gen_obs_map


class MyProblem(Problem):
    def __init__(self, save_eval=False, dump_fn=None, dump_record_fn=None):
        lower_bound = np.array([0.01, 0.01, 0.01, 0.01])
        upper_bound = np.array([100.0, 100.0, 100.0, 100.0])
        self.config = get_config()
        self.eval_obs_map = [gen_obs_map(4, 0.5) for _ in range(self.config['eval_time'])]
        self.save_eval = save_eval
        self.eval_res = []
        self.reward_records = []
        # constraint: no collide with obstacle or wall, success rate >= ?%
        problem_kwargs = {
            'n_var': 4, 'n_obj': 2, 'n_constr': 2,
            'xl': lower_bound, 'xu': upper_bound
        }
        assert not (not save_eval and dump_fn is not None), '"dump_fn" is not None while save_eval is false'
        self.dump_fn = dump_fn
        self.dump_record_fn = dump_record_fn
        super().__init__(**problem_kwargs)

    def run(self, x):
        info_list = multiprocessing_one_generation(
            self.config.num_proc, x, self.config.time_step,
            self.eval_obs_map, self.config.available_devices
        )
        res = []
        rewards_records = []
        for x in info_list:
            suc = 0
            cli = 0
            tlr = 0
            path_ratio = []
            speed_ratio = []
            for y in x[1][0]:
                if y.done.value == 1:
                    suc += 1
                    path_ratio.append(y.length_shortest_ratio)
                    speed_ratio.append(y.speed_maximun_ratio)
                elif y.done.value == 2 or y.done.value == 3:
                    cli += 1
                else:
                    tlr += 1
            r_l = [y.risk_factor for y in x[1][0]]
            r_max = max(r_l)
            if len(path_ratio) > 0:
                path_ratio_mean = sum(path_ratio) / len(path_ratio)
            else:
                path_ratio_mean = 1000000
            if len(speed_ratio) > 0:
                speed_ratio_mean = sum(speed_ratio) / len(speed_ratio)
            else:
                speed_ratio_mean = 1000000
            # efficiency (smaller is better)
            obj1 = 0.5*path_ratio_mean + 0.5*speed_ratio_mean
            # safety (smaller is better)
            obj2 = r_max
            # no collision
            con1 = cli - 0.1
            # success rate >= 70%
            con2 = 0.69 - suc / len(x[1][0])
            res.append([obj1, obj2, con1, con2])
            rewards_records.append((x[0], x[1][1]))
        return np.array(res), rewards_records

    def _evaluate(self, x, out, *args, **kwargs):
        res, reward_records = self.run(x)
        f = np.column_stack([res[:, 0], res[:, 1]])
        g = np.column_stack([res[:, 2], res[:, 3]])
        if self.save_eval:
            self.eval_res.append((x, res, f, g))
            self.reward_records.extend(reward_records)
            if self.dump_fn is not None:
                self.dump_fn(self.eval_res)
            if self.dump_record_fn is not None:
                self.dump_record_fn(self.reward_records)
        out['F'] = f
        out['G'] = g

