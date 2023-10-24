import copy
import random

import numpy as np



def reduce_samples(samples):
    new_samples = []
    already_sampled ={}
    for s in samples:
        if tuple(s['f']) in already_sampled.keys():
            continue
        else:
            new_samples += [s]
            already_sampled[tuple(s['f'])] = 1
    return new_samples

class User():

    def __init__(self, planner, samples, user_cfg):
        """

        :param planner:
        :param w:
        :param type:
        """
        self.type = user_cfg['type']
        self.planner = planner
        self.samples = samples
        self.scalarization = user_cfg['scalarization']
        self.sol = self.sample_weight()
        self.w = self.sol['w']
        print('user sol, w=', self.sol['w'], 'f=', self.sol['f'])

    def sample_weight(self):

        # if self.scalarization == 'linear':
        #     samples = copy.deepcopy(self.samples['linear'])
        # elif self.scalarization == 'chebyshev':
        #     samples = copy.deepcopy(self.samples['chebyshev'])
        samples = copy.deepcopy(self.samples[self.scalarization ])
        # print('RAW user samples', len(samples))
        # samples = reduce_samples(samples)
        # print('reduced user samples', len(samples))
        # raise
        sol = np.random.choice(samples)
        # idx = random.randint(2,4)
        print('user sample set', len(samples),[{'w': s['w'], 'f': s['f']} for s in samples])
        # sol = samples[3]
        return sol



    def compute_error(self,sol):
        return self.planner.compute_pair_regret(sol, self.sol)


    def choice_feedback(self, sol_A, sol_B):
        """

        :param sol_A:
        :param sol_B:
        :return:
        """
        cost_A = self.planner.get_cost_of_traj(sol_A, self.w)
        cost_B = self.planner.get_cost_of_traj(sol_B, self.w)
        print("user feedback with mode: ", self.planner.scalarization_mode)
        if cost_A <= cost_B:
            return sol_A, sol_B
        else:
            return sol_B, sol_A