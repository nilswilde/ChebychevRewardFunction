import copy
import random

import numpy as np
from scipy.optimize import linprog
from heapq import *


class Learner():

    def __init__(self, planner, samples, learner_cfg):
        self.planner = planner
        self.dim = planner.dim
        self.samples = samples
        print('samples learner', [{'w': s['w'], 'f': s['f']} for s in samples])
        self.scalarization = learner_cfg['sample_type']
        self.query_mode=learner_cfg['query_mode']
        self.K = learner_cfg['K']
        self.current_sol = self.get_initial_solution()
        self.current_exp= self.get_initial_solution()
        self.feedback = []

        self.chebyshev_feasible_sets = None
        print('init learner', learner_cfg, 'initial sol', self.current_sol['w'],self.current_sol['f'])

    def receive_feedback(self, preferred, rejected):
        self.feedback.append((copy.deepcopy(preferred), copy.deepcopy(rejected)))
        if self.scalarization == 'chebyshev':
            self.chebyshev_feasible_sets = self.brute_force_chebyshev_tree()
            print('\n\nreceive feedback, new cheb f sets', len(self.chebyshev_feasible_sets))

        self.current_sol = copy.deepcopy(preferred)
        self.current_exp = self.compute_expectation()
        # print('current best', self.current_sol['w'])


    def compute_expectation(self):
        samples = [self.current_sol['w']]
        for _ in range(1000):
            w = np.random.random(self.dim)
            w = list(np.divide(w, sum(w)))
            if self.weight_feasible(w):
                samples.append(w)
        w_exp = list(np.mean(samples,axis=0))
        return self.planner.find_optimum(w_exp)


    def get_initial_solution(self):
        w = [1/self.dim]*self.dim
        sol = self.planner.find_optimum(w)
        # return self.samples[-1]
        return sol

    def get_query(self):
        def random_query():
            sol_A = self.current_sol
            sol_B = np.random.choice(self.samples)
            return sol_A, sol_B

        def random_posterior_query():
            sol_A = self.current_sol
            samples = copy.deepcopy(self.samples)
            np.random.shuffle(samples)
            sol_B = sol_A
            for sol in samples:
                if sol['f'] == sol_A['f']:
                    continue
                if self.weight_feasible(sol['w']):
                    sol_B = sol
                    print('found feasible alternative', self.weight_feasible(sol_A['w']))
                    break
            if sol_A['w'] == sol_B['w']:
                print('No new solution found')
            idx_A, idx_B = "A", "B"
            if sol_B == sol_A:
                idx_B = idx_A
            if sol_A in self.samples:
                idx_A = self.samples.index(sol_A)
            if sol_B in self.samples:
                idx_B = self.samples.index(sol_B)
            print('random query:', idx_A, idx_B)#, sol_A['w'], sol_A['f'], sol_B['w'], sol_B['f'])
            return copy.deepcopy(sol_A), copy.deepcopy(sol_B)

        def random_posterior_query2():
            sol_A = self.current_sol
            samples = copy.deepcopy(self.samples)
            np.random.shuffle(samples)
            for sol in samples:
                if self.weight_feasible(sol['w']):
                    sol_A = copy.deepcopy(sol)
                    break
            sol_B = copy.deepcopy(sol_A)
            samples = copy.deepcopy(self.samples)
            for sol in samples:
                if sol['w'] == sol_A:
                    continue
                if self.weight_feasible(sol['w']):
                    sol_B = copy.deepcopy(sol)
                    print('found feasible alternative', self.weight_feasible(sol_A['w']))
                    break
            if sol_A['w'] == sol_B['w']:
                print('No feasible solution found')
            return copy.deepcopy(sol_A), copy.deepcopy(sol_B)

        def max_regret_query():
            print("comp regret", self.scalarization, self.planner.scalarization_mode)
            sol_A = self.current_sol
            sol_B = sol_A
            max_regret = -float('inf')
            samples = copy.deepcopy(self.samples)
            np.random.shuffle(samples)
            for sol in samples:
                if sol['f'] == sol_A['f']:
                    continue
                if self.weight_feasible(sol['w']):

                    regret, regret_rel = self.planner.compute_pair_regret(sol, sol_A)
                    if regret >= max_regret:
                        max_regret = regret
                        sol_B = sol
            idx_A, idx_B = "A", "B"
            if sol_B == sol_A:
                idx_B = idx_A
            if sol_A in self.samples:
                idx_A = self.samples.index(sol_A)
            if sol_B in self.samples:
                idx_B = self.samples.index(sol_B)
            print('max regret query:', idx_A, idx_B, 'regret', max_regret)
            return copy.deepcopy(sol_A), copy.deepcopy(sol_B)

        # def max_regret_query2():
        #     sol_A = self.current_sol
        #     sol_B = sol_A
        #     max_regret = -float('inf')
        #     for tmp_A in self.samples:
        #         for tmp_B in self.samples:
        #             if self.weight_feasible(tmp_A['w']) and self.weight_feasible(tmp_B['w']):
        #                 regret, regret_rel = self.planner.compute_pair_regret(tmp_A, tmp_B)
        #                 if regret > max_regret:
        #                     max_regret = regret
        #                     sol_A = copy.deepcopy(tmp_A)
        #                     sol_B = copy.deepcopy(tmp_B)
        #     idx_A, idx_B = "A", "B"
        #     if sol_B == sol_A:
        #         idx_B=idx_A
        #     if sol_A in self.samples:
        #         idx_A = self.samples.index(sol_A)
        #     if sol_B in self.samples:
        #         idx_B = self.samples.index(sol_B)
        #     print('max regret query:', idx_A,idx_B, 'regret', max_regret)
        #     return copy.deepcopy(sol_A), copy.deepcopy(sol_B)

        if self.query_mode == 'random':
            return random_query()
        if self.query_mode == 'randomPosterior':
            return random_posterior_query()
        if self.query_mode == 'maxregret':
            return max_regret_query()

    def weight_feasible(self, w):
        if self.scalarization == 'linear':
            for (pref, rej) in self.feedback:
                f_1, f_2 = pref['f'], rej['f']
                if np.dot(np.subtract(f_1,f_2), w) >= 0:
                    return False
            return True
        elif self.scalarization == 'chebyshev':
            if len(self.feedback) == 0:
                return True
            return self.decide_chebyshev_feasibiliy(w)


    def decide_chebyshev_feasibiliy(self, w):
        if len(self.feedback) == 0:
            return True
        for f_set in self.chebyshev_feasible_sets:
            if f_set is None:
                print("none f_set!")
                raise
                return True
            rhs = np.dot(f_set, w)
            if np.all(rhs <= 0):
                return True
        return False



    def brute_force_chebyshev_tree(self):

        def feasible_set_non_empty(A_tmp):
            if A_tmp is None:
                return False
            res = linprog([1]*self.dim, A_ub=A_tmp, b_ub=[0]*len(A_tmp), bounds=[0, 1])
            if res.success:
                return True
            else:
                return False

        # print("find chebyshev feasible set")
        open_list = []
        feasible_sets = []
        for j in range(self.dim):
            indicator = [j]
            A = self.construct_chebyshev_feasible_set(indicator)
            if feasible_set_non_empty(A):
                open_list += [{'i':indicator, 'F_set': A, 'value':len(indicator)}]
        while len(open_list) > 0:
            curr = open_list.pop(-1)

            # print("curr", curr['value'], curr['i'])
            # print("open list", [(elem['value'], elem['i']) for elem in open_list])
            if len(curr['i']) < len(self.feedback):
                for j in range(self.dim):
                    new_indicator = curr['i'] + [j]
                    A_new = self.construct_chebyshev_feasible_set(new_indicator)
                    if feasible_set_non_empty(A_new):
                        open_list += [{'i':new_indicator, 'F_set': A_new, 'value':len(new_indicator)}]

            else:
                print("chebyshev feasible set found", curr['i'], curr['F_set'])
                # return [curr['F_set']]
                feasible_sets += [copy.deepcopy(curr['F_set'])]

        print(len(feasible_sets), " chebby feedback set worked")
        return feasible_sets

    # def brute_force_chebyshev_tree(self):
    #
    #     indices = []
    #     open_list = [[j] for j in range(self.dim)]
    #     while len(open_list) > 0:
    #         curr = open_list.pop(0)
    #         if len(curr) < len(self.feedback):
    #             for j in range(self.dim):
    #                 new_elem = curr + [j]
    #                 open_list.append(new_elem)
    #         else:
    #             indices.append(curr)
    #     # print('possible j_max index vectors',indices)
    #
    #     for set_idcs in indices:
    #         A = self.construct_chebyshev_feasible_set(set_idcs)
    #
    #         if A is None:
    #             continue
    #         res = linprog([1]*self.dim, A_ub=A, b_ub=[-.00001]*len(A), bounds=[0, 1])
    #         if res.success:
    #             print('Sampled a feasible set!')
    #             # print('feedback idcs', set_idcs)
    #             # print('feasible set\n', np.array(A))
    #             return A
    #     print("no feedback set worked")
    #     return None


    def construct_chebyshev_feasible_set(self, indices):
        # print('construct chebyshev set', indices)
        feasible_set = []
        for feedback_idx in range(len(indices)):
            A_feedback = []
            (pref, rej) = self.feedback[feedback_idx]
            f_1, f_2 = pref['f'], rej['f']
            j = indices[feedback_idx]
            for i in range(self.dim):
                # rhs at j > lhs at i
                a_ji = [0]*self.dim
                a_ji[i] += f_1[i]
                a_ji[j] -= f_2[j]
                A_feedback.append(a_ji)
                # rhs at j > rhs at i
                if i == j:
                    continue
                a_ji = [0] * self.dim
                a_ji[i] += f_2[i]
                a_ji[j] -= f_2[j]
                A_feedback.append(a_ji)

            # check if the individual feedback divides the preferred and the rejected weights
            if np.all(np.dot(A_feedback, pref['w'])<=0) and np.any(np.dot(A_feedback, rej['w'])>=0):
                feasible_set += A_feedback
            else:
                return None
        return feasible_set

    # def construct_allchebyshev_feasible_set(self):
    #     print('construct chebyshev set')
    #     feasibility_tree = []
    #     for (pref, rej) in self.feedback:
    #         f_1, f_2 = pref['f'], rej['f']
    #         feasible_sets = {}  # collect all potential feasible sets that could be derived from pref < rej feedback
    #         for j in range(self.dim):  # loop over every possible RHS index
    #             A_j = []  # this describes the feasible set of j is the maximizer of the RHS for the feedback
    #             for i in range(self.dim):
    #                 a_ji = [0]*self.dim
    #                 a_ji[i] = f_1[i]
    #                 a_ji[j] = -f_2[j]
    #                 A_j.append(a_ji)
    #             feasible_sets[j] = A_j
    #         feasibility_tree.append(feasible_sets)
    #     return feasibility_tree





