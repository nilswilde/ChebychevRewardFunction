import numpy as np
from algorithm import get_point_of_equal_cost, max_reg_in_neighbourhood_linprog#, compute_linear_combination
import copy, pickle
import os
from config import CFG

class Planner():
    '''
    Generic Planner Class
    '''
    def __init__(self, dim, scalarization, label='generic'):
        self.dim = dim # number of features
        self.label = label + '_' + scalarization
        self.scalarization_mode = scalarization
        self.value_bounds = [{'lb': 0, 'ub': 1} for _ in range(self.dim)]# min and max values for each feature
        self.basis = None # basic solutions, e.g., for the [1 0 0], [0 1 0], [0 0 1] vectors
        self.sampled_solutions = []  # high number of sampled solutions, only for evaluation

    def __repr__(self):
        return self.label

    def get_cost_of_traj(self, traj, w):
        if self.scalarization_mode == 'linear':
            return np.dot(w, traj['f'])
        if self.scalarization_mode == 'chebyshev':
            return np.max(np.multiply(w, traj['f']))


    def get_value_bounds(self):
        '''
        :return:
        '''
        value_bounds = [{'lb': float('inf'), 'ub': -float('inf')} for _ in range(self.dim)]
        for i in range(self.dim):
            w = [0.0] * self.dim
            w[i] = 1.0
            sol = self.find_optimum(w)
            value_bounds[i]['lb'] = sol['f'][i]
            for j in range(self.dim):
                value_bounds[j]['ub'] = max(sol['f'][j], value_bounds[j]['ub'])
        print('value bounds', value_bounds)
        return value_bounds

    def get_basis(self):
        '''

        :return:
        '''
        basis = []
        value_bounds = [{'lb': float('inf'), 'ub': -float('inf')} for _ in range(self.dim)]
        for i in range(self.dim):
            w = [0.0] * self.dim
            w[i] = 1.0
            sol = self.find_optimum(w)
            basis += [sol]
            value_bounds[i]['lb'] = [sol['f'][i]]
        self.basis = basis
        return basis

    def find_optimum(self, w, lookup_table=None):
        '''
        Solve the planning problem for a given weight w
        :param w:
        :return:
        '''
        return {'w':w,
                'f': [1]*self.dim,
                'states': [(i,i)for i in range(10)]}

    def find_optima_for_set_of_weights(self, weights, lookup_table=None):
        trajects, opt_costs = [], []
        for w in weights:
            traj = self.find_optimum(w, lookup_table)
            cost = self.get_cost_of_traj(traj, w)
            traj['u'] = cost
            trajects.append(traj)
            opt_costs.append(cost)
        return trajects

    def compute_pair_regret(self, traj_P, traj_Q):
        c_QQ = self.get_cost_of_traj(traj_Q, traj_Q['w'])
        c_PQ = self.get_cost_of_traj(traj_P, traj_Q['w'])
        return c_PQ-c_QQ, c_PQ/c_QQ


    def compute_regrets(self, weights):
        '''

        :param weights:
        :param trajects:
        :return:
        '''
        regrets = []
        for w_P in weights:
            traj_P = self.find_optimum(w_P)
            regrets_row = []
            for w_Q in weights:
                traj_Q = self.find_optimum(w_Q)
                regrets_row.append(self.compute_pair_regret(traj_P, traj_Q))
            regrets.append(regrets_row)
        return regrets

    def get_neighbourhood_regret(planner, neighbourhood, traj):
        """

        :param planner:
        :param neighbourhood:
        :param traj:
        :return:
        """
        min_regret, best_neighbour = float('inf'), None
        for traj_neigh in neighbourhood:
            reg,_ = planner.compute_pair_regret(traj_neigh, traj)
            if reg < min_regret:
                min_regret = reg
                best_neighbour = copy.deepcopy(traj_neigh)
        return min_regret, best_neighbour

    def get_neighbourhood_regret_upper_bound(self, neighbourhood, w_new, scalars):
        """

        :param planner:
        :param neighbourhood:
        :param traj:
        :return:
        """

        best_cost = float('inf')
        # scalars = compute_linear_combination([traj['w']for traj in neighbourhood], w_new)
        self_costs = []
        for traj_neigh in neighbourhood:
            upper_bound_cost = self.get_cost_of_traj(traj_neigh, w_new)
            # print('comp upper bound cost',upper_bound_cost, traj_neigh['f'], w_new)
            self_costs += [self.get_cost_of_traj(traj_neigh, traj_neigh['w'])]
            if upper_bound_cost < best_cost:
                best_cost = upper_bound_cost
        # print('regret bound', best_cost - np.dot(scalars, self_costs), best_cost,np.dot(scalars, self_costs), scalars, self_costs)
        regret_bound = best_cost - np.dot(scalars, self_costs)

        return regret_bound

    def get_neighbourhood_max_regret_weight(self, neighbourhood):
        """

        :param planner:
        :param neighbourhood:
        :return:
        """

        # w, lambdas = get_point_of_equal_cost(neighbourhood)  # implementation for 2 feature system only
        w, lambdas = max_reg_in_neighbourhood_linprog(neighbourhood, self)  # implementation for n features
        return w, lambdas

    def compute_minmax_regret(self, samples):
        """

        :param samples:
        :return:
        """
        if len(self.sampled_solutions) == 0:
            self.generate_evaluation_samples()
        max_regret_abs, max_regret_rel = -float('inf'), 0
        int_regret_abs, int_regret_rel = 0, 0
        max_absreg_weight, max_relreg_weight = None, None
        for i in range(len(self.sampled_solutions)):
            abs_regrets_at_w, rel_regrets_at_w = [], []  # save the regrets for each trajectory at test point i
            for traj in samples:
                r_abs, r_rel = self.compute_pair_regret(traj, self.sampled_solutions[i])
                abs_regrets_at_w.append(r_abs)
                rel_regrets_at_w.append(r_rel)
            if min(abs_regrets_at_w) > max_regret_abs:  # check if the smallest regret at i is a new overall maximum
                max_regret_abs = min(abs_regrets_at_w)
                max_absreg_weight = self.sampled_solutions[i]['w']
            if min(rel_regrets_at_w) > max_regret_rel:
                max_regret_rel = min(rel_regrets_at_w)
                max_relreg_weight = self.sampled_solutions[i]['w']
            int_regret_abs += min(abs_regrets_at_w)
            int_regret_rel += min(rel_regrets_at_w)
        print('max reg at', max_absreg_weight)
        return {
            'max_regret': round(max_regret_abs,5),
            'max_relative_regret': round(max_regret_rel,5),
            'total_regret': round(int_regret_abs,5),
            'total_relative_regret': round(int_regret_rel,5)
        }

    def save_samples(self, samples, tag='uniform'):
        path = 'rewardLearning/ctrl_samples'
        if not os.path.isdir(path):
            os.mkdir(path)

        w_set = [sol['w'] for sol in samples]
        feature_set = [sol['f'] for sol in samples]
        input_set = [sol['states'] for sol in samples]
        np.savez(path+'/' + self.label + '_' + tag+'.npz', feature_set=feature_set, input_set=input_set,
                 w_set=w_set)

    def save_object(self):
        with open(self.label+'.pickle', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


def load_planner(label):
    with open(label+'.pickle', 'rb') as inp:
        loaded_object = pickle.load(inp)
    return loaded_object
