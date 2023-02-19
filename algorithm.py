import copy, random

import numpy as np
import math as m
from scipy.optimize import linprog
from scipy.optimize import minimize
from evaluation import normalize_features

def compute_best_k_samples(planner, K):
    """
    Our main problem: For a given planning problem, we want to find the best k samples
    :param planner:
    :param K:
    :return:
    """
    basis = planner.get_basis()
    set_neighbourhoods = [copy.deepcopy(basis)]
    samples = copy.deepcopy(basis)
    for k in range(K):
        print("Find robust samples, iter", k+1, '/', K)
        max_regret, max_neighbourhood, new_sample = 0, None, None
        for neighbourhood in set_neighbourhoods:
            # traj_robust, _ = bin_search_robust_weight(planner, neighbourhood)
            w_robust, lambdas = planner.get_neighbourhood_max_regret_weight(neighbourhood)
            traj_robust = planner.find_optimum(w_robust)
            regret, _ = planner.get_neighbourhood_regret(neighbourhood, traj_robust)
            if regret > max_regret and traj_robust not in samples:
                max_regret = regret
                max_neighbourhood = neighbourhood
                new_sample = traj_robust

        if max_regret == 0: # Abort since no new regret
            break
        set_neighbourhoods = update_neighbourhoods(set_neighbourhoods, max_neighbourhood, new_sample, lambdas)
        # print('adding sample', new_sample['w'])
        samples.append(new_sample)
        # print('curr minmax regret', planner.compute_minmax_regret(samples))
    # samples += planner.get_basis()
    samples = sorted(samples, key=lambda d: d['w'][0])

    return samples


def compute_best_k_samples_heuristic(planner, K):
    """
    Our main problem: For a given planning problem, we want to find the best k samples
    :param planner:
    :param K:
    :return:
    """

    print("RUN ALGORITHM")
    basis = planner.get_basis()
    set_neighbourhoods = [copy.deepcopy(basis)]
    samples = copy.deepcopy(basis)
    for k in range(K):
        print("\nFind robust samples, iter", k+1, '/', K)
        max_regret, max_neighbourhood, w_best= 0, None, None
        for neighbourhood in set_neighbourhoods:
            w_robust, lambdas = planner.get_neighbourhood_max_regret_weight(neighbourhood)
            if w_robust is None: # no regret in the neighbourhood, do not place a sample here
                continue
            regret_upper_bound = planner.get_neighbourhood_regret_upper_bound(neighbourhood, w_robust, lambdas)
            if regret_upper_bound > max_regret:
                max_regret = regret_upper_bound
                max_neighbourhood = neighbourhood
                w_best = w_robust
        if max_regret == 0: # even the best sample had no regret with respect, search exhausted
            break
        new_sample = planner.find_optimum(w_best)
        set_neighbourhoods = update_neighbourhoods(set_neighbourhoods, max_neighbourhood, new_sample, lambdas)
        samples.append(new_sample)
    # print('final samples')
    # for traj in samples:
    #     print('w=',traj['w'], 'f=',traj['f'])
    return samples


def update_neighbourhoods(set_neighbourhoods, split_neighbourhood, new_sample, lambdas):
    """

    :param set_neighbourhoods:
    :param split_neighbourhood:
    :param new_sample:
    :return:
    """
    set_neighbourhoods.remove(split_neighbourhood)
    new_neighbourhoods = []
    for idx in range(len(split_neighbourhood)):
        old_sample = split_neighbourhood[idx]
        if lambdas[idx] == 0:
            continue
        new_hood = copy.deepcopy(split_neighbourhood)
        new_hood.remove(old_sample)
        new_hood.append(new_sample)
        new_neighbourhoods.append(new_hood)
    return set_neighbourhoods + new_neighbourhoods


def get_point_of_equal_cost(neighbourhood):
    """

    :param neighbourhood:
    :return:
    """

    def compute_linear_combination(set_of_weights, other_weight):
        # compute the lambdas such that 'other_weight' is a linear combination of the set_of_weights
        import sys
        if np.linalg.cond(set_of_weights) < 1 / sys.float_info.epsilon:
            return np.linalg.solve(np.transpose(set_of_weights), other_weight)
        else:
            inverse = np.linalg.pinv(set_of_weights)
            scalars = np.dot(inverse, other_weight)
            return scalars
    # this only works in the 2D case
    f_1, f_2 = neighbourhood[0]['f'], neighbourhood[1]['f']
    delta_0 = f_1[0] - f_2[0]
    delta_1 = f_1[1] - f_2[1]
    if delta_0 == delta_1 == 0:
        w = neighbourhood[0]['w']
        return w, compute_linear_combination([traj['w']for traj in neighbourhood], w)
    w_1 = - delta_1 / (delta_0 - delta_1)
    w_2 = 1 - w_1
    w = [w_1, w_2]
    lambdas = compute_linear_combination([traj['w']for traj in neighbourhood], w)
    return w, lambdas


def max_reg_in_neighbourhood_linprog(neighbourhood, planner):
    """

    :param neighbourhood:
    :param planner:
    :return:
    """
    dim = len(neighbourhood[0]['w'])
    # order of decision variables: K, w (vector), lambdas (vector)
    # First, inequality constraints
    A_ub, b_ub = [], []
    for traj in neighbourhood:
        row = [-1] + list(traj['f']) + [0]*dim
        A_ub.append(row)
        b_ub.append(0)
    A_ub = np.multiply(-1, A_ub)  # invert as python takes constraints as <=

    # Now equality constraints
    A_eq = [[0] + [1]*dim + [0]*dim, [0] + [0]*dim + [1]*dim]  # all weights and lambdas equal to sum
    b_eq = [1,1]
    for i in range(dim):  # the max regret weight is a linear combination of the neighbourhood vertices
        v = [0]*dim
        v[i] = -1
        row = [0] + v + [traj['w'][i] for traj in neighbourhood]
        A_eq.append(row)
        b_eq.append(0)
    # define objective function
    self_costs = [planner.get_cost_of_traj(traj, traj['w']) for traj in neighbourhood]
    c = [-1] + [0]*dim + self_costs
    # define bounds
    bounds = [(0, None)] + [(0, 1)]*dim + [(0, 1)]*dim
    # solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method='revised simplex')
    w = res.x[1:dim+1]
    lambdas = res.x[dim+1:2*dim+1]
    return list(w), list(lambdas)

def compute_k_grid_samples(planner, K):
    weights = []
    if planner.dim == 2:
        for k in range(K):
            w_1_val = (k+1)/(K+1)
            weights.append([w_1_val, 1-w_1_val])
    else:
        for k in range(K):
            w = np.random.random(planner.dim)
            w = np.divide(w, sum(w))
            weights.append(list(w))
    samples = []
    for w in weights:
        traj = planner.find_optimum(w)
        samples.append(traj)
    samples += planner.get_basis()
    samples = sorted(samples, key=lambda d: d['w'][0])
    return samples



def compute_robust_solution(planner, samples):
    """
    Computing a robust weight using samples
    :param planner:
    :param samples:
    :return:
    """
    best_sol = None
    min_max_regret = float('inf')
    for sol_1 in samples:
        max_reg = 0
        for sol_2 in samples:
            r_abs, _ = planner.compute_pair_regret(sol_1, sol_2)
            max_reg = max(max_reg, r_abs)
            # max_reg += r_abs
        if max_reg < min_max_regret:
            min_max_regret = max_reg
            best_sol = copy.deepcopy(sol_1)
    return best_sol




