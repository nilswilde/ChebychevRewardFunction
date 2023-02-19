import copy

import numpy as np
import math as m
import dubins


def get_distance(pos1, pos2):
    return m.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def compute_dubins(turning_radius):
    """
    find a Dubin's path between any two fixed trajectories
    :param turning_radius: a given minimal turning radius
    :return: a trajectory, i.e., list of triplets (x,y,theta), and the features for that trajectory
    """
    q0 = (0, 0, 0)
    q1 = (0, 1, -m.pi / 2)
    # q1 = (2, 6, 0)
    # turning_radius = 1.0
    step_size = 0.01

    path = dubins.shortest_path(q0, q1, turning_radius)
    traj, _ = path.sample_many(step_size)
    return traj, get_features(traj, path, turning_radius)


def get_features(traj, path, radius):
    """
    custom designed features to evaluate a trajectory
    :param traj:
    :return:
    """
    L1 = path.segment_length(0)
    L3 = path.segment_length(2)
    L = path.path_length()
    IS_curvature = (L1 + L3) + (1 / radius) ** 2
    max_curvature = 1 / radius
    straight_length = L - L1 - L3
    return [L, IS_curvature / L]

    # time, theta_change, thetaprim_change  = [], [], []
    # out_of_box = 0
    # for idx in range(len(traj)-1):
    #     if traj[idx][1]<-1:
    #         out_of_box =1
    #     time += [get_distance(traj[idx], traj[idx+1])]
    #     theta_change += [abs(traj[idx][2]- traj[idx+1][2])*get_distance(traj[idx], traj[idx+1])]
    #     if idx<len(traj)-2:
    #         thetaprim_change+= [(abs(traj[idx][2] - traj[idx + 1][2])- abs(traj[idx+1][2] - traj[idx + 2][2])) ** 2]
    # # return [2**(2**(sum(time))), 100*max(theta_change)**(2)]
    # return [sum(time), sum(theta_change)]


def generate_trajectories():
    radia = np.arange(0, 100) * .01 + .1
    trajects = []
    for r_idx in range(len(radia)):
        r = radia[r_idx]
        # print('radius', r)
        states, phi = compute_dubins(r)
        trajects.append({'states': states, 'phi': phi})
    # trajects = normalize_features(trajects)
    trajects.reverse()
    return trajects


# def get_cost_of_traj(traj, w):
#     return round(np.dot(w, traj['f']),5)
#     return round(np.dot(w, traj['f']),5)


def get_opt_trajectory(w, trajects):
    min_cost = float('inf')
    best_traject = trajects[0]
    for traj in trajects:
        cost = get_cost_of_traj(traj, w)
        if cost < min_cost:
            min_cost = cost
            best_traject = traj
    return best_traject


def get_basis(trajects):
    traj_1 = get_opt_trajectory([1, 0], trajects)
    traj_2 = get_opt_trajectory([0, 1], trajects)
    return [{'w': [1, 0], 'phi': traj_1['phi'], 'states': traj_1['states']},
            {'w': [0, 1], 'phi': traj_2['phi'], 'states': traj_2['states']}]


def get_point_of_equal_cost(f_1, f_2, trajects):
    delta_0 = f_1[0] - f_2[0]
    delta_1 = f_1[1] - f_2[1]
    w_1 = - delta_1 / (delta_0 - delta_1)
    w_2 = 1 - w_1
    w = [w_1, w_2]
    # print('robust weight', w, np.dot(w, f_1), np.dot(w,f_2), f_1, f_2)
    traj = get_opt_trajectory(w, trajects)
    return {'w': w, 'phi': traj['phi'], 'states': traj['states']}


def bin_search_robust_weight(trajects):
    basis = get_basis(trajects)
    bf_1, bf_2 = basis[0]['phi'], basis[1]['phi']
    bw_1, bw_2 = basis[0]['w'], basis[1]['w']
    u1, u2 = np.dot(bf_1, bw_1), np.dot(bf_2, bw_2)
    neighbourhood = copy.deepcopy(basis)
    history = []
    curr_regrets = (-1, -1)
    for iter in range(10):
        f_1, f_2 = neighbourhood[0]['phi'], neighbourhood[1]['phi']
        w_1, w_2 = neighbourhood[0]['w'], neighbourhood[1]['w']
        w_new = get_point_of_equal_cost(f_1, f_2, trajects)
        history.append(w_new)
        f_new = w_new['phi']
        uN1, uN2 = np.dot(f_1, w_1), np.dot(f_2, w_2)
        reg_1, reg_2 = np.dot(f_new, bw_1) - u1, np.dot(f_new, bw_2) - u2  # regret of new sample under the basis
        # reg_1, reg_2 = np.dot(f_new, w_1) - uN1, np.dot(f_new, w_2) - uN2 # regret of new sample under the basis
        print('curr regrets at endpoints', reg_1, reg_2)
        if reg_1 == reg_2 or curr_regrets == (reg_1, reg_2):
            print('opt found', iter)
            return w_new, history
        if reg_1 <= reg_2:
            neighbourhood[0] = w_new
        else:
            neighbourhood[1] = w_new
            # neighbourhood[0] = rururu.chru(tigerohtiger)
        curr_regrets = (reg_1, reg_2)
    print('final solution', reg_1, reg_2)
    return w_new, history





def normalize_features(trajects):
    print("NORMALiZE")
    times = [traj['f'][0] for traj in trajects]
    jerks = [traj['f'][1] for traj in trajects]

    for idx in range(len(trajects)):
        trajects[idx]['f'][0] = (trajects[idx]['f'][0]-min(times))/(max(times)-min(times))
        trajects[idx]['f'][1] = (trajects[idx]['f'][1]-min(jerks))/(max(jerks)-min(jerks))
    # for idx in range(len(trajects)):
    #     trajects[idx]['f'][0] = (trajects[idx]['f'][0])/min(times)
    #     trajects[idx]['f'][1] = (trajects[idx]['f'][1]) /min(jerks)

    return trajects





def compute_optimal_trajectories(weights, sampled_trajects):
    opt_costs, opt_trajs = [], []
    for w in weights:
        opt_traj = get_opt_trajectory(w, sampled_trajects)
        opt_cost = get_cost_of_traj(opt_traj, w)
        opt_trajs.append(opt_traj)
        opt_costs.append(opt_cost)
        # print('w',w, trajects.index(opt_traj), opt_cost, opt_traj['phi'])
    return opt_costs, opt_trajs
