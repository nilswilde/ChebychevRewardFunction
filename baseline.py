import copy, random
import time

import numpy as np
from algorithm import compute_k_grid_samples, update_neighbourhoods


def compute_size_of_neighbourhood(neighbourhood):
    """

    :param neighbourhood:
    :return:
    """
    mean_f = list(np.mean([s['f'] for s in neighbourhood], axis=0))
    min_dist = float('inf')
    max_dist = -float('inf')
    for s in neighbourhood:
        dist = np.linalg.norm(np.subtract(mean_f, s['f']))
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
    # print('size of hood',[s['w'] for s in neighbourhood],[s['w'] for s in neighbourhood], min_dist)
    return min_dist, mean_f

def find_guided_optimum(planner, target_feature, dim, utopia, nadir, grid_mode=True):
    if grid_mode:
        print('find best guided solution', target_feature, utopia, nadir, 'grid mode?', grid_mode)
        # ground_set = compute_k_grid_samples(copy.deepcopy(planner), K=1000)
        ground_set = copy.deepcopy(planner.generate_trajectories())
        ground_set = normalize_samples(ground_set,dim, utopia, nadir)

        w = list(-np.subtract(target_feature, [1] * planner.dim))
        w = [1]*planner.dim
        # print('ground set', [s['f'] for s in ground_set])
        new_sample, best_alginment = None, float('inf')
        norm_nadir = [1]*dim
        for s in ground_set:
            alignment = np.dot(np.subtract(target_feature, norm_nadir), np.subtract(s['f'], norm_nadir)) \
                        / np.dot(np.linalg.norm(np.subtract(target_feature, norm_nadir)), np.linalg.norm(np.subtract(s['f'], norm_nadir)))
            alignment2 = np.linalg.norm(alignment-1)
            if alignment2 <= .001:
                # f_norm = normalize_samples([s],dim, utopia, nadir)[0]['f']
                cost = np.dot(w, s['f'])
                if cost < best_alginment:
                    best_alginment = cost
                    new_sample = s
                    new_sample['w']=w
        print('best sample', new_sample['w'], new_sample['f'])
        return new_sample
    else:
        print('Find guided optimum - no grid')
        sol = planner.find_optimum_constrained(target_feature, utopia, nadir)
        sol_normalized = normalize_samples([sol], dim, utopia, nadir)[0]
        return sol_normalized

def normalize_samples(samples, dim, utopia, nadir):
    for j in range(len(samples)):
        # print(samples[j]['f'], nadir,utopia)
        f_j = copy.deepcopy(samples[j]['f'])
        for i in range(dim):
            if nadir[i] != utopia[i]:
                f_j[i] = (f_j[i] - utopia[i]) / (nadir[i]-utopia[i])
            else:
                f_j[i] = (f_j[i] - utopia[i])
        samples[j]['f'] = f_j
    return samples

def un_normalize_samples(samples, dim, utopia, nadir):
    new_samples = copy.deepcopy(samples)
    for j in range(len(new_samples)):
        f_j = copy.deepcopy(new_samples[j]['f'])
        for i in range(dim):
            f_j[i] = f_j[i] * (nadir[i]-utopia[i]) + utopia[i]
        new_samples[j]['f'] = f_j
    return new_samples

def adapted_weight_sampling(planner, K, grid_mode=False, random_selection=False):
    """
    Our main problem: For a given planning problem, we want to find the best k samples
    :param planner:
    :param K:
    :return:
    """

    # np.random.seed(10)
    print("RUN AWS, K=", K)
    dim = planner.dim
    basis = planner.get_basis()

    utopia = [basis[i]['f'][i] for i in range(dim)]
    nadir = [0]*dim
    for i in range(dim):
        print('basis sol', basis[i]['w'], basis[i]['f'])
        nadir[i] = max([s['f'][i] for s in basis])
    print('utopia', utopia, 'nadir', nadir)

    samples = copy.deepcopy(basis)
    samples = normalize_samples(samples, dim, utopia, nadir)
    set_neighbourhoods = [copy.deepcopy(samples)]
    # print('initial samples')
    # for s in samples:
    #     print(s['w'], s['f'])
    for k in range(K):
        print("\nAdaptive samples, iter", k, '/', K, len(samples) + 1)
        sizes = []
        max_nhood_size, max_neighbourhood, target_feature = 0, None, None
        for neighbourhood in set_neighbourhoods:
            size, mean_f = compute_size_of_neighbourhood(neighbourhood)
            if random_selection:
                size *= random.random()
            sizes += [size]
            if size > max_nhood_size:
                max_nhood_size = size
                max_neighbourhood = neighbourhood
                target_feature = mean_f
        print('search in hood',[(s['w']) for s in max_neighbourhood],[(s['f']) for s in max_neighbourhood],max_nhood_size, sizes)
        new_sample = find_guided_optimum(planner, target_feature, dim, utopia, nadir, grid_mode=grid_mode)
        # print('expected features', target_feature)
        print('new sample', new_sample['w'], new_sample['f'],' from hood',[(s['w'], s['f']) for s in max_neighbourhood])
        print(un_normalize_samples([new_sample], dim, utopia, nadir))
        samples += [copy.deepcopy(new_sample)]


        set_neighbourhoods = update_neighbourhoods(set_neighbourhoods, max_neighbourhood, new_sample, [1]*dim)
    # return samples

    unormalized_samples = un_normalize_samples(samples, dim, utopia, nadir)

    print('AWS samples')
    for traj in unormalized_samples:
        print('w=', traj['w'], 'f=', traj['f'])
    return unormalized_samples

