import numpy as np

from main import get_planner_class
from algorithm import *
import time, os, errno, csv
import pandas as pd
from config import CFG
from Planner import load_planner
from algorithm import *
import matplotlib.pyplot as plt

label_dict = {'linear': '$\mathtt{SUM}$', 'chebyshev': '$\mathtt{MAX}$'}
def presample(K=20, load=False):
    all_samples = {}
    planners = {}
    if not load:
        for scalarization in ['linear', 'chebyshev']:
            planner = get_planner_class(CFG['planner_type'],
                                        scalarization)  # generate planner first to then be used for different K

            samples = compute_k_grid_samples(planner, K)
            planners[scalarization] = planner

            # samples = normalize(planner, samples)
            samples = reduce_samples(samples)
            all_samples[scalarization] = samples
            planner.sampled_solutions = samples

    else:
        for scalarization in ['linear', 'chebyshev']:
        # for scalarization in ['chebyshev']:
            planner = load_planner(CFG['planner_type']+'_'+scalarization, 'presamples/')
            planners[scalarization] = planner
            samples = planner.sampled_solutions
            all_samples[scalarization] = samples
            # planner.plot_trajects_and_features(samples)
            from DubinsPlanner.Dubins_plots import illustrate_2d



    return all_samples, planners

def reduce_samples(samples):
    sample_dict = {}
    for s in samples:
        f = tuple(s['f'])
        if f in sample_dict.keys():
            sample_dict[f] += [s]
        else:
            sample_dict[f] = [s]

    new_samples = []
    for f in sample_dict.keys():
        s = random.choice(sample_dict[f])
        new_samples.append(s)
    print("reduced sample set", len(samples), len(new_samples))
    return new_samples
def filter_dominated_samples(samples, other_samples):
    new_samples = []
    for s in samples:
        dominated = False
        for s_other in other_samples:
            f = np.array(s['f'])
            f_other = np.array(s_other['f'])
            if np.any(f_other<f) and np.all(f_other <= f):
                dominated = True
                break
        if not dominated:
            new_samples += [s]
    return new_samples

def point_is_covered(point, samples):
    F = [s['f'] for s in samples]
    dominates_sample = np.all(F <= point, axis=1)
    return np.any(dominates_sample)

def point_dominated(pos, samples):

    dominated = False
    for s in samples:
        f_other = np.array(s['f'])
        # print(pos, f_other, np.any(f_other < pos),  np.all(f_other <= pos))
        if np.any(f_other < pos) and np.all(f_other <= pos):
            dominated = True
            break
    return dominated
def point_is_dominating(point, samples):
    F = [s['f'] for s in samples]
    dominates_sample = np.all(F >= point, axis=1)
    return np.any(dominates_sample)

def compute_bounds(samples):
    dim = len(samples[0]['w'])
    bounds = {'lb': [float('inf')] * dim, 'ub': [-float('inf')] * dim}
    for s in samples:
        for i in range(dim):
            bounds['lb'][i] = min(bounds['lb'][i], s['f'][i])
            bounds['ub'][i] = max(bounds['ub'][i], s['f'][i])
    print(bounds)
    return bounds
def normalize(samples, bounds):

    normalized_samples = []
    for s in samples:
        s_new = copy.deepcopy(s)
        # print(s)
        # s_new['f'] = np.subtract(s['f'], bounds['lb'])
        s_new['f'] = list(np.divide(s['f'], bounds['ub']))
        # print(s_new)
        if np.max(s_new['f']) <1000:
            normalized_samples += [s_new]
    return normalized_samples

def compute_dispersion(samples):
    dim = len(samples[0]['w'])
    disp_values = []
    if dim == 2:
        for i in range(len(samples)-1):
            s_i = samples[i]
            s_j = samples[i+1]
            disp = np.linalg.norm(np.subtract(s_i['f'], s_j['f'])) / 2
            disp_values+=[disp]
    else:
        dispersion_balls = []
        for i in range(len(samples)):
            # print(samples[i]['w'])
            for j in range(i + 1, len(samples)):
                # if max(samples[i]['f']) == 1 or max(samples[j]['f']) == 1:
                #     continue
                mid_point = list(np.mean([samples[i]['f'], samples[j]['f']],axis=0))
                dist = float('inf')
                for k in range(len(samples)):
                    dist = min(dist, np.linalg.norm(np.subtract(samples[k]['f'], mid_point)))
                if not point_dominated(mid_point, samples):  # and not point_is_dominating(mid_point):
                    dispersion_balls.append({'pos': mid_point, 'r': dist})
        dispersion_points = []
        for point in dispersion_balls:
            # for other_point in midpoints:
            #     if other_point['r'] != point['r']:
            #         other_pos = {'f':other_point['pos']}
            #         if point_dominated(point['pos'], [other_pos]):
            #             not_dominated = False
            #             break
            if not point_dominated(point['pos'], samples):
                dispersion_points.append(point)
        disp_values = [elem['r'] for elem in dispersion_points]
    return round(np.max(disp_values),2), round(np.mean(disp_values),3), round(np.var(disp_values),4)

def count_unique_samples(samples):
    count =0
    threshold = .01
    for i in range(len(samples)):
        min_dist = float('inf')
        for j in range(i+1, len(samples)):
            dist = np.linalg.norm(np.subtract(samples[i]['f'], samples[j]['f']))
            min_dist = min(min_dist, dist)
        if min_dist > threshold:
            count += 1
    return count


def compute_measures(samples, bounds):

    num_samples = 10000
    np.random.seed(11)
    dim = len(samples[0]['w'])
    covered_points, dominating_points, dominated_points, pareto_points = [], [], [], []
    for _ in range(num_samples):
        # pos = np.random.uniform(bounds['lb'],bounds['ub'],dim)
        pos = np.random.random(dim)
        if point_is_covered(pos, samples):
            covered_points.append(pos)
        if point_dominated(pos, samples):
            dominated_points.append(pos)
        elif point_is_dominating(pos, samples):
            dominating_points.append(pos)
        else:
            pareto_points.append(pos)

    disp_measures = compute_dispersion(samples)
    unique_sols = count_unique_samples(samples)
    print('#unique sols', unique_sols)
    # print('dispersion max, mean, var', disp_measures)
    # print('Dominated Volume', len(dominated_points) / num_samples)
    # print('covered Volume', len(covered_points) / num_samples)
    # print('Pareto points', len(pareto_points) / num_samples)
    F = [s['f'] for s in samples]
    # for f in F:
    #     print(f)
    # print('mean', np.mean(F, axis=0), 'variance', np.mean(np.var(F, axis=0)))
    # print('min', np.min(F,axis=0))
    # print('max', np.max(F,axis=0))

    return {'disp':disp_measures,
            'vol':round(len(dominated_points) / num_samples,2),
            'coverage':round(len(covered_points) / num_samples,2),
            'var': round(np.mean(np.var(F, axis=0)),3),
            'unique': unique_sols
            }



def run_analysis():
    num_presamples = 100
    all_samples, planners = presample(num_presamples, load=True)
    ground_set = all_samples['linear'] + all_samples['chebyshev']
    bounds = compute_bounds(ground_set)
    normalized_ground_set = normalize(ground_set, bounds)

    for label in all_samples.keys():
        samples = all_samples[label]

        samples = normalize(samples, bounds)
        # print(label, '# raw samples', len(samples))
        non_dominated_samples = filter_dominated_samples(samples, samples)
        # print(label, '# non_dominated samples', len(non_dominated_samples), [s['w'] for s in non_dominated_samples])


        measures = compute_measures(non_dominated_samples, bounds)
        print('&', label_dict[label], '&', measures['disp'][0],'&', measures['coverage'], '&', measures['unique'],"\\")
        # print('&', label_dict[label], '&', measures['disp'][0], '&', measures['disp'][2],'&', measures['vol'], "\\")

        planners[label].plot_trajects_and_features(non_dominated_samples, title='', block=False)
        planners[label].plot_trajects_and_features(samples, title='', block=False)

    plt.show()





if __name__ == '__main__':
    run_analysis()