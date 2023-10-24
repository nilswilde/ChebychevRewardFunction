import numpy as np

from main import get_planner_class
from algorithm import *
import time, os, errno, csv
import pandas as pd
from os import listdir
from os.path import isfile, join
from config import CFG
from Planner import load_planner
from algorithm import *
import matplotlib.pyplot as plt
from DubinsPlanner.Dubins_plots import illustrate_2d

label_dict = {
    'linear': '$\mathtt{SUM}$',
    'chebyshev': '$\mathtt{MAX}$',
    'AWS': '$\mathtt{AWS}$',
    'AWSR': '$\mathtt{AWS-R}$',
}
def presample(K=20, load=False):
    all_samples = {'linear':[], 'chebyshev':[], 'AWS':[], 'AWSR':[]}
    all_samples = {'linear':[], 'chebyshev':[]}
    planners = {'linear':[], 'chebyshev':[], 'AWS':[], 'AWSR':[]}
    planners = {'linear':[], 'chebyshev':[]}

    # folder = 'presamples/' + CFG['planner_type'] + '/driver204/'
    folder = 'presamples/' + CFG['planner_type'] + '/'
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    print(folder)
    print(files)
    for f in files:
        # planner = load_planner(CFG['planner_type']+'_'+scalarization, folder)
        scalarization = f[0:-7].split('_')[1]
        print(f, scalarization)
        planner = load_planner(f[0:-7], folder)
        planners[scalarization] += [planner]
        samples = planner.sampled_solutions
        all_samples[scalarization] += [samples]

    return all_samples, planners

# def reduce_samples(samples):
#     sample_dict = {}
#     for s in samples:
#         f = tuple(s['f'])
#         if f in sample_dict.keys():
#             sample_dict[f] += [s]
#         else:
#             sample_dict[f] = [s]
#
#     new_samples = []
#     for f in sample_dict.keys():
#         s = random.choice(sample_dict[f])
#         new_samples.append(s)
#     print("reduced sample set", len(samples), len(new_samples))
#     return new_samples
def filter_dominated_samples(samples, other_samples):
    new_samples = []
    for s in samples:
        dominated = False
        f = np.array(s['f'])
        for s_other in other_samples:
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

def compute_bounds(planner, all_samples):
    print(planner)
    dim = planner.dim
    bounds = {'lb': [float('inf')] * dim, 'ub': [-float('inf')] * dim}
    # samples = planner.get_basis()

    print('comp bounds')
    for i in range(dim):
        for label in all_samples.keys():
            for samples in all_samples[label]:
                for s in samples:
                    for j in range(dim):
                        bounds['lb'][j] = min(s['f'][j], bounds['lb'][j])
                        bounds['ub'][j] = max(s['f'][j], bounds['ub'][j])

    # print(bounds)
    return bounds
def normalize(samples, bounds):

    normalized_samples = []
    for s in samples:
        s_new = copy.deepcopy(s)
        # print(s)
        # s_new['f'] = np.subtract(s['f'], bounds['lb'])
        s_new['f'] = list(np.divide(s['f'], bounds['ub']))
        s_new['f'] = list(np.divide(np.subtract(s['f'], bounds['lb']), np.subtract(bounds['ub'], bounds['lb'])))
        # print(bounds)
        # print(s['f'], s_new['f'])
        # print(s_new)
        if np.max(s_new['f']) <1000:
            normalized_samples += [s_new]
    return normalized_samples

def compute_dispersion(samples):

    disp_values = []
    print('compute dispersion')
    dispersion_balls = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            mid_point = list(np.mean([samples[i]['f'], samples[j]['f']],axis=0))
            dist = float('inf')
            for k in range(len(samples)):
                dist = min(dist, np.linalg.norm(np.subtract(samples[k]['f'], mid_point)))
            if not point_dominated(mid_point, samples):  # and not point_is_dominating(mid_point):
                dispersion_balls.append({'pos': mid_point, 'r': dist})
    dispersion_points = []
    for point in dispersion_balls:
        if not point_dominated(point['pos'], samples):
            dispersion_points.append(point)
        disp_values = [elem['r'] for elem in dispersion_points]
    return np.max(disp_values), np.mean(disp_values), np.var(disp_values)

def count_unique_samples(samples):
    print('count # sols')
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
    print('compute measures', len(samples))
    num_samples = 1000
    np.random.seed(11)
    dim = len(samples[0]['w'])
    covered_points, dominating_points, dominated_points, pareto_points = [], [], [], []
    for _ in range(num_samples):
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
            'unique': round(unique_sols)
            }



def run_analysis():
    num_presamples = 100
    all_samples, planners = presample(num_presamples, load=True)
    planner = planners['linear'][0]
    bounds = compute_bounds(planner, all_samples)
    # bounds = {'lb': [0.11569187888147026, 1.2030898252905395, 0.3678443502826312, 0.5950556395672522], 'ub': [2.4762400346018962, 1.2030898252905395, 95.41997573799412, 0.6125273765255264]}
    for label in all_samples.keys():
        all_measures = []
        for samples in all_samples[label]:
            # samples = all_samples[label]
            # print('BOUNDS', bounds)
            # bounds = {'lb': [6.082982314858931e-06, 0.005789875515139416, 0.000697144390402471, 0.5950556395672522], 'ub': [2.2841193792889287, 1.1720989826669637, 25.43291390706057, 0.9134597589417143]}
            # bounds = {'lb': [2.6018253384774148e-05, 0.011984775030540008, 5.86309134575913e-07, 0.5950556395672522], 'ub': [1.355465431613033, 1.005530181392903, 62.85755395274889, 0.6785829824958303]}
            print('BOUNDS',bounds)
            samples = normalize(samples, bounds)
            # print(label, '# raw samples', len(samples))
            non_dominated_samples = filter_dominated_samples(samples, samples)
            # print(label, '# non_dominated samples', len(non_dominated_samples), [s['w'] for s in non_dominated_samples])
            # measures = compute_measures(non_dominated_samples, bounds)
            # all_measures += [measures]
            planners[label][0].plot_trajects_and_features(non_dominated_samples, title='', block=False)
            # planners[label][0].plot_trajects_and_features(samples, title=label+'_full', block=False)
        disp = round(np.mean([elem['disp'][0] for elem in all_measures]),2)
        cov = round(np.mean([elem['coverage'] for elem in all_measures]),2)
        var = round(np.mean([elem['unique'] for elem in all_measures]),3)
        # print('&', label_dict[label], '&', measures['disp'][0],'&', measures['coverage'], '&', measures['var'],"\\")
        print('&', label_dict[label], '&', disp,'&', cov, '&', var,"\\")
    planner = planners['linear']
    # planner.plot_linear_convexification(samples['linear'])
    print('samples', all_samples.keys())
    # for scalarization in all_samples.keys():
    #     illustrate_2d(planners[scalarization][0], all_samples[scalarization], label=scalarization, block=False)
    # illustrate_2d(planner, None, label='', block=False)
    plt.show()
    # plt.show()





if __name__ == '__main__':
    run_analysis()