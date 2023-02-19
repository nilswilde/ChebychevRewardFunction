import copy

import numpy as np
from user import *
from learning_algorithm import *
from main import get_planner_class
from algorithm import *
import time, os, errno, csv
import pandas as pd
from config import CFG



def learn_user(user, learner, max_iter=10):
    """

    :param user:
    :param learner:
    :param max_iter:
    :return:
    """

    metrics = [compute_current_error(learner, user, 0, estimator='curr'),
               compute_current_error(learner, user, 0, estimator='exp')]
    for iter in range(max_iter):
        sol_A, sol_B = learner.get_query()
        # print("present query", sol_A['w'], sol_B['w'])
        prefered, rejected = user.choice_feedback(sol_A, sol_B)
        learner.receive_feedback(prefered, rejected)
        metrics += [compute_current_error(learner, user, iter+1, estimator='curr'),
                    compute_current_error(learner, user, iter+1, estimator='exp')]

    return metrics



# def run_trial():
#     """
#
#     :return:
#     """
#     identifier = int(time.time() * 100)
#     metrics_data = []
#     planner = get_planner_class(planner_type = 'Dubins4D')
#     # user_samples = {'GreedyHeuristic': compute_best_k_samples_heuristic(planner, 5),
#     #                 'Uniform': compute_k_grid_samples(planner, 5)}
#
#     for _ in range(3):
#         # for K in [20, 50, 100]:
#         for K in [20]:
#         # K = 20
#             learning_samples = {'GreedyHeuristic': compute_best_k_samples_heuristic(planner, K),
#                         'Uniform': compute_k_grid_samples(planner, K)}
#
#             user_modes = ['random','Uniform', 'GreedyHeuristic']
#             user_modes = ['Uniform', 'GreedyHeuristic']
#             for mode in user_modes:
#                 user_cfg = {'type':'deterministic', 'weight_mode':mode}
#                 user = User(planner, samples=learning_samples, user_cfg=user_cfg)
#                 for sample_type in ['Uniform', 'GreedyHeuristic']:
#                     for query_method in ['randomPosterior', 'maxregret']:
#                         learner_cfg = {'sample_type':sample_type, 'query_mode':query_method, 'K':K}
#                         learner = Learner(planner, learning_samples[sample_type], learner_cfg)
#                         metrics = learn_user(copy.deepcopy(user), learner)
#                         metrics_data += metrics
#                 save_metrics(metrics_data, identifier)

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

def presample(K=20):
    samples = {}
    planners = {}
    for scalarization in ['linear', 'chebyshev']:
        planner = get_planner_class(CFG['planner_type'],
                                    scalarization)  # generate planner first to then be used for different K

        samples[scalarization] = compute_k_grid_samples(planner, K)
        planners[scalarization] = planner
        # planner.sampled_solutions = normalize_features(planner, planner.sampled_solutions)
        samples[scalarization] = reduce_samples(samples[scalarization])
    return samples, planners

def run_trial():
    """
    :return:
    """
    num_presamples = 30
    identifier = int(time.time() * 100)
    metrics_data = []
    for _ in range(1): # repeat with different goal locations
        K = 10 # number of learning iterations

        print("Run experiment for planner: ", CFG['planner_type'])

        samples, planners = presample(num_presamples)
        print('lin samples', [(s['w'],s['f']) for s in samples['linear']])
        print('che samples', [(s['w'],s['f']) for s in samples['chebyshev']])
        user_samples, user_planners = presample(num_presamples)
        for _ in range(1):
            user_modes = ['chebyshev','linear']
            for user_mode in user_modes:
                user_cfg = {'type':'deterministic', 'scalarization':user_mode}
                user = User(user_planners[user_mode], samples=user_samples, user_cfg=user_cfg)
                learner_modes = ['chebyshev','linear']
                for learner_mode in learner_modes:
                    for query_method in ['randomPosterior', 'random', 'maxregret']:
                    # for query_method in ['maxregret']:
                        print('\n\nRun Trial: User', user_mode, 'Learner', learner_mode, 'query mode', query_method )
                        learner_cfg = {'sample_type':learner_mode, 'query_mode':query_method, 'K':K}
                        learner = Learner(copy.deepcopy(planners[learner_mode]), copy.deepcopy(samples[learner_mode]), learner_cfg)
                        metrics = learn_user(copy.deepcopy(user), learner, K)
                        metrics_data += metrics
                        # for elem in metrics:
                        #     print(elem)
                        print('wuser', user.sol['w'], user.sol['f'])
                        print('west', learner.current_sol['w'], learner.current_sol['f'])
                save_metrics(metrics_data, identifier)

def compute_current_error(learner, user, iter, estimator='curr'):
    """
    
    :param learner:
    :param user:
    :param iter:
    :return:
    """
    if estimator == 'curr':
        curr_sol = learner.current_sol
    else:
        curr_sol = learner.current_exp
    wlearn = curr_sol['w']

    abs_regret, rel_regret = user.compute_error(curr_sol)

    wuser, wlearn = np.divide(user.w,np.linalg.norm(user.w)), np.divide(wlearn,np.linalg.norm(wlearn))
    alignment = np.dot(wuser, wlearn)
    metrics = {'iter':iter,
               'learner_scalarization':learner.scalarization,
               'K':learner.K,
               'query':learner.query_mode,
               'label':learner.query_mode,
               'estimator':estimator,
               'regret':abs_regret,
               'relative regret': rel_regret,
               'Relative Error': rel_regret,
               'Alignment Error': alignment,
               'user_scalarization':user.scalarization}
    return metrics


def save_metrics(metrics, identifier):
    print("Shave")
    metrics_df = pd.DataFrame(metrics)
    folder = "simulation_data/"
    filename = 'LEARNING_ID:' + str(identifier) + '.csv'
    try:
        os.makedirs(folder+'/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    metrics_df.to_csv(folder + filename)


if __name__ == '__main__':
    n = random.randint(0,1000)
    n = 17
    np.random.seed(n)
    random.seed(n)
    print('random seed', n)
    for _ in range(5):
        run_trial()