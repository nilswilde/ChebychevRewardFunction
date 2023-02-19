from config import CFG
from DubinsPlanner.Dubins_plots import *
from DubinsPlanner.DubinsPlanner import *
from motion_planning_problem.StateLattice import StateLattice
# from driver_planner import DriverPlanner
from algorithm import *
from evaluation import compute_metrics
from motion_planning_problem.StateLattice import StateLattice

def get_planner_class(planner_type, scalarization):
    if planner_type == 'Dubins2D':
        return Dubins2DPlanner(scalarization)
    elif planner_type == 'Dubins2DObst':
        return Dubins2DPlannerObstacle(scalarization)
    elif planner_type == 'Dubins3DObst':
        return Dubins3DPlannerObstacle(scalarization)
    elif planner_type == 'Dubins3D':
        return Dubins3DPlanner(scalarization)



if __name__ == '__main__':
    print("Run experiment for planner: ", CFG['planner_type'])
    for _ in range(CFG['num_trials']):
        planner_orig = get_planner_class(CFG['planner_type']) # generate planner first to then be used for different K
        for K in CFG['K_values']:
            planner = copy.deepcopy(planner_orig)
            labels = ['Greedy', 'Uniform']
            samples = {'GreedyHeuristic': compute_best_k_samples_heuristic(planner, K),
                       'Uniform': compute_k_grid_samples(planner, K)}
            # planner.sampled_solutions = normalize_features(planner, planner.sampled_solutions)
            metric = compute_metrics(planner, samples, K)
            if CFG['show_plots']:
                illustrate_2d(planner, samples['GreedyHeuristic'], label='Greedy Regret Sampling', block=False)
                illustrate_2d(planner, samples['Uniform'], label='Uniform Sampling')
