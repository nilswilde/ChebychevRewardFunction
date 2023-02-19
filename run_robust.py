from config import CFG
from DubinsPlanner.Dubins_plots import *
from DubinsPlanner.DubinsPlanner import *
from motion_planning_problem.StateLattice import StateLattice
# from driver_planner import DriverPlanner
from mTSP.mTSPSolver import mTSPSolver
from algorithm import *
from evaluation import compute_metrics, evaluate_robust_solution
from motion_planning_problem.StateLattice import StateLattice

def get_planner_class(planner_type):
    if planner_type == 'Dubins2D':
        return Dubins2DPlanner()
    elif planner_type == 'Dubins3D':
        return Dubins3DPlanner()
    elif planner_type == 'Dubins4D':
        return Dubins4DPlanner()
    elif planner_type == 'Lattice':
        return StateLattice()
    elif 'mTSP' in planner_type:
        num_vertices = int(planner_type[4:])
        return mTSPSolver(num_vertices)


if __name__ == '__main__':
    print("Run experiment for planner: ", CFG['planner_type'])
    for _ in range(CFG['num_trials']):
        planner_orig = get_planner_class(CFG['planner_type']) # generate planner first to then be used for different K
        K = CFG['robust_samples']
        planner = copy.deepcopy(planner_orig)
        labels = ['GreedyHeuristic', 'Uniform']
        samples = {'GreedyHeuristic': compute_best_k_samples_heuristic(planner, K),
                   'Uniform': compute_k_grid_samples(planner, K),
                   'Expected':  [planner.find_optimum(w=[1/planner.dim]*planner.dim)]}
        rob_samples = {
            'GreedyHeuristic':compute_robust_solution(planner, samples['GreedyHeuristic']),
            'Uniform':compute_robust_solution(planner, samples['Uniform']),
            'Expected': compute_robust_solution(planner, samples['Expected'])
        }
        print('Rob Samples',
              '\nGreedy:', rob_samples['GreedyHeuristic'],
              '\nUni:', rob_samples['Uniform'],
              '\nExp:', rob_samples['Expected'],
              )

        # metric = compute_metrics(planner, samples, K)
        metric = evaluate_robust_solution(planner, samples, rob_samples, K)
        if CFG['show_plots']:
            illustrate_2d(planner, samples['GreedyHeuristic'], highlight=rob_samples['GreedyHeuristic'], label='Greedy Regret Sampling', block=False)
            illustrate_2d(planner, samples['GreedyHeuristic'], highlight=rob_samples['Expected'], label='Expected', block=False)
            illustrate_2d(planner, samples['Uniform'], highlight=rob_samples['Uniform'],label='Uniform Sampling')
