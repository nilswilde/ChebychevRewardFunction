from config import CFG
from DubinsPlanner.Dubins_plots import *
from DubinsPlanner.DubinsPlanner import *
from motion_planning_problem.StateLattice import StateLattice
# from driver_planner import DriverPlanner
from algorithm import *
from evaluation import compute_metrics
from motion_planning_problem.StateLattice import StateLattice
from main import get_planner_class
from Lattice_Planner.graph import Graph
# def get_planner_class(planner_type, scalarization):
#     if planner_type == 'Dubins2D':
#         return Dubins2DPlanner(scalarization)
#     elif planner_type == 'Dubins2DObst':
#         return Dubins2DPlannerObstacle(scalarization)
#     elif planner_type == 'Dubins3D':
#         return Dubins3DPlanner(scalarization)

def presample(planner, K=20):
    samples = {}
    for scalarization in ['linear', 'chebyshev']:
    # for scalarization in ['linear']:
        planner.scalarization_mode = scalarization

        samples[scalarization] = compute_k_grid_samples(planner, K)
        planner.sampled_solutions = normalize_features(planner, planner.sampled_solutions)

    return samples


if __name__ == '__main__':

    n = 11
    random.seed(n)
    np.random.seed(n)

    print("Run experiment for planner: ", CFG['planner_type'])
    K = 50
    planner = get_planner_class(CFG['planner_type'], 'linear')

    samples = presample(planner, K)
    # print('samples linear', [{'w': s['w'], 'f': s['f']} for s in samples['linear']])
    # print('samples cheb', [{'w': s['w'], 'f': s['f']} for s in samples['chebyshev']])
    # metric = compute_metrics(planner, samples, K)
    if CFG['show_plots']:
        # for scalarization in ['linear', 'chebyshev']:
        for scalarization in samples.keys():
            illustrate_2d(planner, samples[scalarization], label=scalarization+'', block=False)
        plt.show()
