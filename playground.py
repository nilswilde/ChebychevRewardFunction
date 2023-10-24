from config import CFG
from DubinsPlanner.Dubins_plots import *
from DubinsPlanner.DubinsPlanner import *
from DubinsPlanner.DubinsAdvanced import DubinsAdvanced
from motion_planning_problem.StateLattice import StateLattice
# from driver_planner import DriverPlanner
from algorithm import *
from evaluation import compute_metrics
from motion_planning_problem.StateLattice import StateLattice
from main import get_planner_class
from baseline import adapted_weight_sampling
from Lattice_Planner.graph import Graph


def presample(planner_original, K=20):
    print("Run Presampling")
    samples = {}
    for scalarization in ['linear','chebyshev']:
    # for scalarization in ['chebyshev']:
        planner = copy.deepcopy(planner_original)
        n = 1
        random.seed(n)
        np.random.seed(n)
        planner.scalarization_mode = scalarization
        samples[scalarization] = compute_k_grid_samples(planner, K)
        planner.sampled_solutions = samples[scalarization]
        planner.save_object(tag=scalarization)
    # print("RUNNING AWS")
    # planner.scalarization_mode = 'linear'
    # samples['AWS'] = adapted_weight_sampling(copy.deepcopy(planner), K,grid_mode=True)
    # planner.sampled_solutions = samples['AWS']
    # planner.save_object(tag='AWS')
    # samples['AWSR'] = adapted_weight_sampling(copy.deepcopy(planner), K, grid_mode=True, random_selection=True)
    # planner.sampled_solutions = samples['AWSR']
    # planner.save_object(tag='AWSR')
    return samples


if __name__ == '__main__':

    n = 7
    random.seed(n)
    np.random.seed(n)
    for _ in range(1):
        print("Run experiment for planner: ", CFG['planner_type'])
        K = 200
        planner = get_planner_class(CFG['planner_type'], 'linear')

        samples = presample(planner, K)
        metric = compute_metrics(planner, samples, K, save=True)
        if CFG['show_plots']:
            # planner.plot_linear_convexification(samples['linear'])
            for scalarization in samples.keys():
                illustrate_2d(planner, samples[scalarization], label=scalarization, block=False)
            plt.show()




