CFG = {
    # which experiment to run
    'planner_type': 'Dubins3D',
    'planner_type': 'Dubins2D',
    'planner_type': 'Dubins2DObst',
    # 'planner_type': 'Dubins3DObst',
    # 'planner_type': 'mTSP15',
    'planner_type': 'Graph',

    'K_values': [0, 1, 3, 5, 10],
    # 'K_values': [0, 3, 5, 10, 20],
    # 'K_values': [3],

    'num_trials': 30,  # number of repeats. In each repeat a new planner instance is generated (with random goals)
    'show_plots': True,  # show plots of the sampled trajectories and pareto fron at the end of each trial

    # problem specific settings
    'mTSP_numRobots': 10,

    'scalarization_mode': 'chebychev',
    # 'scalarization_mode': 'linear',

    # Robustness Experiment
    'robust_samples': 20,


}
