from Planner import Planner
from Lattice_Planner.graph import Graph
import numpy as np
import matplotlib.pyplot as plt


class GraphPlanner(Planner):
    def __init__(self,  scalarization):
        super().__init__(2, scalarization, 'Graph' + str(2) + 'D')

        self.g = Graph()
        self.s = self.g.get_closest_vertex((300, 200))[0]
        self.t = self.g.get_closest_vertex((800, 400))[0]
        # self.g.plot()

    def find_optimum(self, w, LUT=None):
        """

        :param w:
        :return:
        """
        self.g.set_edge_costs(w, self.scalarization_mode)

        path = self.g.compute_shortest_path(self.s, self.t)

        path_pos = self.g.get_path_positions(path)

        f = self.g.compute_path_features(path)
        print('new path, w=', w, ', f= ', f, np.dot(w, f))
        return {'w': w,
                'f': f,
                'states': path_pos
                }


    def generate_evaluation_samples(self):
        """

        :return:
        """
        m = 1
        step_size = 1 / 10 ** m
        weights = [[round(w, m), round(1 - w, m)] for w in np.arange(0, 1 + step_size, step_size)]
        print("generating", len(weights), 'evaluation samples')
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights)
        return self.sampled_solutions

    def plot_trajects_and_features(self, trajects, x_axis_vals, highlight, x_label='idx', title=''):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
        all_sols = self.generate_evaluation_samples()
        print("plotting", len(self.sampled_solutions), 'sampled solutions')

        ax = axes[0]
        # ax.imshow(self.g.map_img)
        fig, ax = self.g.plot(fig, ax, block=False)
        ax.set_title('' + title, fontsize=18)
        # plot start and goal
        ax.plot(self.s[0], self.s[1], 'D', 'green', zorder=10000)
        ax.plot(self.t[0], self.t[1], 'X', 'purple', zorder=10000)
        for traj in all_sols:
            ax.plot([x[0]+np.random.random() for x in traj['states']], [x[1]+np.random.random()  for x in traj['states']], color='black')

        for traj in trajects:
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']])

        ax.set_xlabel('x pos', fontsize=16)
        ax.set_ylabel('y pos', fontsize=16)
        ax.set_aspect('equal')

        ax = axes[1]
        phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
        ax.plot(phi_1, phi_2, 'x', color='grey', label='all trajects')

        phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
        ax.plot(phi_1, phi_2, 'D', color='r', label='optimal trajectories')

        if highlight is not None:
            ax.plot(highlight['f'][0], highlight['f'][1], 'D', color='green', label='optimal trajectories',
                    markersize=6)
        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

