from Planner import Planner
from Lattice_Planner.graph import Graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class GraphPlanner(Planner):
    def __init__(self, scalarization):
        super().__init__(2, scalarization, 'Graph' + str(2) + 'D')

        self.g = Graph()
        self.s = self.g.get_closest_vertex((0, 1100))[0]
        self.t = self.g.get_closest_vertex((1100, 450))[0]
        # self.g.plot()

    def find_optimum(self, w, LUT=None):
        """

        :param w:
        :return:
        """

        print('find optimum', w, self.scalarization_mode)
        self.g.set_edge_costs(w)

        path = self.g.compute_shortest_path(self.s, self.t, scalarization_mode=self.scalarization_mode)

        # path1 = self.g.compute_shortest_path(self.s, self.t, scalarization_mode='linear')
        # path2 = self.g.compute_shortest_path(self.s, self.t, scalarization_mode='chebyshev')
        # print('WEIGHT', w)
        # print("path1", path1)
        # print("path2", path2)
        # self.g.plot(paths=[path1, path2])

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
        step_size = .5
        weights = [[round(w, m), round(1 - w, m)] for w in np.arange(0, 1 + step_size, step_size)]
        print("generating", len(weights), 'evaluation samples')
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights)
        return self.sampled_solutions

    def plot_trajects_and_features(self, trajects, x_axis_vals, highlight, x_label='idx', title=''):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5))
        # all_sols = self.generate_evaluation_samples()
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        cols = ['b', 'r', 'g', 'purple', 'yellow', 'teal', 'pink', 'grey', 'black']
        cmap = cm.get_cmap('jet')
        ax = axes[0]
        # ax.imshow(self.g.map_img)
        fig, ax = self.g.plot(fig, ax, block=False)
        ax.set_title('' + title, fontsize=18)
        # plot start and goal
        ax.plot(self.s[0], self.s[1], 'D', 'green', zorder=10000)
        ax.plot(self.t[0], self.t[1], 'X', 'purple', zorder=10000)

        for i in range(len(trajects)):
            traj = trajects[i]
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color=cmap(i/len(trajects)),linewidth=5)

        ax.set_xlabel('x pos', fontsize=16)
        ax.set_ylabel('y pos', fontsize=16)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax = axes[1]

        # phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
        # ax.plot(phi_1, phi_2, 'x', color='grey', label='all trajects')

        phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
        for idx in range(len(phi_1)):
            ax.plot(phi_1[idx], phi_2[idx], 'D', color=cmap(idx/len(trajects)), label='optimal trajectories')

        # if highlight is not None:
        #     ax.plot(highlight['f'][0], highlight['f'][1], 'D', color='green', label='optimal trajectories',
        #             markersize=6)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)
        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

        fig.tight_layout()

