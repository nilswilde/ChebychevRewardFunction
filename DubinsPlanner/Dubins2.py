import random
import matplotlib.pyplot as plt
from Planner import *
import math as m
import numpy as np
import dubins
from matplotlib import cm
import seaborn as sns
def get_distance(x_1, x_2):
    return m.sqrt((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2)

def in_collision(traj, obstacles=[]):
    # return False
    for pos in traj:
        for o in obstacles:
            if get_distance(pos, o['pos']) < o['r']:
                return True
        if not 0<=pos[0]<=3.5 or not 0<= pos[1]<=3.5:
            return True
    return False

def filter_dominated_samples(samples):
    new_samples = []
    for s in samples:
        dominated = False
        for s_other in samples:
            f = np.array(s['f'])

            f_other = np.array(s_other['f'])
            if np.any(f_other<f) and np.all(f_other<= f):
                dominated = True
                break
        if not dominated:
            new_samples += [s]
    new_samples.sort(key=lambda tup: tup['f'][0])
    return new_samples



def closeness_to_obstacles(traj, obstacles):

    closeness_measure = 0
    if in_collision(traj,obstacles):
        return None
    for pos in traj:
        min_dist = float('inf')
        for obst in obstacles:
            dist = get_distance(pos, obst['pos'])-obst['r']
            if dist < .0:
                return None
            min_dist = min(min_dist, dist)
            if min_dist < 0.00:
                return None
        closeness = np.exp(-min_dist)
        # closeness = 1/min_dist
        # closeness = 3-min_dist
        closeness_measure = max(closeness_measure, closeness)
        # closeness_measure += closeness / len(traj)
    # return 5-min_dist
    return closeness_measure# -2.4


def generate_random_goal():
    while True:
        x = random.randint(0, 6) - 1
        y = random.randint(0, 6) - 1
        theta = random.randint(1, 8) * m.pi / 4
        if x == y == 0 or y == theta == 0:
            continue
        return (x, y, theta)


class DubinsAdvanced(Planner):
    def __init__(self, scalarization):
        super().__init__(2, scalarization, 'Dubins2DAdvanced')
        self.generated_stuff = None
        self.min_radius = .2
        self.max_radius = 1.0
        self.label = 'Dubins2DAdvanced'

        # self.goal = (3, 2.2, -m.pi / 2)
        self.goal = (3, 2, -m.pi / 4)
        # self.goal = (3, 2, 0)
        print("2D Dubings, goal", self.goal)

        self.obstacles = [
            {
                # 'pos': (1.5, 1), 'r': .4, },
                'pos': (1.8, 2.3), 'r': .2, },
            {'pos': (1.8, 1.3), 'r': .2, },
        ]
        self.generate_trajectories()


    def compute_dubins(self, via_point, radius):
        """
        find a Dubin's path between any two fixed trajectories
        :param turning_radius: a given minimal turning radius
        :return: a trajectory, i.e., list of triplets (x,y,theta), and the features for that trajectory
        """
        q0 = (0.3, 0.3, 0)
        q1 = via_point
        q2 = self.goal
        # radius = .4
        path1 = dubins.shortest_path(q0, q1, radius)
        path2 = dubins.shortest_path(q1, q2, radius)
        step_size = 0.01
        traj1, _ = path1.sample_many(step_size)
        traj2, _ = path2.sample_many(step_size)
        traj = traj1 + traj2
        return traj, self.get_features(traj)

    def generate_trajectories(self, force_new=False):
        '''
        Pre-generate a large set of Dubins' paths for different turn radia
        :return:
        '''
        if self.generated_stuff is not None and not force_new:
            return self.generated_stuff
        print('generate base set')
        num_basic_samples = 20#int((self.max_radius - self.min_radius) * 200) + 1
        # radia = np.linspace(self.min_radius, self.max_radius, num_basic_samples)
        x_pos_list = np.linspace(1, 2, 10)
        y_pos_list = np.linspace(0, 3, 10)
        theta_list = np.linspace(0, np.pi/2, 8+1)
        radia = np.linspace(0.3, 1, 8)
        trajects = []
        for y in y_pos_list:
            for x in x_pos_list:
                for theta in theta_list:
                    for radius in radia:
                        states, f = self.compute_dubins((x, y, theta), radius)
                        if not in_collision(states, self.obstacles):
                            trajects.append({'f': f, 'states': states})

        trajects.reverse()
        self.generated_stuff = trajects
        return trajects

    def find_optimum(self, w, sample_mode=False):
        """

        :param w:
        :return:
        """
        min_cost = float('inf')
        best_traject = None
        radius_sampled_trajects = self.generate_trajectories()
        for traj in radius_sampled_trajects:
            cost = self.get_cost_of_traj(traj, w)
            if cost < min_cost:
                min_cost = cost
                best_traject = traj
        return {'w': w,
                'f': best_traject['f'],
                'states': best_traject['states']
                }

    def get_features(self, traj):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L = len(traj)*.01
        obst_distances = closeness_to_obstacles(traj, self.obstacles)
        if obst_distances is None:
            return None
        features = [L, 10*obst_distances]
        return features

    def compute_minmax_regret(self, samples):
        """

        :param samples:
        :return:
        """
        if len(self.sampled_solutions) == 0:
            self.generate_evaluation_samples()
        max_regret_abs, max_regret_rel = 0, 0
        int_regret_abs, int_regret_rel = 0, 0
        for i in range(len(self.sampled_solutions)):
            abs_regrets_at_w, rel_regrets_at_w = [], []
            for traj in samples:
                r_abs, r_rel = self.compute_pair_regret(traj, self.sampled_solutions[i])
                abs_regrets_at_w.append(r_abs)
                rel_regrets_at_w.append(r_rel)
            if min(abs_regrets_at_w) > max_regret_abs:
                max_regret_abs = min(abs_regrets_at_w)
            if min(rel_regrets_at_w) > max_regret_rel:
                max_regret_rel = min(rel_regrets_at_w)
            int_regret_abs += min(abs_regrets_at_w)
            int_regret_rel += min(rel_regrets_at_w)

        return {
            'max_regret': max_regret_abs,
            'max_relative_regret': max_regret_rel,
            'total_regret': int_regret_abs,
            'total_relative_regret': int_regret_rel
        }

    def generate_evaluation_samples(self):
        """

        :return:
        """
        m = 3
        step_size = 1 / 10 ** m
        weights = [[round(w, m), round(1 - w, m)] for w in np.arange(0, 1 + step_size, step_size)]
        print("generating", len(weights), 'evaluation samples')
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights, sample_mode=True)


    def plot_trajects_and_features(self, samples, title='', block=False):

        def compute_matching_to_pareto_front(samples, pareto_samples):
            matching = {}
            for i in range(len(samples)):
                for j in range(len(pareto_samples)):
                    if samples[i]['f'] == pareto_samples[j]['f']:
                        matching[i] = j
                        break
            return matching

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        all_sols = self.generate_trajectories()#[0::10]

        pareto_optimal_solutions = filter_dominated_samples(all_sols)
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        for ax in axes:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')

        pal = plt.cm.viridis(np.linspace(0, 1, len(pareto_optimal_solutions)))
        ax = axes[0]
        # ax.set_title('' + title, fontsize=18)
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
            # ax.add_patch(Rectangle((o['x_0'], o['y_0']), o['x_1']-o['x_0'], o['y_1']-o['y_0']))
        if samples is None:
            for traj in all_sols:
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='lightgrey')
            for i in range(len(pareto_optimal_solutions)):
                traj = pareto_optimal_solutions[i]
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=2, color=pal[i])
        else:
            matching = compute_matching_to_pareto_front(samples, pareto_optimal_solutions)
            for i in range(len(samples)):
                traj = samples[i]
                if i in matching.keys():
                    col = pal[matching[i]]
                else:
                    print('dominated solution', traj['f'])
                    col = 'black'
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=2, color=col)
        # plot bounds
        ax.plot([-.1,3.6,3.6,-.1,-.1], [-.1,-.1,3.6,3.6,-.1], '--', color='darkgrey')
        ax.set_xlim([-.1,3.7])
        ax.set_ylim([-.1,3.7])
        ax.set_aspect('equal')

        # plot Pareto front
        ax = axes[1]
        if samples is None:
            # plot ground set of trajectories
            all_sols_plot= all_sols[0::10]
            phi_1, phi_2 = [traj['f'][0] for traj in all_sols_plot], [traj['f'][1] for traj in all_sols_plot]
            ax.plot(phi_1, phi_2, '.', color='lightgrey', label='all trajects')

            for i in range(len(pareto_optimal_solutions)):
                s = pareto_optimal_solutions[i]
            # phi_1, phi_2 = [traj['f'][0] for traj in pareto_optimal_solutions], [traj['f'][1] for traj in pareto_optimal_solutions]
                ax.plot(s['f'][0], s['f'][1], 'D', color=pal[i], label='all trajects')
        else:
            # plot samples
            phi_1, phi_2 = [traj['f'][0] for traj in samples], [traj['f'][1] for traj in samples]
            for i in range(len(phi_1)):
                if i in matching.keys():
                    col = pal[matching[i]]
                else:
                    col = 'black'
                ax.plot(phi_1[i], phi_2[i], 'D', color=col, label='optimal trajectories')
        ax.set_xlim([3.2, 6.1])
        ax.set_ylim([5.3, 10])
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

        fig.tight_layout()
        if block:
            plt.show()

    def plot_linear_convexification(self, samples, title='', block=False):

        def compute_matching(lin_samples, base_samples):
            print('computed lin cost matching', len(base_samples))
            matching={}
            for i in range(len(base_samples)):
                s_base = base_samples[i]
                best_cost, best_j = 1000000, s_base
                for j in range(len(lin_samples)):
                    s_lin = lin_samples[j]
                    regret = np.dot(s_lin['w'], s_base['f']) - np.dot(s_lin['w'], s_lin['f'])

                    if regret < best_cost:
                        best_cost = regret
                        best_j = j
                if best_cost < 1000000:
                    matching[i] = best_j
            print('MATCHING',matching)
            return matching


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        all_sols = self.generate_trajectories()
        samples = samples#[0::5]
        all_sols = filter_dominated_samples(all_sols)#[0::5]
        matching = compute_matching(samples, all_sols)
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        # for ax in axes:
        #     ax.set_yticks([])
        #     ax.set_xticks([])
            # ax.axis('off')

        ax = axes[0]
        ax.set_title('' + title, fontsize=18)
        pal = plt.cm.hsv(np.linspace(0, 1, len(samples)))
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
            # ax.add_patch(Rectangle((o['x_0'], o['y_0']), o['x_1']-o['x_0'], o['y_1']-o['y_0']))

        for  i in range(len(all_sols)):
            traj = all_sols[i]
            col = 'lightgrey'
            col = pal[matching[i]]
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color=col,alpha=.15)
        # for traj in pareto_optimal_solutions[0::4]:
        #     ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='darkseagreen')

        for i in range(len(samples)):
            traj = samples[i]
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=4, color=pal[i])

        # ax.set_xlabel('x pos', fontsize=12)
        # ax.set_ylabel('y pos', fontsize=12)
        # axes[0].set_title('Sampled Trajectories'+title)
        ax.set_aspect('equal')

        # plot Pareto front
        ax = axes[1]

        # plot ground set of trajectories
        phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
        for i in range(len(phi_1)):
            col = pal[matching[i]]
            ax.plot(phi_1[i], phi_2[i], 'D', color='lightgrey', alpha=.8)
            lin_phi = samples[matching[i]]['f']
            ax.plot([lin_phi[0],phi_1[i]], [lin_phi[1],phi_2[i]], '--', color=col, alpha=.3)



        # plot samples
        phi_1, phi_2 = [traj['f'][0] for traj in samples], [traj['f'][1] for traj in samples]
        for idx in range(len(phi_1)):
            ax.plot(phi_1[idx], phi_2[idx], 'D', color=pal[idx%len(pal)], label='optimal trajectories')

        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

        fig.tight_layout()
        if block:
            plt.show()










