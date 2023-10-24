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
    return new_samples


def closeness_to_obstacles(traj, obstacles):

    closeness_measure = 0
    for pos in traj:
        min_dist = float('inf')
        for obst in obstacles:
            dist = get_distance(pos, obst['pos'])-obst['r']
            min_dist = min(min_dist, dist)
            if min_dist<0:
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


class DubinsPlanner(Planner):
    def __init__(self, num_feat, scalarization):
        super().__init__(num_feat, scalarization, 'Dubin' + str(num_feat) + 'D')
        self.generated_stuff = None
        self.obstacles = []

    def compute_dubins(self, turning_radius):
        """
        find a Dubin's path between any two fixed trajectories
        :param turning_radius: a given minimal turning radius
        :return: a trajectory, i.e., list of triplets (x,y,theta), and the features for that trajectory
        """
        q0 = (0, 0, 0)
        q1 = self.goal  # q1 = (2, 6, 0)
        # turning_radius = 1.0
        step_size = 0.01

        path = dubins.shortest_path(q0, q1, turning_radius)
        traj, _ = path.sample_many(step_size)

        return traj, self.get_features(path, turning_radius)

    def generate_trajectories(self, force_new=False):
        '''
        Pre-generate a large set of Dubins' paths for different turn radia
        :return:
        '''
        if self.generated_stuff is not None and not force_new:
            return self.generated_stuff

        num_basic_samples = int((self.max_radius - self.min_radius) * 200) + 1
        radia = np.linspace(self.min_radius, self.max_radius, num_basic_samples)
        trajects = []
        for r_idx in range(len(radia)):
            r = radia[r_idx]
            states, f = self.compute_dubins(r)
            if not in_collision(states, self.obstacles):
                trajects.append({ 'f': f, 'states': states})

        trajects.reverse()
        self.generated_stuff = trajects
        return trajects

    def find_optimum(self, w, sample_mode=False):
        """

        :param w:
        :return:
        """
        # if not sample_mode:
        #     return self.find_optimum_numerical(w)

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

    # def find_optimum_numerical(self,w):
    #     """
    #     A numerimal optimization approach to find the best Dubins trajectory for a given weight
    #     :param w:
    #     :return:
    #     """
    #     def cost(radius):
    #         states, f = self.compute_dubins(radius)
    #         if f is None:
    #             return 10000000000
    #         cost = self.get_cost_of_traj({'f':f}, w)
    #
    #         return cost
    #
    #     # def bounds(radius):
    #     #     return self.min_radius < radius < self.max_radius
    #
    #     from scipy.optimize import minimize
    #     from scipy.optimize import Bounds
    #     bounds = Bounds(self.min_radius, self.max_radius)
    #     bounds = [(self.min_radius, self.max_radius)]
    #     x_0 = (self.max_radius+self.min_radius) /2
    #     x_0 = self.max_radius
    #     # res = minimize(cost, x_0, method='trust-constr', bounds=bounds)
    #     res = minimize(cost, x_0, method='L-BFGS-B', bounds=bounds)
    #     # print(res)
    #     best_radius = res.x[0]
    #     states, f = self.compute_dubins(best_radius)
    #     print('numerical solution', w, f, best_radius)
    #     return {'w':w,
    #             'f':f,
    #             'states': states}

    def find_optimum_constrained(self, target_f, utopia, nadir):
        from scipy.optimize import minimize
        from scipy.optimize import Bounds
        from scipy.optimize import NonlinearConstraint

        def normalize(f, dim, utopia, nadir):
            f_norm = copy.deepcopy(f)
            for i in range(dim):
                f_norm[i] = (f[i] - utopia[i]) / (nadir[i] - utopia[i])
            return f_norm

        def cost(radius):
            radius = radius[0]
            if not self.min_radius <= radius <= self.max_radius:
                return 10000000
            _, f = self.compute_dubins(radius)
            if f is None:
                return 1000000000000
            f_norm = normalize(f, self.dim, utopia, nadir)
            cost = np.dot(w, f_norm)
            return cost

        def comp_alignment(f_normalized, f_target_normalized):
            alignment = np.dot(np.subtract(f_target_normalized, nadir_normalized),
                               np.subtract(f_normalized, nadir_normalized)) \
                        / np.dot(np.linalg.norm(np.subtract(f_target_normalized, nadir_normalized)),
                                 np.linalg.norm(np.subtract(f_normalized, nadir_normalized)))
            # alignment2 = np.linalg.norm(alignment - 1)
            return alignment

        def cons_f(radius):
            radius = radius[0]
            if not self.min_radius <= radius <= self.max_radius:
                return 1000000000000
            _, f = self.compute_dubins(radius)
            # print(f)
            if f is None:
                return 1000000000000
            f_norm = normalize(f, self.dim, utopia, nadir)
            return comp_alignment(f_norm, target_f)

        print('solve constrained search', target_f)

        bounds = Bounds(self.min_radius, self.max_radius)
        nonlinear_constraint = NonlinearConstraint(cons_f, 1, 1)
        nadir_normalized = [1] * self.dim
        w = -np.subtract(target_f, [1] * self.dim)
        x_0 = (self.max_radius + self.min_radius) / 2
        x_0 = self.min_radius
        res = minimize(cost, x_0, method='trust-constr', bounds=bounds, constraints=[nonlinear_constraint])
        print(res)
        best_radius = res.x[0]
        states, f = self.compute_dubins(best_radius)
        w = np.divide(w, np.sum(w))
        sol = {'w': list(w),
               'f': f,
               'states': states}
        # print('constrained op sol', sol, )
        f_final_norm = normalize(f, self.dim, utopia, nadir)
        # print(f_final_norm, comp_alignment(f_final_norm, target_f))

        return sol

    def plot_trajects_and_features(self, samples, title='', block=False):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        all_sols = self.generate_trajectories()

        pareto_optimal_solutions = filter_dominated_samples(all_sols)
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        # for ax in axes:
        #     ax.set_yticks([])
        #     ax.set_xticks([])
            # ax.axis('off')

        ax = axes[0]
        ax.set_title('' + title, fontsize=18)
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
            # ax.add_patch(Rectangle((o['x_0'], o['y_0']), o['x_1']-o['x_0'], o['y_1']-o['y_0']))
        if samples is None:
            for traj in all_sols[0::4]:
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='lightgrey')
            for traj in pareto_optimal_solutions[0::4]:
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='darkseagreen')
        else:
            pal = sns.color_palette("tab10")
            pal = plt.cm.hsv(np.linspace(0,1,len(samples)))

            for i in range(len(samples)):
                traj = samples[i]
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=3, color=pal[i])

        # ax.set_xlabel('x pos', fontsize=12)
        # ax.set_ylabel('y pos', fontsize=12)
        # axes[0].set_title('Sampled Trajectories'+title)
        ax.set_aspect('equal')

        # plot Pareto front
        ax = axes[1]
        if samples is None:
            # plot ground set of trajectories
            phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
            ax.plot(phi_1, phi_2, '.', color='lightgrey', label='all trajects')

            phi_1, phi_2 = [traj['f'][0] for traj in pareto_optimal_solutions], [traj['f'][1] for traj in pareto_optimal_solutions]
            ax.plot(phi_1, phi_2, 'D', color='darkseagreen', label='all trajects')
        else:
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
        all_sols = filter_dominated_samples(all_sols)[0::5]
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

class Dubins2DPlanner(DubinsPlanner):

    def __init__(self, scalarization):
        super().__init__(2, scalarization)
        self.goal = (3, 2, -m.pi / 2)
        # self.goal = (1, 2, m.pi)
        print("2D Dubings, goal", self.goal)
        self.min_radius = .2
        self.max_radius = 1.0

    def get_features(self, path, radius):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L1 = path.segment_length(0)
        L3 = path.segment_length(2)
        L = path.path_length()
        IS_curvature = (L1 + L3) + (1 / radius) ** 2
        max_curvature = 1 / radius
        straight_length = L - L1 - L3

        features = [L, max_curvature]
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
        max_absreg_weight, max_relreg_weight = None, None
        for i in range(len(self.sampled_solutions)):
            abs_regrets_at_w, rel_regrets_at_w = [], []
            for traj in samples:
                r_abs, r_rel = self.compute_pair_regret(traj, self.sampled_solutions[i])
                abs_regrets_at_w.append(r_abs)
                rel_regrets_at_w.append(r_rel)
            if min(abs_regrets_at_w) > max_regret_abs:
                max_regret_abs = min(abs_regrets_at_w)
                max_absreg_weight = self.sampled_solutions[i]['w']
            if min(rel_regrets_at_w) > max_regret_rel:
                max_regret_rel = min(rel_regrets_at_w)
                max_relreg_weight = self.sampled_solutions[i]['w']
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


class Dubins2DPlannerObstacle(Dubins2DPlanner):

    def __init__(self, num_feat=2):

        super().__init__(num_feat)
        self.label = 'Dubins2DObst'
        self.goal = (3, 2, -m.pi / 2)

        goals = [(3, 2.2, -m.pi / 2),(3, 2.2, -m.pi / 3)]
        self.goal = (0.4, 3, -m.pi )
        self.goal = (3, 2.2, -m.pi / 2)
        # self.goal = (2, 1.0, m.pi )

        self.obstacles = [
            {
                # 'pos': (0.7, .05), 'r': .2, },
                'pos': (1.8, 2.3), 'r': .2, },
                          {'pos': (1.8, 1.3), 'r': .2, },
                          # {'pos': (0.5, 2), 'r': .2, },
                          ]
        # self.generate_trajectories(force_new=True)
        # print("2D Dubings, goal", self.goal)

    def get_features(self, path, radius):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L1 = path.segment_length(0)
        L3 = path.segment_length(2)
        L = path.path_length()

        traj, _ = path.sample_many(0.1)

        obst_distances = []
        obst_distances = closeness_to_obstacles(traj, self.obstacles)
        if obst_distances is None:
            return None
        # features = [L, max(obst_distances)/L]
        max_dist = get_distance((0,0), (self.goal[0],self.goal[1]))
        features = [L, 10*obst_distances]
        # features = [L-4, max(obst_distances)**(1)*10-6] # hand made normalization, not great but meh
        # features = [L-3, max(obst_distances)**(1)*10-5]
        # print('f =', features)
        return features



class Dubins3DPlanner(DubinsPlanner):
    """
    Dubins Plannign problem with n=3 features
    """
    def __init__(self, scalarization):
        super().__init__(3, scalarization)
        self.goal = (2, 1, 3 * m.pi / 4)
        # self.goal = (1.5, 1.5, 0)
        # self.goal = (2, 2, 0)
        # self.goal = (2, 3, m.pi / 2)
        # self.goal = (2, 3, m.pi / 2)
        # self.goal = generate_random_goal()
        # self.goal = (0, 2, 4.71238898038469)
        print("Dubins3D goal",self.goal)
        # self.sampled_trajects = self.generate_trajectories()

    @staticmethod
    def get_features(path, radius):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L1 = path.segment_length(0)
        L3 = path.segment_length(2)
        L = path.path_length()
        IS_curvature = (L1 + L3)**2 + (1 / radius) ** 2
        max_curvature = 1 / radius
        straight_length = L - L1 - L3
        return [L, IS_curvature / L, max_curvature]


    def generate_evaluation_samples(self):
        """

        :return:
        """
        weights = []
        for w1 in np.linspace(0, 1, 20):
            for w2 in np.linspace(0, 1, 20):
                for w3 in np.linspace(0, 1, 20):
                    w = [w1, w2, w3]
                    if sum(w) == 0: continue
                    w = np.divide(w, sum(w))
                    weights.append(list(w))
        opt_trajs = self.find_optima_for_set_of_weights(weights)
        self.sampled_solutions = opt_trajs


class Dubins3DPlannerObstacle(Dubins3DPlanner):

    def __init__(self, num_feat=3):

        super().__init__(num_feat)
        self.goal = (3, 2, -m.pi / 2)
        # self.goal = (3, 2.2, -m.pi / 2)
        # self.goal = (1.5, 1.65, 0)
        # self.goal = (1, 2, m.pi)
        # self.obstacles = [{'x_0': .8, 'y_0': .8, 'x_1': .9, 'y_1': .9}]
        # self.obstacles = [{'x_0': 1.4, 'y_0': .9, 'x_1': 1.6, 'y_1': 1.4}]
        self.obstacles = [
            {'pos': (1.8, 2.0), 'r': .2, },
                          {'pos': (1.8, 1.3), 'r': .2, },
                          {'pos': (0.5, 2), 'r': .2, },
                          ]
        self.generate_trajectories(force_new=True)

    def get_features(self, path, radius):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L1 = path.segment_length(0)
        L3 = path.segment_length(2)
        L = path.path_length()
        IS_curvature = (L1 + L3) + (1 / radius) ** 2
        max_curvature = 1 / radius
        straight_length = L - L1 - L3


        traj, _ = path.sample_many(0.1)

        obst_distances = [0]
        # for obst in self.obstacles:
        #     min_dist = min_distance_to_obst(traj, obst)
        #     obst_distances += [min_dist]
        features = [L, max(obst_distances)*10,IS_curvature]
        # features = [L-4, max(obst_distances)**(1)*10-6, IS_curvature/5] # hand made normalization, not great but meh
        # features = [L-3, max(obst_distances)**(1)*10-5]
        # print('features', features)
        return features