import random

from Planner import *
import math as m
import numpy as np
import dubins

def get_distance(x_1, x_2):
    return m.sqrt((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2)

def in_collision(traj, obstacles=[]):
    # return False
    for pos in traj:
        for o in obstacles:
            if get_distance(pos, o['pos']) < o['r']:
                return True
    return False


def min_distance_to_obst(traj, obst):
    min_dist = float('inf')
    for pos in traj:
        dist = max(0, get_distance(pos, obst['pos'])-obst['r'])
        min_dist = min(min_dist, dist)
    # return 5-min_dist
    return np.exp(-min_dist)


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
        min_radius, max_radius = .1, 1.0
        # min_radius, max_radius = .2, 3.0
        num_basic_samples = int((max_radius - min_radius) * 200) + 1
        radia = np.linspace(min_radius, max_radius, num_basic_samples)
        trajects = []
        for r_idx in range(len(radia)):
            r = radia[r_idx]
            states, f = self.compute_dubins(r)
            if not in_collision(states, self.obstacles):
                trajects.append({'states': states, 'f': f})

        trajects.reverse()
        self.generated_stuff = trajects
        return trajects

    def find_optimum(self, w, radius_sampled_trajects=None):
        """

        :param w:
        :return:
        """
        min_cost = float('inf')
        best_traject = None
        if radius_sampled_trajects is None:
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


class Dubins2DPlanner(DubinsPlanner):

    def __init__(self, scalarization):
        super().__init__(2, scalarization)
        self.goal = (3, 2, -m.pi / 2)
        # self.goal = (1, 2, m.pi)
        print("2D Dubings, goal", self.goal)

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
        radius_sampled_trajects = self.generate_trajectories()
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights, radius_sampled_trajects)


class Dubins2DPlannerObstacle(Dubins2DPlanner):

    def __init__(self, num_feat=2):

        super().__init__(num_feat)
        self.goal = (3, 2, -m.pi / 2)

        goals = [(3, 2.2, -m.pi / 2),(3, 2.2, -m.pi / 3)]
        self.goal = (0.4, 3, -m.pi )
        self.goal = (3, 2.2, -m.pi / 2)

        self.obstacles = [
            {'pos': (1.8, 2.0), 'r': .2, },
                          {'pos': (1.8, 1.3), 'r': .2, },
                          {'pos': (0.5, 2), 'r': .2, },
                          ]
        self.generate_trajectories(force_new=True)
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
        IS_curvature = (L1 + L3) + (1 / radius) ** 2
        max_curvature = 1 / radius
        straight_length = L - L1 - L3


        traj, _ = path.sample_many(0.1)

        obst_distances = []
        for obst in self.obstacles:
            min_dist = min_distance_to_obst(traj, obst)
            obst_distances += [min_dist]
        features = [L, max(obst_distances)]
        # features = [L-4, max(obst_distances)**(1)*10-6] # hand made normalization, not great but meh
        # features = [L-3, max(obst_distances)**(1)*10-5]
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

        obst_distances = []
        for obst in self.obstacles:
            min_dist = min_distance_to_obst(traj, obst)
            obst_distances += [min_dist]
        features = [L, max(obst_distances)*10,IS_curvature]
        # features = [L-4, max(obst_distances)**(1)*10-6, IS_curvature/5] # hand made normalization, not great but meh
        # features = [L-3, max(obst_distances)**(1)*10-5]
        # print('features', features)
        return features