from motion_planning_problem.lattice_planner_config import RESOLUTION
from motion_planning_problem.lattice_planner_config import STATE_RANGE
from motion_planning_problem.lattice_planner_config import MPRIM_TOL
from motion_planning_problem.lattice_planner_config import OBSTACLE_MAP, RESOLUTION_X, RESOLUTION_Y, OBSTACLES, OBSTACLE_POLYGONS
from Planner import *

from motion_planning_problem.MotionPrimitive import create_euclidean_motion_primitives
from math import pi, sqrt, exp
from shapely.geometry import Polygon, Point

import numpy as np
from time import time 


euc_state_changes = [
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (-1, -1, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (2, 1, 0),
        (2, -1, 0),
        (-2, 1, 0),
        (-2, -1, 0),
    ]

    # euc_state_changes = [
    #     # (1, 1, 0),
    #     # (1, -1, 0),
    #     # (-1, 1, 0),
    #     (2, 0, 0),
    #     (0, 2, 0),
    #     # (-1, -1, 0),
    #     (0, -2, 0),
    #     (-2, 0, 0),
    #     # (2, 1, 0),
    #     # (2, -1, 0),
    #     # (-2, 1, 0),
    #     # (-2, -1, 0),
    # ]


mps = create_euclidean_motion_primitives(
    state_changes = euc_state_changes
)

def euc_dist(state1, state2):
    return sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)

def heuristic_euc(state1, state2):
    return sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)

def candidate_feature_1(state, mp):
    """
    feature is the cost of the motion primitive
    """
    return mp.cost

def candidate_feature_2(state, mp):
    """
    feature is angle between the arriving mp and the departing mp
    """
    angle =  abs(state[2] - mp.goal[2]) 
    return min([angle, 2*pi - angle])

def candidate_feature_3(state, mp):
    """
    feature that calculates the distance to the closest obstacle
    """
    min_dist = 500

    index_x = int((state[0] + mp.goal[0]/2.0)/RESOLUTION_X)
    index_y = int((state[1] + mp.goal[1]/2.0)/RESOLUTION_Y)
    
    p = Point(index_x, index_y)

    for obstacle in OBSTACLE_POLYGONS:
        dist = obstacle.exterior.distance(p)
        if dist < min_dist:
            min_dist = dist
    return exp(-min_dist*RESOLUTION_X)


class StateLattice(Planner):
    def __init__(self, num_feat=3):
        super().__init__(num_feat, 'MP'+str(num_feat)+'D')
        self._mps = mps
        self._vertices = []
        self._edges = {}
        self._verbose = False
        self._rotation = 0
        self._path = None
        self.start = [49, 0, -pi / 2]
        self.goal = [5, 40, 0]
        features = [candidate_feature_1, candidate_feature_2, candidate_feature_3]
        self._features = features[0:num_feat]
    
    @staticmethod
    def is_reached_goal(state, goal):
        if euc_dist(state, goal) < 1:
            return True
        return False

    def is_state_reached(self, state):
        for _s in self._states:
            equal = True
            for _i in range(len(_s)):
                if not _s[_i] == state[_i]:
                    equal = False
                    break
        return equal

    def _parse_path(self, parent, final_index):
        """
        returns the path to the goal
        """
        self._path = [(self._states[final_index], parent[final_index][1])]
        current = final_index
        while not parent[current][0] == 0:
            current = parent[current][0]
            self._path.append((self._states[current], parent[current][1]))
        
        self._path.append((self._states[0], parent[0][1]))
        self._path.reverse()
    

    def _is_state_colliding(self, state):
        
        n, m = OBSTACLE_MAP.shape
        index_x = int(state[0]/RESOLUTION_X)
        index_y = int(state[1]/RESOLUTION_Y)

        if index_x > n - 1 or index_y > m - 1 or index_x < 0 or index_y < 0:
            return True

        if OBSTACLE_MAP[index_x, index_y] == 0:
            return True
        return False

    def _is_colliding(self, state, mp):
        n, m = OBSTACLE_MAP.shape
        for _i in range(len(mp.edge)):
            index_x = int((state[0] + mp.edge[_i][0])/RESOLUTION_X)
            index_y = int((state[1] + mp.edge[_i][1])/RESOLUTION_Y)

            if index_x > n - 1 or index_y > m - 1:
                continue
            if OBSTACLE_MAP[index_x, index_y] == 0:
                return True
        return False

    def _is_equal(self, state1, state2):
        for _i in range(len(state1)):
            if not(state1[_i] == state2[_i]):
                return False
        return True

    def _get_new_state_index(self, state):
        for _i in range(len(self._states)):
            if self._is_equal(state, self._states[_i]):
                return _i
        
        raise ValueError

    def A_star(self, start, goal, w, h = heuristic_euc):
        """
        find the path to goal using A_star
        @param goal: goal location
        @param h: the heuristic function
        """
        from queue import PriorityQueue
        
        self._goal = goal
        num_features = len(self._features)
        w = np.array(w)

        self._states = [start]
        
        open_set = PriorityQueue()
        g_score = [0]
        f_score = [w[0]*h(start, goal)]
        features_costs = [[0 for _i in range(num_features)]]
        
        states_hash = {}
        states_hash[tuple(start)] = 0

        in_open_set = [1]
        open_set.put((f_score[0], 0))
        
        parent = [(0, -1)]
        iter_count = 0
        while not open_set.empty():
            iter_count += 1
            _, current_index = open_set.get()
            current_state = self._states[current_index]
            in_open_set[current_index] = 0

            if StateLattice.is_reached_goal(current_state, goal):
                # print("Reached the goal!")
                self._parse_path(parent, current_index)
                return features_costs[current_index]
            
            for _i in range(len(self._mps)):
                mp = self._mps[_i]
                candidate_state = np.array(current_state) + np.array(mp.goal)

                # this is added just for omni-directional -- remove for Dubins
                candidate_state[2] = mp.goal[2]

                if self._is_colliding(current_state, mp):
                    continue
                
                features = np.array(
                    [
                        self._features[_ff](current_state, mp) for _ff in range(num_features)
                    ])
                
                dist = np.dot(w, features) + g_score[current_index]
                if tuple(candidate_state) in states_hash:
                    index = states_hash[tuple(candidate_state)]
                    if dist >= g_score[index]:
                        continue
                    # if found a better path to an already observed state
                    # add it back to the open set
                    if in_open_set[index] == 0:
                        in_open_set[index] = 1
                        open_set.put((f_score[index] + g_score[index], index))
                    g_score[index] = dist
                    parent[index] = (current_index, _i)
                    features_costs[index] = [
                        features_costs[current_index][_ff] + features[_ff] for _ff in range(num_features)
                    ]
                else:
                    if self._is_state_colliding(current_state):
                        continue

                    self._states.append(candidate_state)
                    g_score.append(dist)
                    f_score.append(w[0]*h(candidate_state, goal))
                    n = len(self._states) - 1
                    open_set.put((f_score[n] + g_score[n], n))
                    in_open_set.append(1)
                    parent.append((current_index, _i))
                    states_hash[tuple(candidate_state)] = n
                    features_costs.append([
                        features_costs[current_index][_ff] + features[_ff] for _ff in range(num_features)
                    ])
            # if iter_count % 20 == 0:
            #     self.plot(current_vertex=current_index, save=True, filename='res/' + str(iter_count) + ".png")

    def find_optimum(self, w, start=None, goal=None):
        if start is None: start = self.start
        if goal is None: goal = self.goal
        costs = self.A_star(start, goal, w)
        return {'w':w, 'f':costs}

    def generate_evaluation_samples(self):
        """

        :return:
        """
        weights = []
        for w1 in np.linspace(0, 1, 10):
            for w2 in np.linspace(0, 1, 10):
                for w3 in np.linspace(0, 1, 10):
                    w = [w1, w2, w3]
                    if sum(w) == 0: continue
                    w = np.divide(w, sum(w))
                    weights.append(list(w))
        opt_trajs = self.find_optima_for_set_of_weights(weights)
        self.sampled_solutions = opt_trajs

    def plot(self, current_vertex = None, save=False, filename ='fig.png'):
        """
        plot the lattice and the path
        """
        import matplotlib.pylab as plt

        # for _i in range(len(self._states)):
        #     plt.plot(
        #         self._states[_i][1]/RESOLUTION_Y,
        #         self._states[_i][0]/RESOLUTION_X,
        #         'bo'
        #     ) 
        
        if not (current_vertex == None):
            plt.plot(
                self._states[current_vertex][1]/RESOLUTION_Y,
                self._states[current_vertex][0]/RESOLUTION_X,
                'go'
            )

        plt.plot(self._goal[1]/RESOLUTION_Y, self._goal[0]/RESOLUTION_X, 'r*')
        for obstacle in OBSTACLE_POLYGONS:
            x,y = obstacle.exterior.xy
            plt.plot(y,x)
        plt.imshow(OBSTACLE_MAP, cmap='gray', vmin=0, vmax=1)

        if not (self._path == None):
            for _i in range(len(self._path) - 1):
                plt.plot(
                    [self._path[_i][0][1]/RESOLUTION_Y, self._path[_i + 1][0][1]/RESOLUTION_Y],
                    [self._path[_i][0][0]/RESOLUTION_X, self._path[_i + 1][0][0]/RESOLUTION_X],
                    'r-',
                    linewidth=4
                )
        

        if save == True:
            plt.savefig(filename)
        else:
            plt.show()
    

# test

if __name__ == "__main__":
    euc_state_changes = [
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (-1, -1, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (2, 1, 0),
        (2, -1, 0),
        (-2, 1, 0),
        (-2, -1, 0),
    ]

    # euc_state_changes = [
    #     # (1, 1, 0),
    #     # (1, -1, 0),
    #     # (-1, 1, 0),
    #     (2, 0, 0),
    #     (0, 2, 0),
    #     # (-1, -1, 0),
    #     (0, -2, 0),
    #     (-2, 0, 0),
    #     # (2, 1, 0),
    #     # (2, -1, 0),
    #     # (-2, 1, 0),
    #     # (-2, -1, 0),
    # ]


    mps = create_euclidean_motion_primitives(
        state_changes = euc_state_changes
    )
    lattice = StateLattice(num_feat=3)
    print(lattice.find_optimum([1, 0.5, 10], [49, 0, -pi/2], [5, 40, 0]))
    # print(lattice.find_optimum([0, 1, 0], [49, 0, -pi/2], [5, 40, 0]))
    lattice.plot()