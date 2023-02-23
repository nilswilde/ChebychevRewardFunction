import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math as m
import copy, time, random

from scipy.optimize import linprog
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

import gurobipy as gp
from gurobipy import GRB
def get_distance(vertex, other):
    return m.sqrt((vertex[0] - other[0]) ** 2 + (vertex[1] - other[1]) ** 2)


class Graph:

    def __init__(self, img_file='map.png'):
        """

        :param img_file: a black and white image. white is free space black are static obstacles
        """
        self.img_file = img_file
        self.map_img = mpimg.imread(img_file)
        self.vertices = []  # list with robot configurations / vertices
        self.vertex_obst_dists = {}
        self.edges = {}  # dictionary with (v,u) keys, values are the distance
        self.edge_list = []
        self.neighbours = {}
        self.costs = {}  # dictionary with (v,u) keys, values are the distance
        self.edge_features = {}
        self.weighted_edge_features = {}

        self.dim = 2 # number of features

        self.incidence_matrix = None

        self.x_range = len(self.map_img[0])
        self.y_range = len(self.map_img)
        self.generate_prm(500, 5)

    def generate_prm(self, number_vertices=10, k=8, symmetric=True):
        """

        :param number_vertices: number of vertices to be sampled
        :param k: connection factor
        :param symmetric: should the PRM graph be symmetric? Default: True
        :return:
        """
        np.random.seed(1)
        vertices_reached = []

        min_dist = .05 * self.x_range
        while True:
            print("create PRM graph, n = ", number_vertices, ', k = ', k)
            self.vertices = []

            for _ in range(number_vertices * 10):
                if len(self.vertices) >= number_vertices:
                    break
                v = (int(np.random.random() * self.x_range), int(np.random.random() * self.y_range))
                if self.vertex_in_free_space(v):
                    _, dist = self.get_closest_vertex(v)
                    if dist > min_dist:
                        self.vertices.append(v)
                        dist_to_obst = self.get_dist_to_closest_obst(v)
                        self.vertex_obst_dists[v] = dist_to_obst
            for v in self.vertices:
                self.neighbours[v] = []

            for v in self.vertices:
                dists_to_neighbours = []
                for u in self.vertices:
                    if v != u:
                        dists_to_neighbours.append({'u': u,
                                                    'dist': get_distance(u, v)})
                dists_to_neighbours = sorted(dists_to_neighbours, key=lambda i: i['dist'])
                num_connections = 0
                for i in range(k):
                    u = dists_to_neighbours[i]['u']
                    d = dists_to_neighbours[i]['dist']
                    collision_free = True
                    for step in np.linspace(0, 1, 11):
                        intermediate_point = np.add(np.multiply(step, u), np.multiply((1 - step), v))
                        if not self.vertex_in_free_space(intermediate_point):
                            collision_free = False
                            break
                    if collision_free:
                        if u not in self.neighbours[v]:
                            self.neighbours[v] += [u]

                        if (v,u) not in self.edges.keys():
                            vertices_reached += [u]
                            num_connections += 1
                            self.edges[(v, u)] = d
                            self.costs[(v, u)] = d
                            # compute features
                            self.edge_features[(v, u)] = self.compute_edge_features((v, u))
                            if symmetric and (u, v) not in self.edges.keys():
                                if v not in self.neighbours[u]:
                                    self.neighbours[u] += [v]

                                self.edges[(u, v)] = d
                                self.costs[(u, v)] = d
                                self.edge_features[(u, v)] = self.compute_edge_features((u, v))
                if num_connections == 0 and v not in vertices_reached:  # if a vertex cannot be connected to any other vertex, remove it from the graph
                    self.vertices.remove(v)


            if max(self.edges.values()) < float('inf'):  # check if we were able to construct a connected graph
                break

        self.setup_node_arc_matrix()

    def set_edge_costs(self, w):
        eps = .00001
        # eps = .0000
        for e in self.edges.keys():
            self.costs[e] = np.dot(w, self.edge_features[e]) + eps
            self.weighted_edge_features[e] = list(np.multiply(w, self.edge_features[e])+eps)


    def compute_edge_features(self, e):
        obst_dist = (self.vertex_obst_dists[e[0]] + self.vertex_obst_dists[e[1]]) / 2
        closeness = np.exp(-.05*obst_dist)
        # closeness = (np.max(list(self.vertex_obst_dists.values())) - obst_dist + 1) / 1000000
        return [self.edges[e]/100, closeness*1000]

    def compute_path_features(self, path):
        dist, risk = 0, 0
        for idx in range(len(path) - 1):
            v, u = path[idx], path[idx + 1]
            edge_dist, edge_risk = self.edge_features[(v, u)]
            dist += edge_dist
            risk += edge_risk
            # risk = max(risk, edge_risk)

        return [dist, risk]

    def setup_node_arc_matrix(self):
        self.incidence_matrix = [[0] * len(self.edges) for _ in range(len(self.vertices))]
        edge_list = list(self.edges.keys())
        for e_idx in range(len(edge_list)):
            e = edge_list[e_idx]
            v_idx, u_idx = self.vertices.index(e[0]), self.vertices.index(e[1])
            self.incidence_matrix[v_idx][e_idx] = 1
            self.incidence_matrix[u_idx][e_idx] = -1


    def reconstruct_path(self, s, t, decision_vector_raw):
        decision_vector = [round(x, 2) for x in decision_vector_raw]
        next_pointer = {}
        open_list = []
        edge_list = list(self.edges.keys())
        # print('s,t', s,t)
        for e_idx in range(len(edge_list)):
            if decision_vector[e_idx] == 1:
                v, u = edge_list[e_idx]
                open_list.append([u])
                next_pointer[v] = u
        vertices = [s]
        # print(next_pointer)
        # print(open_list)
        while True:
            v = vertices[-1]
            if v == t:
                return vertices
            if len(vertices)>50:
                print('vertices', self.vertices)
                print('edges', self.edges.keys())
                print('trace', v, 'next', next_pointer[v], vertices)
                self.plot(paths=[vertices])
                raise

            try:
                vertices += [next_pointer[v]]
            except:
                print(decision_vector)
                print(list(self.costs.values()))
                self.plot(paths=[vertices])
                raise
        return None

    def setup_flow_constraints(self, s,t):
        A_eq = copy.deepcopy(self.incidence_matrix)
        s_idx, t_idx = self.vertices.index(s), self.vertices.index(t)
        b_eq = [0] * len(self.vertices)
        b_eq[s_idx] = 1
        b_eq[t_idx] = -1
        return A_eq, b_eq


    def check_connectivity(self):
        for v in self.vertices:
            for u in self.vertices:
                if (v, u) not in self.path_pointers.keys():
                    return False
        return True

    def vertex_in_free_space(self, vertex):
        """
        collision check for png black and white images representing obstacles
        :param vertex:
        :return:
        """
        pixel = self.map_img[int(round(vertex[1]))][int(round(vertex[0]))]
        return pixel[0] != 0

    def get_closest_vertex(self, pos):
        min_dist = float('inf')
        u = None
        for v in self.vertices:
            d = get_distance(v, pos)
            if d < min_dist:
                min_dist = d
                u = v
        return u, min_dist

    def get_dist_to_closest_obst(self, pos):
        t = time.time()
        res = 20
        min_dist = float('inf')
        x_points = np.linspace(0, self.x_range - 1, num=int(self.x_range / res))
        y_points = np.linspace(0, self.y_range - 1, num=int(self.x_range / res))
        for x in x_points:
            for y in y_points:
                if not self.vertex_in_free_space((x, y)):
                    dist = get_distance(pos, (x, y))
                    min_dist = min(min_dist, dist)
        # print("comp obst dist", round(time.time()-t,3), pos, min_dist)
        return min_dist

    def plot(self, fig=None, ax=None, paths=[], title='', show_graph=True, block=True):
        """
        simple plot of the environment only
        :param block:
        :return:
        """
        print("plot graph", paths)
        if fig is None or ax is None:
            print('gen subplots')
            fig, ax = plt.subplots()
        # fig.suptitle(title)
        x, y = [], []
        ax.imshow(self.map_img)

        if show_graph:
            # plot graph edges
            for e in self.edges:
                c = 'lightgrey'
                x_start, y_start = e[0][0], e[0][1]
                x_goal, y_goal = e[1][0], e[1][1]
                ax.plot([x_start, x_goal], [y_start, y_goal], color=c, zorder=1)

            # plot vertices
            for i in range(len(self.vertices)):
                v = self.vertices[i]
                x.append(v[0])
                y.append(v[1])
                ax.scatter(v[0], v[1], color='lightgrey', s=50, zorder=2)

        cols = ['b', 'g', 'r', 'purple']
        for path_idx in range(len(paths)):
            path = paths[path_idx]
            if path is not None:
                for idx in range(len(path)):
                    v = path[idx]
                    x.append(v[0]+random.random()), y.append(v[1]+random.random())
                    ax.scatter(v[0], v[1], color=cols[path_idx], s=60, zorder=3)
                    if idx < len(path) - 1:
                        u = path[idx + 1]
                        ax.plot([v[0], u[0]], [v[1], u[1]], color=cols[path_idx], zorder=1, linewidth = 5)

        plt.show(block=block)
        return fig, ax

    def get_path_positions(self, path, speed=1):

        pos_log = []
        u = path[0]
        for idx in range(len(path) - 1):
            v, u = path[idx], path[idx + 1]
            pos_log += [v]
            # print('is even a vertex??', v, v in self.vertices)
            steps = int(self.edges[(v, u)] / speed) - 1
            for i in range(steps):
                w = (i / steps * u[0] + (1 - i / steps) * v[0], i / steps * u[1] + (1 - i / steps) * v[1])
                pos_log.append(w)
        pos_log.append(u)
        return pos_log

    def animate_path(self, path):
        """

        :param path:
        :return:
        """
        dt = 0.01
        log = self.get_path_positions(path)

        # Initialize the plot
        fig, ax = self.plot(paths=[path], show_graph=False, block=False)

        # Create and add a circle patch to the axis
        patch = Circle(log[0], radius=15)
        ax.add_patch(patch)

        # Animation function to update and return the patch at each frame
        def animate(i):
            patch.center = log[i]
            return patch,

        # Specify the animation parameters and call animate
        ani = FuncAnimation(fig,
                            animate,
                            frames=len(log),  # Total number of frames in the animation
                            interval=int(1000 * dt),  # Set the length of each frame (milliseconds)
                            blit=True,  # Only update patches that have changed (more efficient)
                            repeat=False)  # Only play the animation once

        plt.show()

    def compute_shortest_path(self, s, t, scalarization_mode='linear'):

        return self.shortest_path(s, t, scalarization_mode)
        # if scalarization_mode == 'linear':
        #     return super().shortest_path_LP_minsum(s, t)
        #
        # else:
        #     return super().shortest_path_LP_minmax(s, t)
    def shortest_path_LP_minsum(self, s, t):
        """

        """
        print('shortest path LP minsum')
        # setup float constraints
        A_eq, b_eq = self.setup_flow_constraints(s, t)
        # setup objective
        c = list(self.costs.values())
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1])
        return self.reconstruct_path(s, t, res.x)


    def shortest_path_LP_minmax(self, s, t):
        """

        """
        print('shortest path LP minmax')
        # setup float constraints
        A_eq, b_eq = self.setup_flow_constraints(s, t)
        A_eq = [a_i + [0] for a_i in A_eq]
        # setup objective
        c = [0] * len(list(self.costs.values()))
        c += [1]  # the objective is only minimizing the auxillary variable t

        # setup inequality constraints for min max formulation
        A_ub, b_ub = [], []
        for i in range(self.dim):
            weighted_edge_features = list(self.weighted_edge_features.values())
            weighted_features_i = [elem[i] for elem in weighted_edge_features]
            a_i = weighted_features_i + [-1]
            A_ub += [a_i]
            b_ub += [0]

        bounds = [(0, 1)] * len(self.edges.keys())
        bounds += [(None, None)]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        x = [round(x) for x in res.x]
        return self.reconstruct_path(s, t, x)



    # def shortest_path_LP_minsum(self, s, t):
    #     """
    #
    #     """
    #
    #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #     return self.single_shortest_path_linear(s,t)
    #     A_eq, b_eq = self.setup_flow_constraints(s, t)
    #     A_eq = [a_i + [0] for a_i in A_eq]
    #
    #     c = [0] * len(list(self.costs.values()))
    #     c += [1]  # the objective is only minimizing the auxillary variable t
    #
    #     # setup inequality constraints for min max formulation
    #     weighted_edge_features = list(self.weighted_edge_features.values())
    #
    #     # setup inequality constraints for min max formulation
    #     weighted_edge_features = list(self.weighted_edge_features.values())
    #     A_ub, b_ub = [], []
    #
    #     weighted_dist = [elem[0] for elem in weighted_edge_features]
    #     a_i_basic = weighted_dist + [-1]
    #     weighted_risk = [elem[1] for elem in weighted_edge_features]
    #     for j in range(len(weighted_risk)):
    #         a_i = copy.deepcopy(a_i_basic)
    #         a_i[j] += weighted_risk[j]
    #         A_ub += [a_i]
    #         b_ub += [0]
    #
    #     m = gp.Model("minsum")
    #     m.Params.LogToConsole = 0
    #     # Create variables
    #     for e in range(len(weighted_edge_features)):
    #         m.addVar(vtype=GRB.BINARY, name="x_" + str(e))
    #     t_aux = m.addVar(vtype=GRB.CONTINUOUS, name="t")
    #     m.setObjective(t_aux, GRB.MINIMIZE)
    #     m.addMConstr(np.array(A_eq), None, '=', np.array(b_eq))
    #     m.addMConstr(np.array(A_ub), None, '<=', np.array(b_ub))
    #     m.optimize()
    #
    #     x = []
    #     for v in m.getVars():
    #         if v.VarName != 't':
    #             x.append(int(v.X))
    #
    #     path = self.reconstruct_path(s, t, x)
    #     # self.plot(paths=[path])
    #     return path
    #
    #
    # def shortest_path_LP_minmax(self, s, t):
    #     """
    #
    #     """
    #     t_s = time.time()
    #     print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    #     return self.single_shortest_path_chebyshev(s,t)
    #     # setup float constraints
    #     A_eq, b_eq = self.setup_flow_constraints(s,t)
    #     A_eq = [a_i + [0] for a_i in A_eq]
    #
    #     c = [0] * len(list(self.costs.values()))
    #     c += [1] # the objective is only minimizing the auxillary variable t
    #
    #     # setup inequality constraints for min max formulation
    #     weighted_edge_features = list(self.weighted_edge_features.values())
    #     A_ub, b_ub = [], []
    #     # i = 0 -- the total distance feature
    #     weighted_features_i = [elem[0] for elem in weighted_edge_features]
    #     a_i = weighted_features_i + [-1]
    #     A_ub += [a_i]
    #     b_ub += [0]
    #
    #     # i = 1 the max closeness feature
    #     weighted_features_i = [elem[1] for elem in weighted_edge_features]
    #     # a_i = weighted_features_i + [-1]
    #     # A_ub += [a_i]
    #     # b_ub += [0]
    #     for j in range(len(weighted_features_i)):
    #         a_i = [0]*len(weighted_features_i)
    #         a_i[j] = weighted_features_i[j]
    #         a_i += [-1]
    #         A_ub += [a_i]
    #         b_ub += [0]
    #
    #     m = gp.Model("minmax")
    #     m.Params.LogToConsole = 0
    #     # Create variables
    #     for e in range(len(weighted_features_i)):
    #         m.addVar(vtype=GRB.BINARY, name="x_"+str(e))
    #     t_aux = m.addVar(vtype=GRB.CONTINUOUS, name="t")
    #     m.setObjective(t_aux, GRB.MINIMIZE)
    #     m.addMConstr(np.array(A_eq), None, '=', np.array(b_eq))
    #     m.addMConstr(np.array(A_ub), None, '<=', np.array(b_ub))
    #
    #     m.optimize()
    #
    #     x = []
    #     for v in m.getVars():
    #         if v.VarName != 't':
    #             x.append(int(v.X))
    #
    #     # print('Obj: %g' % m.ObjVal)
    #     # print('x', x)
    #     path = self.reconstruct_path(s, t, x)
    #     # self.plot(paths=[path])
    #     print('ILP time', round(time.time()-t_s, 4))
    #     return path




    def shortest_path(self, s, g, scalarization):
        """

        """
        import heapq
        def _get_cost_of_path(path):
            features_vals = [0, 0]
            length = 0
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1])
                length += self.edges[e]
                weighted_edge_features = self.weighted_edge_features[e]
                features_vals[0] += weighted_edge_features[0]
                features_vals[1] = features_vals[1]+weighted_edge_features[1]
            weighted_cost_vec = features_vals
            if scalarization == 'linear':
                return sum(weighted_cost_vec), length
            else:
                return max(weighted_cost_vec), length

        print('Run shortest path search, cehbyshev', s, g)

        g_score = {}
        f_score = {}
        paths_to_v = {}
        for v in self.vertices:
            g_score[v] = float('inf')
            f_score[v] = float('inf')
            paths_to_v[v] = []

        g_score[s] = 0
        f_score[s] = 0
        paths_to_v[s] = [[]]
        open_set = [(0, s, [])]
        heapq.heapify(open_set)
        max_path_records = 1 if scalarization == 'linear' else 100

        while len(open_set) > 0:
            _, curr, path_to_curr = heapq.heappop(open_set)
            if curr == g:
                break
            neighbours = self.neighbours[curr]
            for neigh in neighbours:
                path_to_neigh = tuple(list(path_to_curr) + [curr])
                tent_score, _ = _get_cost_of_path(list(path_to_neigh) + [neigh])
                if neigh not in path_to_neigh:  # avoid cycles
                    if path_to_neigh not in paths_to_v[neigh] and len(paths_to_v[neigh]) < max_path_records:
                        paths_to_v[neigh] += [path_to_neigh]
                        g_score[neigh] = tent_score
                        f_score[neigh] = tent_score
                        heapq.heappush(open_set, (f_score[neigh], neigh, path_to_neigh))

        best_cost, best_length, best_path = float('inf'), float('inf'), None
        for path in paths_to_v[g]:
            full_path = list(path) + [g]
            cost, length = _get_cost_of_path(full_path)
            if cost < best_cost:
                best_cost = cost
                best_path = full_path

        return best_path
