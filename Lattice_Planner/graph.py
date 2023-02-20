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


def get_distance(vertex, other):
    return m.sqrt((vertex[0] - other[0]) ** 2 + (vertex[1] - other[1]) ** 2)



class Graph:

    def __init__(self, img_file='Lattice_Planner/map_simple.png'):
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

        self.incidence_matrix = None

        self.x_range = len(self.map_img[0])
        self.y_range = len(self.map_img)
        self.generate_prm(50,4)


    def generate_prm(self, number_vertices=10, k=8, symmetric=True):
        """

        :param number_vertices: number of vertices to be sampled
        :param k: connection factor
        :param symmetric: should the PRM graph be symmetric? Default: True
        :return:
        """
        min_dist = .05 * self.x_range
        while True:
            print("create PRM graph, n = ", number_vertices, ', k = ', k)
            self.vertices = []

            for _ in range(number_vertices*10):
                if len(self.vertices) >= number_vertices:
                    break
                v = (int(np.random.random() * self.x_range), int(np.random.random() * self.y_range))
                if self.vertex_in_free_space(v):
                    _, dist = self.get_closest_vertex(v)
                    if dist>min_dist:
                        self.vertices.append(v)
                        dist_to_obst = self.get_dist_to_closest_obst(v)
                        self.vertex_obst_dists[v] = dist_to_obst

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
                    collision_free = True
                    for step in np.linspace(0, 1, 11):
                        intermediate_point = np.add(np.multiply(step, u), np.multiply((1 - step), v))
                        if not self.vertex_in_free_space(intermediate_point):
                            collision_free = False
                            break
                    if collision_free:
                        num_connections += 1
                        self.edges[(v, u)] = dists_to_neighbours[i]['dist']
                        self.costs[(v, u)] = dists_to_neighbours[i]['dist']
                        # compute features
                        self.edge_features[(v,u)] = self.compute_edge_features((v,u))
                        if symmetric:
                            self.edges[(u, v)] = dists_to_neighbours[i]['dist']
                            self.costs[(u, v)] = dists_to_neighbours[i]['dist']
                            self.edge_features[(u, v)] = self.compute_edge_features((u,v))
                if num_connections == 0:  # if a vertex cannot be connected to any other vertex, remove it from the graph
                    self.vertices.remove(v)

            if max(self.edges.values()) < float('inf'): # check if we were able to construct a connected graph
                break

        self.setup_node_arc_matrix()

    def set_edge_costs(self, w, mode='linear'):
        for e in self.edges.keys():
            if mode == 'linear':
                self.costs[e] = np.dot(w, self.edge_features[e])
            else:
                self.costs[e] = np.max(np.multiply(w, self.edge_features[e]))
        # self.find_all_pairs_shortest_distances()

    def compute_edge_features(self, e):
        obst_dist = (self.get_dist_to_closest_obst(e[0])+self.get_dist_to_closest_obst(e[1])) / 2
        return [self.edges[e], 10000*np.exp(-.1*obst_dist)]

    def compute_path_features(self, path):
        dist, risk = 0, 0
        for idx in range(len(path)-1):
            v, u = path[idx], path[idx+1]
            edge_dist, edge_risk = self.edge_features[(v,u)]
            dist += edge_dist
            risk += edge_risk

        return [dist, risk]


    def setup_node_arc_matrix(self):
        self.incidence_matrix = [[0]*len(self.edges) for _ in range(len(self.vertices))]
        edge_list = list(self.edges.keys())
        for e_idx in range(len(edge_list)):
            e = edge_list[e_idx]
            v_idx, u_idx = self.vertices.index(e[0]), self.vertices.index(e[1])
            self.incidence_matrix[v_idx][e_idx] = 1
            self.incidence_matrix[u_idx][e_idx] = -1



    def compute_shortest_path(self, s, t, scalarization_mode='linear'):
        if scalarization_mode == 'linear':
            return self.shortest_path_LP_minsum(s, t)
        else:
            return self.shortest_path_LP_minsum(s, t)

    def reconstruct_path(self, s, t, decision_vector):
        next_pointer = {}
        edge_list = list(self.edges.keys())
        for e_idx in range(len(edge_list)):
            if decision_vector[e_idx] == 1:
                v, u = edge_list[e_idx]
                next_pointer[v] = u
        vertices = [s]
        while True:
            v = vertices[-1]
            if v == t:
                return vertices
            vertices += [next_pointer[v]]

        return None

    def shortest_path_LP_minsum(self, s, t):
        """

        """


        # setup float constraints
        A_eq = self.incidence_matrix
        b_eq = [0] * len(self.vertices)
        s_idx, t_idx = self.vertices.index(s), self.vertices.index(t)
        b_eq[s_idx] = 1
        b_eq[t_idx] = -1
        # setup objective
        c = list(self.costs.values())
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds= [0,1])
        x = [round(x) for x in res.x]
        return self.reconstruct_path(s, t, x)


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
        res = 50
        min_dist = float('inf')
        x_points = np.linspace(0, self.x_range-1, num=int(self.x_range/res))
        y_points = np.linspace(0, self.y_range-1, num=int(self.x_range/res))
        for x in x_points:
            for y in y_points:
                if not self.vertex_in_free_space((x,y)):
                    dist = get_distance(pos, (x,y))
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
                c = 'grey'
                x_start, y_start = e[0][0], e[0][1]
                x_goal, y_goal = e[1][0], e[1][1]
                ax.plot([x_start, x_goal], [y_start, y_goal], color=c, zorder=1)


            # plot vertices
            for i in range(len(self.vertices)):
                v = self.vertices[i]
                x.append(v[0])
                y.append(v[1])
                ax.scatter(v[0], v[1], color='lightgrey', s=50, zorder=2)


        for path in paths:
            if path is not None:
                for idx in range(len(path)):
                    v = path[idx]
                    x.append(v[0]), y.append(v[1])
                    ax.scatter(v[0], v[1], color='g', s=60, zorder=3)
                    if idx < len(path) - 1:
                        u = path[idx + 1]
                        ax.plot([v[0], u[0]], [v[1], u[1]], color='g', zorder=1)

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


