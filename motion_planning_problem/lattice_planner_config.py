
from math import pi
import numpy as np
from shapely.geometry import Polygon, Point

SOLVER_TIME = 10  # the maximum time for the MILP solver
WORLD_SIZE  = 10  # size of the world


GRID_SIZE = 50 # grid size 
RESOLUTION = [2, 2, pi/2] # resolution of the grid
STATE_RANGE = [
    [-GRID_SIZE, GRID_SIZE],
    [-GRID_SIZE, GRID_SIZE],
    [0, 2*pi],
]
MPRIM_TOL = [0.1, 0.1, 0.05] # tolerance for each component of the state

DUBINS_RADUIS = 1 # minimum radius of the dubins vehicle
PATH_STEP_SIZE = 0.05 # the discretization level of dubins path

RESOLUTION_X = 0.1
RESOLUTION_Y = 0.1

OBSTACLE_MAP = np.ones((int(GRID_SIZE/RESOLUTION_X), int(GRID_SIZE/RESOLUTION_Y)))


OBSTACLES = [
    [50, 250, 100, 200],
    [300, 400, 100, 300],
    [100, 400, 350, 440],
]

for _i in range(len( OBSTACLES)):
    OBSTACLE_MAP[OBSTACLES[_i][0]:OBSTACLES[_i][1], OBSTACLES[_i][2]:OBSTACLES[_i][3]] = 0

OBSTACLE_POLYGONS = []
for _i in range(len( OBSTACLES)):
    OBSTACLE_POLYGONS.append(
        Polygon([[OBSTACLES[_i][0], OBSTACLES[_i][2]],
        [OBSTACLES[_i][1], OBSTACLES[_i][2]],
        [OBSTACLES[_i][1], OBSTACLES[_i][3]],
        [OBSTACLES[_i][0], OBSTACLES[_i][3]]])
    )
