from distutils.command.config import config
from motion_planning_problem.lattice_planner_config import DUBINS_RADUIS, PATH_STEP_SIZE
import numpy as np
from math import sqrt, pi

def euc_dist(v1, v2):
    return sqrt(
        (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 
    )

class MotionPrimitive:
    """Motion primitive class representing the set of actions
    at each point
    """
    def __init__(self, start, goal, cost, edge = None, restrictive_states = None):
        self.start = start
        self.goal = goal
        self.cost = cost
        self.edge = edge
        # represent the set of the part of the robot state that should be equal
        # while concatenating two motion primitives
        self.rs = restrictive_states 


def create_dubins_motion_primitives(state_changes):
    """creates a set of motion primitives for dubins vehicle

    Args:
        location_change ([type]): the change in the location of dubins vehicle
        headings ([type]): the headings the vehicle at the end point
    """
    try:
        import dubins
    except:
        print("Error in importing dubins!")
        return []
    
    motion_primitives = []
    for state_change in state_changes:
        q0 = (0, 0, 0)
        q1 = (state_change[0], state_change[1], state_change[2])
        path = dubins.shortest_path(q0, q1, DUBINS_RADUIS)
        configurations, _ = path.sample_many(PATH_STEP_SIZE)
        motion_primitives.append(
            MotionPrimitive(
                start = [0, 0, 0],
                goal = [state_change[0], state_change[1], state_change[2]],
                cost = path.path_length(),
                edge = configurations
            )
        )
    return motion_primitives


def create_euclidean_motion_primitives(state_changes):
    """create a set of motion primitives for omni-directional vehicles
    
    Args:
        location_change ([type]): the change in the location of the vehicle
    """
    from math import atan2

    motion_primitives = []
    for state_change in state_changes:
        length = euc_dist([0,0], state_change)
        num_steps = int(length/PATH_STEP_SIZE)
        
        dir = [
            state_change[0]/length*PATH_STEP_SIZE,
            state_change[1]/length*PATH_STEP_SIZE
        ]
        edge = [[dir[0]*_i, dir[1]*_i] for _i in range(num_steps)]

        state_change = [
            state_change[0],
            state_change[1],
            atan2(dir[1], dir[0])
        ]

        motion_primitives.append(
            MotionPrimitive(
                start = [0, 0, 0],
                goal = state_change,
                cost = length,
                edge = edge,
                restrictive_states=[]
            )
        )

    return motion_primitives



def plot_motion_primitives(motion_primitives):
    """plot the set of motion primitives
    """
    try:
        import matplotlib.pylab as plt
    except:
        print("Error in importing matplotlib!")
        return
    
    
    for mp in motion_primitives:
        if not (mp.edge == None):
            qs = np.array(mp.edge)
            xs = qs[:, 0]
            ys = qs[:, 1]
            plt.plot(xs, ys, 'b-')
        else:
            plt.plot([0, mp.goal[0]], [0, mp.goal[1]], 'b-')

        plt.plot(mp.goal[0], mp.goal[1],'r*')

    plt.axis('equal')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

# test 

if __name__ == "__main__":
    euc_state_changes = euc_state_changes = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (1, 0),
        (0, 1),
    ]


    dubins_state_changes = [
        (2, 2, 0),
        (2, 2, pi/2), 
        (2, 1, 0),
        (2, 0, 0),
        (2, -2, 0),
        (2, -2, -pi/2), 
    ]

    # motion_primitives = create_dubins_motion_primitives(
    #     state_changes=state_changes,
    # )

    motion_primitives = create_euclidean_motion_primitives(euc_state_changes)
    plot_motion_primitives(motion_primitives)