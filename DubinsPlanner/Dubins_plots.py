import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Rectangle
import numpy as np

from DubinsPlanner.auxillary import *

relabel= {'linear':'Linear User Model',
          'chebyshev':'Chebyshev User Model'}

def illustrate_2d(planner, robust_samples, highlight=None,label='',block=True):
    """

    :param planner:
    :param robust_samples:
    :param label:
    :param block:
    :return:
    """
    # if len(planner.sampled_solutions) == 0:
    #     planner.generate_evaluation_samples()
    opt_trajs = planner.sampled_solutions
    weights = [traj['w'] for traj in opt_trajs]

    plot_trajects_and_features(planner, robust_samples, [traj['w'][0] for traj in robust_samples], highlight,
                               x_label='w_1', title=relabel[label])
    # plot_approximation(planner, weights, opt_trajs, robust_samples, highlight, title=label)
    plt.show(block=block)


def plot_trajects_and_features(planner, trajects,  x_axis_vals, highlight, x_label='idx', title=''):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    all_sols = planner.generate_trajectories()
    print("plotting", len(planner.sampled_solutions), 'sampled solutions')
    ax = axes[0]
    ax.set_title('' + title, fontsize=18)
    for o in planner.obstacles:
        circle1 = plt.Circle(o['pos'], o['r'], color='red',alpha=0.5)
        ax.add_patch(circle1)
        # ax.add_patch(Rectangle((o['x_0'], o['y_0']), o['x_1']-o['x_0'], o['y_1']-o['y_0']))
    for traj in all_sols:
        ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='grey')
    # for traj in trajects:
    #     ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']])
    # for traj in planner.sampled_solutions:
    for traj in trajects:
        ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']])
    # if highlight is not None:
    #     ax.plot([x[0] for x in highlight['states']], [x[1] for x in highlight['states']], linewidth=4, color='red')

    ax.set_xlabel('x pos', fontsize=16)
    ax.set_ylabel('y pos', fontsize=16)
    # axes[0].set_title('Sampled Trajectories'+title)
    ax.set_aspect('equal')

    # plot Pareto front

    ax = axes[1]
    # plot ground set of trajectories

    phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
    ax.plot(phi_1, phi_2, 'x', color='grey', label='all trajects')
    # plot ground truth pareto curve
    # phi_1, phi_2 = [traj['f'][0] for traj in planner.sampled_solutions], [traj['f'][1] for traj in planner.sampled_solutions]
    phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
    ax.plot(phi_1, phi_2, 'D', color='r', label='optimal trajectories')
    # plot sampled pareto curve
    # phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
    # ax.plot(phi_1, phi_2, 'o', color='b', label='optimal trajectories', markersize=6)

    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    ax.set_aspect(asp)
    if highlight is not None:
        ax.plot(highlight['f'][0], highlight['f'][1], 'D', color='green', label='optimal trajectories', markersize=6)
    ax.set_xlabel('Trajectory Length', fontsize=16)
    ax.set_ylabel('Closeness', fontsize=16)

    fig.tight_layout()
    plt.savefig(title+'_trajects')


def plot_approximation(planner, weights, opt_trajs, samples, highlight, title=''):

    title_relabel = {'Greedy Regret Sampling':'$\mathtt{MRPS}$',
                     'Uniform Sampling':'$\mathtt{Uniform}$',
                     'Expected':'$\mathtt{Expected}$' }
    # plot the optimal cost, i.e., u(w | w) and the regret of some example weight
    opt_costs = [planner.get_cost_of_traj(traj, traj['w'])  for traj in opt_trajs]
    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3,1]}, figsize=(10, 8),sharex=True)
    ax = axes[0]
    col='blue'
    for i in range(len(samples)):
        ax.plot([w[0] for w in weights], [planner.get_cost_of_traj(samples[i], w) for w in weights],
                     label='cost c(' + str(samples[i] ) + '|w)', linewidth=3)
    ax.plot([w[0] for w in weights], opt_costs, color = 'black', label='opt cost c(w|w)', linewidth=5,zorder=100)

    ax.set_ylabel('Cost', fontsize=24, color=col)
    ax.tick_params(axis='y', labelcolor=col,labelsize=20)
    # ax.set_title(title_relabel[title], fontsize=26)
    ax.set_title(title, fontsize=26)
    # ax.set_ylim([4.0,5])
    # ax.set_ylim([0.0,4])

    ax = axes[1]
    col = 'purple'
    approx = []
    for i in range(len(weights)):
        i_approx = []
        for j in range(len(samples)):
            i_approx += [planner.get_cost_of_traj(samples[j], weights[i])]
        approx += [min(i_approx) - opt_costs[i]]
    ax.plot([w[0] for w in weights], approx, color=col, linewidth=3)
    ax.set_ylabel('Min Regret', fontsize=24, color=col)
    ax.tick_params(axis='y', labelcolor=col,labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlabel('Weight $w_1$', fontsize=24)
    ax.legend()
    # ax.set_title(title)
    # ax.set_ylim([-.02, .42])

    fig.tight_layout()
    plt.savefig(title + '_costs')


def plot_regret_3d(weights, regrets):
    fig = plt.figure()
    X, Y = np.meshgrid([w[0] for w in weights], [w[0] for w in weights])
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, np.array(regrets), rstride=1, cstride=1, cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    ax.set_xlabel('Planner w^P', fontsize=14)
    ax.set_ylabel('Observer w^Q', fontsize=16)
    ax.set_zlabel('r(w^P | w^Q)', fontsize=14)


def plot_pareto_approximation(opt_trajects, title='', block=False):
    """

    :param opt_trajects:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))
    phi_1, phi_2 = [traj['f'][0] for traj in opt_trajects], [traj['f'][1]for traj in opt_trajects]
    ax.plot(phi_1, phi_2, 'x', color='b', label='optimal trajectories',markersize = 6)
    ax.set_xlabel('Features $f_1$', fontsize=16)
    ax.set_ylabel('Features $f_2$', fontsize=16)
    ax.set_title('Approximated Pareto Front', fontsize=18)

    fig.tight_layout()
    plt.savefig('plots/'+title + '_pareto')
    if block:
        plt.show()


def plot_pareto_compare(list_samples, labels, title='', block=False):
    """

    :param list_samples:
    :param labels:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))
    cols = ['b', 'r']
    symbols = ['o', 'x']
    for i in range(len(list_samples)):
        opt_trajects = list_samples[i]
        phi_1, phi_2 = [traj['f'][0] for traj in opt_trajects], [traj['f'][1]for traj in opt_trajects]
        ax.plot(phi_1, phi_2, symbols[i], color=cols[i], label=labels[i],markersize = 10)
    ax.set_xlabel('Features $f_1$', fontsize=16)
    ax.set_ylabel('Features $f_2$', fontsize=16)
    ax.set_title('Approximated Pareto Front', fontsize=18)
    plt.legend()
    fig.tight_layout()

    plt.savefig('plots/'+title + '_pareto')

    if block:
        plt.show()
