from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import copy
from  matplotlib.ticker import FuncFormatter
matplotlib.rcParams["legend.frameon"] = False
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True

plt.rc('legend', fontsize=16)  # using a size in points

error_labels = {'regret': 'Regret',
                'relative regret': 'Relative Regret',
                'Alignment Error': 'Alignment',
                'loglikelihood': 'Log-Likelihood',
                'slider': 'Slider'
                }

path = 'simulation_data/'
# path = 'simulation_data current/'
# path = 'simulation_data good ish !/'


def main():
    df = pd.DataFrame()
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if "csv" in file:
            print('file', file)
            df_file = pd.read_csv(path + "/" + file, index_col=0)
            df = pd.concat([df, df_file], ignore_index=True)
    # print(df)
    # df = replace_solver_labels(df)
    # measures_plot_pairs(df, 'relative regret', 'curr', K=20)
    # measures_plot_pairs(df, 'relative regret', 'exp')
    measures_plot_pairs2(df, 'curr')
    # measures_plot_pairs2(df, 'exp')
    plt.show()


def replace_solver_labels(df):
    relabel = {
        'GreedyHeuristic_maxregret': '$\mathtt{Greedy-Regret}$',
        'GreedyHeuristic_randomPosterior': '$\mathtt{Greedy-Random}$',
        'Uniform_maxregret': '$\mathtt{Uniform-Regret}$',
        'Uniform_randomPosterior': '$\mathtt{Uniform-Random}$',
        'linear': '$\mathtt{Linear}$',
        # 'Relativ'
    }
    for old_label in relabel.keys():
        df['label'] = df['label'].replace([old_label], relabel[old_label])
    return df


def measures_plot_pairs(df, measure='alignment', estimator='curr', K=20):
    """
    :param df:
    :param hueorder:
    :return:
    """
    users = ['Uniform', 'GreedyHeuristic', 'all', 'random']
    users = ['Uniform', 'GreedyHeuristic']
    users = ['linear', 'linear']
    fig, axes = plt.subplots(nrows=1, ncols=len(users), figsize=(6, 6), sharey=True)

    colors = ["#f46513"] * len(users) + ["#0d58f8"] * len(users)
    plt.rcParams["legend.loc"] = 'lower right'
    for idx in range(len(users)):
        sns.set_palette(sns.color_palette([colors[idx], colors[idx + len(users)], '#b8b8b8']))
        df_tmp = copy.deepcopy(df)
        df_tmp = df_tmp[(df_tmp['estimator'] == estimator)]
        if users[idx] != 'all':
            df_tmp = df_tmp[(df_tmp['user_weightmode'] == users[idx])]
        else:
            df_tmp = df_tmp[(df_tmp['user_weightmode'] != 'random')]
        df_tmp = df_tmp[(df_tmp['K'] == K)]
        hueorder = None
        print(df_tmp)
        sns.lineplot(ax=axes[idx], x='iter', y=measure, hue='label', data=df_tmp, ci=90)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].set_xlabel("Iteration", fontsize=22)
        axes[idx].tick_params(axis='x', labelsize=20)
        axes[idx].tick_params(axis='y', labelsize=20)
        axes[idx].set_xlabel("Iteration", fontsize=22)
        axes[idx].set_ylabel(error_labels[measure], fontsize=22)
        axes[idx].title.set_text(users[idx])
        axes[idx].get_legend().set_visible(False)
        handles, labels = axes[idx].get_legend_handles_labels()

        if idx == 0:
            axes[idx].legend(handles, labels, loc='lower left', fontsize=20)
            # axes[idx].legend().set_title('') bbox_to_anchor=(1,0),loc='lower right',

    # fig.tight_layout()
    # plt.savefig(path + "plot_pairs_" + str(int(sigma * 10)) + "_" + str(measure))


def measures_plot_pairs2(df, estimator):
    """
    :param df:
    :param hueorder:
    :return:
    """

    user_scalarizations = ['linear','chebyshev']
    learner_scalarizations = ['linear','chebyshev']
    fig, axes = plt.subplots(nrows=2, ncols=len(user_scalarizations), figsize=(5*len(user_scalarizations), 5*len(learner_scalarizations)), sharex=True, sharey='row')
    # if len(user_scalarizations) == 1:
    #     axes = [axes]
    colors = ["#f46513"] * len(user_scalarizations) + ["#0d58f8"] * len(user_scalarizations)
    plt.rcParams["legend.loc"] = 'lower right'
    for idx in range(len(user_scalarizations)):
        for j in range(len(user_scalarizations)):
            ax = axes[idx][j]
            palette = sns.color_palette([colors[idx], colors[idx + len(user_scalarizations)], '#b8b8b8'])
            # palette = sns.color_palette(['#a1b1d5', "#5c8cf9", "#f7bd8c","#f46513"])
            palette = sns.color_palette(['red', "blue", 'orange'])
            sns.set_palette(palette)
            df_tmp = copy.deepcopy(df)
            # df_tmp = df_tmp[(df_tmp['user_weightmode'] != 'all')]
            # df_tmp = df_tmp[(df_tmp['user_weightmode'] == 'linear')]
            df_tmp = df_tmp[(df_tmp['estimator'] == estimator)]
            # df_tmp = df_tmp[(df_tmp['K'] == K_Vals[idx])]
            df_tmp = df_tmp[(df_tmp['user_scalarization'] == user_scalarizations[idx])]
            df_tmp = df_tmp[(df_tmp['learner_scalarization'] == learner_scalarizations[j])]
            hueorder = None
            print(df_tmp)
            # sns.lineplot(ax=ax, x='iter', y='relative regret', hue='label', data=df_tmp, ci=90, palette='Set1')
            sns.boxplot(ax=ax, x='iter', y='relative regret', hue='label', data=df_tmp,  palette='Set1', showfliers=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if idx == 1:
                ax.set_xlabel("Iteration", fontsize=18)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            if j == 0:
                ax.set_ylabel(error_labels['relative regret'], fontsize=18)
            else:
                ax.set_ylabel('', fontsize=18)
            ax.set_title('U: '+str(user_scalarizations[idx])+', L: '+str(learner_scalarizations[j]), fontsize=20)
            ax.get_legend().set_visible(False)
            handles, labels = ax.get_legend_handles_labels()

            if idx == 0 and j ==0:
                ax.legend(handles, labels, loc='upper right', fontsize=20)
    fig.suptitle(estimator)

    fig.tight_layout()


if __name__ == "__main__":
    main()