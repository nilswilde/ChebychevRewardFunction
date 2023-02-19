from algorithm import *

from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path = 'simulation_data/'
experiment = 'Dubins 3 Features'
experiment = 'Dubin2D'
# experiment = 'mTSP15_10'
relabel = {
        'max_regret': 'Max Regret',
        'max_relative_regret': 'Max Relative Regret',
        'hypervolume': 'Hypervolume',
    }
def replace_measure_labels(df):

    for old_label in relabel.keys():
        df['label'] = df['label'].replace([old_label], relabel[old_label])
    return df

def replace_solver_labels(df):
    sampler_labels = {
        'GreedyHeuristic': '$\mathtt{MRPS}$',
        'Uniform': '$\mathtt{Uniform}$',
        'Expected': '$\mathtt{Expected}$',
    }
    for old_label in sampler_labels.keys():
        print("REPLACING", old_label, sampler_labels[old_label])
        df['sampler'] = df['sampler'].replace([old_label], sampler_labels[old_label])
    return df

def compare_plot(df):


    measures = ['Max Regret', 'Max Relative Regret', 'Hypervolume']
    measures = ['Max Regret', 'Max Relative Regret']
    measures = ['Max Regret', 'Max Relative Regret','total_regret','total_relative_regret']
    with sns.axes_style("white"):
        fig, axes = plt.subplots(nrows=1, ncols=len(measures), figsize=(14, 4), sharey=False, sharex=True)
    sns.set_style("whitegrid")
    # fig.suptitle('\n\nDubins Trajectories with 3 features', fontsize=16)
    fig.suptitle('\n', fontsize=16)

    for measure in measures:
        for k in [3, 5, 10]:
            df1 = copy.deepcopy(df)
            df1 = df1[(df1['K'] == k)]
            a = df1[(df1['sampler'] == "Greedy Heuristic")]
            a = np.array(a[measure])
            # b = df1[(df1['solver'] == "Greedy Optimist")]
            b = df1[(df1['sampler'] == "Uniform")]
            b = np.array(b[measure])
            print('K=', k, measure, stats.ttest_ind(a, b, equal_var=False))



    x = 'K'
    for idx in range(len(axes)):
        ax = axes[idx]
        hue_order = ['$\mathtt{MRPS}$', '$\mathtt{Uniform}$','$\mathtt{Expected}$']
        pal = [ sns.color_palette("Paired")[7],sns.color_palette("Paired")[1],sns.color_palette("Paired")[4]]
        sns.boxplot(ax=axes[idx], y=measures[idx], data=df, x=x, hue='sampler', hue_order= hue_order,  showmeans=False,
                    palette=pal, showfliers=True)

        ax.get_legend().set_visible(False)
        ax.set_xlabel(x, fontsize=18)
        ax.set_ylabel(measures[idx], fontsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=10, fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    path = path + experiment+'/'
    print(path)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print(files)
    df = pd.DataFrame()
    for file in files:
        if 'csv' in file:
            print('file', file)
            df = df.append(pd.read_csv(path + "/" + file))
    # df = replace_measure_labels(df)
    df = replace_solver_labels(df)
    df = df.rename(columns=relabel)
    compare_plot(df)


    # plot_pareto_compare([samples['Greedy'], samples['Uniform']],labels)


