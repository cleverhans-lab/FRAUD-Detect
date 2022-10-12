import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import matplotlib

#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'axes.labelsize': 8,
    'axes.titlesize': 12,
    'axes.titlepad': 2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 1,
    'axes.labelpad': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'legend.title_fontsize': 8,
    'legend.frameon': False,
    'lines.markersize': 1.3,
    'legend.markerscale': 2
})


def unfairness_range_plot(dataset, rseed, model_class, title, num):
    plt.clf()

    # filenames
    input_file = "./results_all/{}_{}_{}.csv".format(dataset, model_class, rseed)
    input_file_baseline = "unfairness_bbox/{}.csv".format(dataset)

    save_path = "../../sample_results/threshold_plots/{}_{}_{}_{}.png".format(dataset, model_class, rseed, title)
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)

    df = pd.read_csv(input_file)
    df_baseline = pd.read_csv(input_file_baseline)
    df_baseline = df_baseline[df_baseline['model_class'] == model_class]
    df_baseline = df_baseline[df_baseline['group'] == 'Members']
    df_baseline = df_baseline[df_baseline['metric'] == 'SP']

    # dropping lowest constraint
    df = df[df['kl_constr'] != .005]
    labels = np.unique(df['kl_constr'])

    fidelity_ogs = np.unique(df['fidelity_explainer'])
    fidelity = np.round(fidelity_ogs, 3)

    fig, ax = plt.subplots(1, 1)

    X = np.asarray(range(1, num * 2 + 1, 2))
    xpos = np.linspace(X - .6, X + .6, num=6)

    cm_subsection = np.linspace(0, .5, 7 + 1)
    cmap = matplotlib.cm.get_cmap('brg')
    colors = [cmap(x) for x in cm_subsection]
    marks = ['o', '^', 's', 'p', 'H', 'D', '*']

    # iterate through number recorded fidelities
    for i in range(len(X)):
        fidel = fidelity_ogs[i]
        group_data = df[df['fidelity_explainer'] == fidel]
        tops = group_data['max_disp'].to_numpy()
        bots = group_data['min_disp'].to_numpy()
        edges = np.array([bots, tops])
        avgs = np.mean(edges, axis=0)
        bots = np.abs(bots - avgs)
        tops = np.abs(tops - avgs)
        edges = np.array([bots, tops])

        # iterate through the number of thresholds
        for j in range(6):
            edge = edges[:, j]
            edge = [[edge[0]], [edge[1]]]
            if i == 1:
                plt.errorbar(xpos[j, i], avgs[j], yerr=edge, fmt='none', color=colors[j], linewidth=.7, capsize=.7)
                plt.plot(xpos[j, i], avgs[j], color=colors[j], marker=marks[j], label=labels[j])
            else:
                plt.errorbar(xpos[j, i], avgs[j], yerr=edge, fmt='none', color=colors[j], linewidth=.7, capsize=.7)
                plt.plot(xpos[j, i], avgs[j], color=colors[j], marker=marks[j], label='_nolegend_')

    plt.hlines(df_baseline['unfairness'], 0, 20, colors=['black'], linestyle='dashed', linewidth=.65)

    fig.set_size_inches(w=5.7511, h=4 / 15 * 5.7511)
    plt.xlabel('Fidelity')
    plt.ylabel('Unfairness')
    ax.legend(title='Threshold', bbox_to_anchor=(1, 1.0), loc='upper left', fontsize=8)
    fidelity = list(fidelity.astype(str))
    fidelity.insert(0, '')
    X = list(X)
    X.insert(0, 0)
    plt.xticks(X, fidelity)
    plt.title(title)
    plt.xlim((0, num * 2))

    plt.savefig(save_path, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    for dataset in ['marketing', 'compas', 'adult_income']:
        for model in ['AdaBoost', 'DNN', 'RF', 'XgBoost']:
            mapy = {'adult_income': 'Adult Income',
                    'compas': 'COMPAS',
                    'marketing': 'Marketing'}
            title = '{} - {}'.format(mapy[dataset], model)
            print(title)
            num = 0
            if dataset in ['compas', 'marketing'] and model == 'XgBoost':
                num = 9
            elif dataset == 'marketing' and model == 'RF':
                num = 8
            elif dataset == 'marketing' and model == 'AdaBoost':
                num = 7
            else:
                num = 10
            unfairness_range_plot(dataset, 0, model, title, num)
