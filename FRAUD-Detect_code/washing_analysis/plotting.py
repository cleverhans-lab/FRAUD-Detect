import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from metrics import *
import seaborn as sb
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple, Union
from data_labels import read_results_frames

import warnings

warnings.filterwarnings("ignore")


def conf_matrix_histogram(result_frame: pd.DataFrame, epsilons: Union[None, list], save_path: str) -> str:
    """
    Creates histogram of conf_matrix values over groups
    :param result_frame:            Given SEABORN-stlye frame
    :param epsilons:
    :param save_path:
    :return:
    """
    # get subgroup names
    groups = list(result_frame.columns.levels[0])
    # get names of groups to consider
    labels = []
    min_maj = []
    matrix_labels = ['tn', 'fp', 'fn', 'tp']
    for group_name in groups:
        if '_' in group_name:
            min_maj.append(group_name)
            sub_name = group_name.split('_')[1]
            for mat_lab in matrix_labels:
                labels.append('{} {}'.format(sub_name, mat_lab))
    labels.insert(4, 'F TP+FP/Fpop')
    labels.insert(9, 'M TP+FP/Mpop')
    labels.append('diff')
    epsilons = list(result_frame['epsilon'].unique())
    epsilons.sort()
    x = 2.5 * np.arange(len(labels))
    shifts = np.arange(-.5, .5, 1.0 / len(epsilons))
    width = 1.0 / len(epsilons)  # bar width
    fig, ax = plt.subplots()

    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        # now have multiple seeds going on
        relevant_eps = result_frame.loc[result_frame['epsilon'] == epsilon]
        # get group confusion matricies
        conf_matrix_one = np.asarray(relevant_eps[(min_maj[0], 'conf_matrix')])
        conf_matrix_two = relevant_eps[(min_maj[1], 'conf_matrix')]
        # average matricies (and kl_conf_matri values) across seeds (we can do fancy error bars later on)
        conf_matrix_one = np.mean(conf_matrix_one, axis=0).ravel()
        conf_matrix_two = np.mean(conf_matrix_two, axis=0).ravel()
        ### norm each matrix by the number of samples in each group
        pop_one = np.sum(conf_matrix_one)
        # print(pop_one)
        # print(conf_matrix_one)
        pop_two = np.sum(conf_matrix_two)
        conf_matrix_one = list(1.0/pop_one * conf_matrix_one)
        # print(conf_matrix_one)
        conf_matrix_two = list(1.0/pop_two * conf_matrix_two)
        # (TP+FP)/group pop
        special_one = (conf_matrix_one[1] + conf_matrix_one[3])
        # print(conf_matrix_one[0] + conf_matrix_one[1])
        special_two = (conf_matrix_two[1] + conf_matrix_two[3])
        conf_matrix_one.append(special_one)
        conf_matrix_two.append(special_two)
        diff = abs(special_two - special_one)
        conf_matrix_two.append(diff)
        kl = np.asarray(relevant_eps[('Across groups', 'kl_conf_matrix')])
        kl = np.mean(kl, axis=0)
        data = np.concatenate((conf_matrix_one, conf_matrix_two), axis=0)
        legend_label = str(epsilon) + ' KL:' + str(round(kl, 5))
        ax.bar(x + shifts[i], data, width, label=legend_label)
    # beauty stuff
    ax.set_ylabel('Number of samples')
    ax.set_xlabel('Subgroups')
    ax.set_title('Subgroup Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    return save_path


def plot_ttest_histogram(result_frame: pd.DataFrame, epsilons: Union[None, list], save_path: str) -> str:
    """
    Plots ttest histogram of p_values across group conf_matrix values
    :param result_frame:
    :param epsilons:
    :param save_path:
    :return:
    """
    # get subgroup names
    if epsilons is None:
        epsilons = list(result_frame.index)
    labels = ['tp', 'fp', 'fn', 'tn']
    x = 2.5 * np.arange(len(labels))
    shifts = np.arange(-.5, .5, 1.0 / len(epsilons))
    width = .2  # bar width
    fig, ax = plt.subplots()

    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        ttest = result_frame.loc[epsilon, ('Across groups', 'ttest_stat')].ravel()
        # kl = result_frame.loc[epsilon, ('Across groups', 'KL_div')]
        # legend_label = str(epsilon) + ' KL:' + str(round(kl, 5))
        # print(x+shifts[i])
        # print(ttest)
        ax.bar(x + shifts[i], ttest, width)  # , label=legend_label)
    ax.set_ylabel('Scores')
    ax.set_xlabel('T-Test p-values')
    ax.set_title('T-test across subgroups')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    return save_path


def plot_over_epsilons(results_frame: pd.DataFrame, metrics: list, save_path: str) -> str:
    """
    Plot metrics (list) over epsilons to track change at higher fairness constraints.
    :param results_frame:
    :param metrics:
    :param save_path:
    :return:
    """
    figs, axs = plt.subplots(len(metrics), 1)
    epsilons = list(results_frame.index)
    for i in range(len(metrics)):
        metric = metrics[i]
        data = list(results_frame.loc[:, ('Across groups', metric)].values)
        color = 'green'
        marker = 'o'
        # plot data point markers
        for j in range(len(data)):
            axs[i].plot(epsilons[j], data[j], color=color, marker=marker)
        axs[i].plot(epsilons, data)
        axs[i].set(ylabel=metrics[i])
    # lg = plt.legend(legend, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    # figs.tight_layout()
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=1)
    plt.clf()
    return save_path


def sb_metric_v_unf(result_frame: pd.DataFrame, x: str, y: str, title: str, save_path: str) -> str:
    """
    Plots passed metric over unfairness for each seed. Also shows the bb unfairness
    :param title:           Title for plot
    :param result_frame:    A dataframe of results
    :param x:               Column in frame to use as x-axis
    :param y:               A metric to plot over whatever, must be column in Across groups
    :param save_path:       Str to save to
    :return:
    """
    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    epsilons = result_frame["epsilon"]
    # hue given by seed, line through epsilon values
    num_seeds = max(result_frame['seed'])
    cm_subsection = np.linspace(0, 1, num_seeds+1)
    colors = [plt.cm.tab10(x) for x in cm_subsection]
    for seed in range(num_seeds+1):
        rows = result_frame.loc[result_frame['seed'] == seed]
        bb_unf = rows[('Detectors', 'bb_unfairness')]
        xx = rows[('Across groups', x)]
        yy = rows[('Across groups', y)]
        # make sure we're plotting in x-axis order
        data = np.stack((xx, yy)).T
        data[:, 0] = 1 - data[:, 0]  # get fairness from unfairness values
        data = data[np.argsort(data[:, 0])]
        if len(data) > 1:
            plt.plot(data[:, 0], data[:, 1], color=colors[seed])
        else:
            plt.plot(data[:, 0], data[:, 1], color=colors[seed], marker='o')
        if len(bb_unf) > 0:
            unf = bb_unf.unique()
            unf = 1 - unf  # get bb fairness
            plt.axvline(x=unf, color=colors[seed], linestyle='dashed')
    plt.title(title)
    plt.ylabel('KL Confusion Matrix')
    plt.xlabel('Demographic Parity Fairness')
    plt.ylim((0, 1.0))
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path)
    return save_path


def my_formatter(x, pos):
    # https://stackoverflow.com/questions/8555652/removing-leading-0-from-matplotlib-tick-label-formatting/8555658
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    x = np.round(x, 2)
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def seed_metric_v_unf(result_frame: pd.DataFrame, x: str, y: str,  save_path: str, title, have_y, ylimy, xlimy) -> str:
    """
    Plots passed metric over unfairness for each seed. Also shows the bb unfairness
    :param result_frame:    A dataframe of results
    :param x:               Column in frame to use as x-axis
    :param y:               A metric to plot over whatever, must be column in Across groups
    :param save_path:       Str to save to
    :return:
    """
    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    major_formatter = FuncFormatter(my_formatter)
    ax = plt.gca()
    epsilons = result_frame["epsilon"]
    # hue given by seed, line through epsilon values
    num_seeds = max(result_frame['seed'])
    cm_subsection = np.linspace(0, 1, num_seeds+1)
    colors = [plt.cm.tab10(x) for x in cm_subsection]
    for seed in range(num_seeds+1):
        rows = result_frame.loc[result_frame['seed'] == seed]
        bb_unf = rows[('Detectors', 'bb_unfairness')]
        xx = rows[('Across groups', x)]
        yy = rows[('Across groups', y)]
        # yy = rows[y]
        # make sure we're plotting in x-axis order
        data = np.stack((xx, yy)).T
        data[:, 0] = 1-data[:, 0]  # get fairness from unfairness values
        data = data[np.argsort(data[:, 0])]
        if len(data) > 1:

            plt.plot(data[:, 0], data[:, 1], color=colors[seed], linewidth=.5)
            plt.plot()
        else:
            plt.plot(data[:, 0], data[:, 1], color=colors[seed], marker='o')
        if len(bb_unf) > 0:
            unf = bb_unf.unique()
            unf = 1-unf  # get bb fairness
            plt.axvline(x=unf, color=colors[seed], linestyle='dashed', linewidth=.5)
    plt.title(title)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    if have_y:
        plt.ylabel('$\mathcal{C}_{KL}$')
    plt.xlabel('Fairness')
    # plt.ylim((.1, .5))
    # plt.xlim((.8, 1.0))
    plt.ylim(ylimy)
    plt.xlim(xlimy)
    #plt.ylabel(y)
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    #plt.tight_layout()

    # fig.set_size_inches(w=1.4513, h=1.088)  # adult and compas
    #fig.set_size_inches(w=1.39098, h=1.088)
    plt.margins(0, 0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    return save_path


def sb_over_epsilons(results_frame: pd.DataFrame, metrics: list, save_path: str) -> str:
    """
    Plots over epsilons with error bars from seaborn.
    :param results_frame:   A dataframe for results
    :param metrics:         List of metrics ot plot over epsilons, must be column in Across groups
    :param save_path:       Str to save to
    :return:
    """
    # if 'kl_ce_ad_lab' in metrics:
    #     results_frame, ad_labels = adjust_kl_ce_ad_lab_frame(results_frame)
    #     metrics.extend(ad_labels)
    #     metrics.remove('kl_ce_ad_lab')
    figs, axs = plt.subplots(len(metrics), 1)  # , sharey='col')
    figs.suptitle('Metrics over Epsilons')
    frame = results_frame['Across groups']
    frame['epsilon'] = results_frame['epsilon']
    for i in range(len(metrics)):
        metric = metrics[i]
        frame = frame.astype({'epsilon': float, metric: float})  # make sure types are all good
        sb.lineplot(ax=axs[i], data=frame, x="epsilon", y=metric, #hue=frame["epsilon"].index.to_list(),
                    ci='sd', estimator='mean')
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    # plt.figure(figsize=(10,10))
    plt.gcf().set_size_inches(8.3, 18.7)
    plt.subplots_adjust(hspace=.4)
    figs.savefig(save_path) #, bbox_inches='tight')
    return save_path


def adjust_kl_ce_ad_lab_frame(frame):
    # expand out agree/disagree stuff
    data = np.asarray(frame['Across groups']['kl_ce_ad_lab'])
    data = np.stack(data, axis=0)
    # labels = ['Agree 0', 'Disagree 0', 'Agree 1', 'Disagree 1']
    labels = ['TN', 'FP', 'TP', 'FN']
    for i in range(len(labels)):
        frame[('Across groups', labels[i])] = data[:, i]
    return frame, labels


def read_unfairness_frames(folder: str, use_test: bool, metric: str) -> Tuple[Dict[str, pd.Series], Union[None, list]]:
    """
    Read in unfainresses of metric from results frames for plot_unfairness
    :param folder:
    :param use_test:
    :param metric:
    :return:
    """
    to_gen_models = [['IM', 'BB'], ['BB', 'GT'], ['IM', 'GT']]
    labels = ['_vs_'.join(models) for models in to_gen_models]
    unfairness_dict = dict.fromkeys(labels)
    index = None
    for label in labels:
        folder = str(os.path.join(folder, label)) + '/'
        if use_test:
            folder = str(os.path.join(folder, label)) + '_testset/'
        frame = read_results_frames(os.path.join(folder, metric), None)
        index = list(frame.index)
        unfairness_dict[label] = list(frame.loc[:, ('Across groups', metric)])
    return unfairness_dict, index


def plot_unfairness(use_test: bool, metric: str, epsilons: list, folder: str, save_path: str) -> str:
    """
    Plots unfainress over epsilons
    :param use_test:
    :param metric:
    :param epsilons:
    :param folder:
    :param save_path:
    :return:
    """
    unf, index = read_unfairness_frames(folder, use_test, metric)
    if epsilons is None:
        epsilons = index
    keys = list(unf.keys())
    for key in keys:
        plt.plot(epsilons, unf[key], label=key)
    plt.title('Unfairnesses over epsilons')
    plt.legend()
    plt.xlabel('Epsilons')
    plt.ylabel('Unfairness value')
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    return save_path


def main_plots(df: pd.DataFrame, fair: str, x: str, y: List[str], bb_model, exp, dataset, save_to):
    # df.to_pickle(os.path.join(sb_folder, 'dp_results_with_seeds.pkl'))
    # df = pd.read_pickle(os.path.join(sb_folder, 'dp_results_with_seeds.pkl'))

    fig, axs = plt.subplots()
    g= axs.twinx()

    ## CLEAN DATA
    plot_df = df[
        [('Across groups', fair), ('Across groups', y[0]), ('epsilon', ''), ('seed', '')]].replace(
        [np.inf, -np.inf], np.nan).droplevel(level=0, axis=1)
    plot_df.columns = plot_df.columns[:2].tolist() + ['epsilon', 'seed']
    plot_df.sort_values(by=['seed', x], inplace=True)

    ## PLOT PER DATASET
    from matplotlib import cm

    cmap = cm.get_cmap('Set1')
    # G IS LEFT
    g = sb.lineplot(data=plot_df[[x, y[0], 'seed']].sort_values(by=['seed', x]), x=x, y=y[0], estimator='mean',
                    ci=95, color=cmap(0), ax=g)

    sb.lineplot(data=plot_df[[x, y[1], 'seed']].sort_values(by=['seed', x]), x=x, y=y[1], estimator='mean', ci=95,
                color=cmap(1), ax=axs)

    axs.spines["left"].set_color(cmap(0))
    axs.spines["left"].set_linewidth(2)

    axs.spines["right"].set_color(cmap(1))
    axs.spines["right"].set_linewidth(2)

    # ax.set_ylim(0, ymax_dict[dataset])
    # ax.set_xticks(np.arange(0, 1.1, step=.2))

    g.set_xlabel(r"$1-\epsilon$", fontsize=12)
    axs.set_ylabel(r"$\mathcal{C}_{KL}$", fontsize=12)
    axs.set_xlabel(r"$1-\epsilon$", fontsize=12)

    g.set_ylabel(r"Parity gap of $I(\cdot)$", fontsize=12)


    # ax.tick_params(axis='y', colors=cmap(1))
    # g.tick_params(axis='y', colors=cmap(0))

    g.set_title("{} and {}".format(bb_model, exp.upper() if exp != "lm" else "LR"))
    plt.savefig(save_to)
