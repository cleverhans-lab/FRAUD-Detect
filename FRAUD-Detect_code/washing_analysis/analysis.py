#!/bin/bash
import random
import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from fairlearn.metrics import *
from fairlearn.reductions import DemographicParity, FalsePositiveRateParity, \
    TruePositiveRateParity, EqualizedOdds
import functools
from config_fairlearn import *

from my_types import *
import numpy.typing as npt
from typing import Dict, Tuple, List, Union, Callable
from data_labels import *
from metrics import *
from plotting import *
import warnings
import sys
import multiprocessing

from joblib import Parallel, delayed


def append_row(epsilon_frames: Dict[str, pd.DataFrame], dataset: str, seed: int, use_test: bool, bb_model,
               metrics: List[Dict[str, Callable]], mit: str, exp: str, min_grp: str, epsilon: float, predictor: str,
               labeler: str, sensitive: List[str], thresholds: List[float], n: float) -> Dict[str, pd.DataFrame]:
    """
    Appends row at seed in epsilon frame for each fairness metric trained for containing all metrics passed, and some
    extra self defined ones.
    :param n:
    :type n:
    :param thresholds:
    :type thresholds:
    :param epsilon_frames:
    :param dataset:
    :param seed:
    :param use_test:
    :param bb_model:
    :param metrics:
    :param mit:
    :param exp:
    :param min_grp:
    :param epsilon:
    :param predictor:
    :param labeler:
    :param sensitive:
    :return:
    """
    # get predictions, data, and labels
    suing_X = get_datasets(dataset, seed, 'suing', exp)
    testing_X = get_datasets(dataset, seed, 'test', exp) if use_test else False
    data_X = testing_X if use_test else suing_X
    membership_data = get_memberships(data_X, sensitive)
    ground_truths = get_labels(dataset, seed, use_test)
    bb_preds = get_BB_preds(dataset, bb_model, seed, use_test)
    fair_metric_calls = [DemographicParity]  # , FalsePositiveRateParity]  #, TruePositiveRateParity, EqualizedOdds]
    for i in range(len(fair_metric_calls)):  # for each fairness metric
        epsilon_frames = get_row(fair_metric_calls, metrics, epsilon_frames, mit, exp, dataset, min_grp, epsilon, seed,
                                 suing_X,
                                 bb_model,
                                 use_test, predictor, bb_preds, labeler, ground_truths, membership_data, sensitive,
                                 data_X, n,
                                 thresholds, i)
    return epsilon_frames


def get_row(fair_metric_calls, metrics, epsilon_frames, mit, exp, dataset, min_grp, epsilon, seed, suing_X, bb_model,
            use_test, predictor, bb_preds, labeler, ground_truths, membership_data, sensitive, data_X, n, thresholds,
            i):
    optim_metric = fair_metric_calls[i]  # callable
    metric_name = list(metrics[1].keys())[i]  # metric name
    epsilon_frame = epsilon_frames[metric_name]
    im_preds, im_probas = get_IM_predictions(mit, exp, dataset, min_grp, epsilon, seed, suing_X, bb_model,
                                             optim_metric, use_test, save=False)
    # decide which model is predictor and labeler
    preds = im_preds if predictor == 'IM' else bb_preds
    trues = ground_truths if labeler == 'GT' else bb_preds

    # get sklearn metrics, fair metrics, and our metrics
    epsilon_frame = get_sklearn_metrics(epsilon_frame, metrics, trues, preds, membership_data, seed, sensitive)
    # For fairlearn normal metrics:
    # epsilon_frame = get_fair_metric_score(epsilon_frame, optim_metric, metric_name, trues, preds, membership_data, seed)
    # For older metrics
    epsilon_frame = get_old_unf(epsilon_frame, metric_name, trues.values, preds, data_X, sensitive, seed)

    # get our metrics
    our_metrics = list(metrics[2].keys())
    for our_metric_name in our_metrics:
        callable = metrics[2][our_metric_name]
        if callable is None:
            break
        args = None
        if our_metric_name != 'ttest' and our_metric_name != 'ttest_stat':
            if our_metric_name == 'kl_ce':
                args = [epsilon_frame.loc[seed, (sensitive[0], 'log_loss')],
                        epsilon_frame.loc[seed, (sensitive[1], 'log_loss')]]
            elif our_metric_name == 'kl_conf_matrix':
                args = [epsilon_frame.loc[seed, (sensitive[0], 'conf_matrix')],
                        epsilon_frame.loc[seed, (sensitive[1], 'conf_matrix')]]
            elif our_metric_name == 'kl_tp_fp':
                args = [epsilon_frame.loc[seed, (sensitive[0], 'conf_matrix')],
                        epsilon_frame.loc[seed, (sensitive[1], 'conf_matrix')]]
            elif our_metric_name == 'kl_ce_samples':
                # integer value is nbins
                args = [im_probas, trues, membership_data, True, 50, sensitive, seed, dataset, epsilon,
                        optim_metric.__name__, mit, exp, bb_model]
            elif our_metric_name == 'kl_ce_ad_lab':
                args = [im_probas, preds, trues, membership_data, sensitive, 50, True]
            elif our_metric_name == 'wasserstein_dist':
                args = [epsilon_frame.loc[seed, (sensitive[0], 'conf_matrix')],
                        epsilon_frame.loc[seed, (sensitive[1], 'conf_matrix')]]
            data = callable(*args)
            epsilon_frame.at[seed, ('Across groups', our_metric_name)] = data
        else:  # ttest is more complex ...
            divisor, trues_cpy, preds_cpy, membership_data_cpy = ttest_set_data(preds, trues, membership_data)
            args = [preds_cpy, trues_cpy.to_numpy(), membership_data_cpy.to_numpy(), True, 100, divisor]
            p_vals, stat = ttest(*args)
            epsilon_frame.at[seed, ('Across groups', 'ttest')] = p_vals
            epsilon_frame.at[seed, ('Across groups', 'ttest_stat')] = stat

    # now get detector and ground truth info
    detectors = list(metrics[3].keys())
    for detector_name in detectors:
        callable = metrics[3][detector_name]
        args = []
        if detector_name == 'GT_washed':
            args = [n, metric_name, data_X, sensitive, ground_truths.values, bb_preds, im_preds]
        elif detector_name == 'threshold_detector':
            value = epsilon_frame.at[seed, ('Across groups', metric_name)]
            args = [thresholds, value]
        elif detector_name == 'bb_unfairness':
            args = [data_X, sensitive, bb_preds, ground_truths.values, metric_name]
        data = callable(*args)
        epsilon_frame.at[seed, ('Detectors', detector_name)] = data

    epsilon_frames[metric_name] = epsilon_frame
    return epsilon_frames


#	def get_log_losses(probas: List[tuple], trues: npt.ArrayLike, membership_data: npt.ArrayLike, groups: list) \
#         -> Dict[str, float]:
#     """
#     Get log losses / cross entropy losses for subgroups and total model performance
#     :param probas:
#     :param trues:
#     :param membership_data:
#     :param groups:
#     :return:
#     """
#     # load up testing set
#     loglosses = dict.fromkeys(groups)
#     for group in groups:
#         if group != 'Across groups':
#             indexes = np.where(membership_data == group)[0]
#             indexes = indexes.tolist()
#             trues_group = trues[indexes]
#             probas_group = probas[indexes]
#             loglosses[group] = skm.log_loss(trues_group, probas_group)
#         else:
#             loglosses[group] = skm.log_loss(trues, probas)
#     return loglosses


def average_result_frames(cv_frames: Dict[str, pd.DataFrame], epsilon_frames: Dict[str, pd.DataFrame],
                          epsilon: float) -> Dict[str, pd.DataFrame]:
    """
    Average rows (seed) in epsilon_frames and store the averaged result in cv_frames at epsilon for each metric
    :param cv_frames:
    :param epsilon_frames:
    :param epsilon:
    :return:
    """
    # coalesce results from epsilon_frames into cv_frames
    for key in list(cv_frames.keys()):
        frame = epsilon_frames[key]
        for (upper, col) in list(cv_frames[key].columns):
            data = np.asarray(frame.loc[:, (upper, col)])
            meaned = np.mean(data, axis=0)

            cv_frames[key].at[epsilon, (upper, col)] = meaned
    return cv_frames


def generate_model_frames(seeds: list, epsilons: list, predictor: str, labeler: str, mitigator: str,
                          explainer: str, bb_model, dataset: str, sensitive: list, metrics: List[Dict[str, Callable]],
                          use_test: bool, thresholds: List[float], n: float) -> Tuple[
    Dict[str, pd.DataFrame], dict, str]:
    """
    Given ranges of seed and epsilons, get dataframe for each fairness metric of cross-validated results of metrics on
    test or suing set (IM training set). Returns a dictionary where keys are the metrics used to train the IM in each
    and values are the results dataframe
    :param seeds:
    :param epsilons:
    :param predictor:
    :param labeler:
    :param mitigator:
    :param explainer:
    :param bb_model:
    :param dataset:
    :param sensitive:
    :param metrics:
    :param use_test:    If true, tests metrics on testing set, if not, then on suing set
    :return:"""
    string = ''
    # make cv_frame, empty dataframe to hold cv rows
    min_grp = sensitive[1]
    columns = make_index(metrics, sensitive)
    key_metrics = list(metrics[1].keys())  # fairness metrics
    # cv frame holds measurements per epsilon
    # cv frames holds cv frames for each fit metric
    cv_frame = pd.DataFrame(index=epsilons, columns=columns)
    # cv_frame.drop('Detectors', axis=1, inplace=True)
    cv_frames = {}
    for key in list(metrics[1].keys()):
        cv_frames[key] = cv_frame.copy()
    to_seaborn = {key: [] for key in key_metrics}
    detector = {key: [] for key in key_metrics}

    for epsilon in epsilons:
        print('Computing epsilon: ' + str(epsilon))
        epsilon_frame = pd.DataFrame(index=seeds, columns=columns)
        epsilon_frames = {}
        for key in key_metrics:
            epsilon_frames[key] = epsilon_frame.copy()

        # this loop shall be parallel!!!
        jobs = multiprocessing.cpu_count() - 4
        epsilon_frames = Parallel(n_jobs=jobs)(
            delayed(append_row)(epsilon_frames, dataset, seed, use_test, bb_model, metrics, mitigator,
                                explainer, min_grp, epsilon, predictor, labeler, sensitive, thresholds, n)
            for seed in seeds
        )
        # now concatenate rows!
        fair_keys = list(metrics[1].keys())
        for key in fair_keys:
            first_frame = epsilon_frames[0][key]
            for i in range(len(first_frame)):
                first_frame.iloc[[i]] = epsilon_frames[i][key].iloc[[i]]
        epsilon_frames = epsilon_frames[0]

        # for seed in seeds:  # add seed row each time
        #     print('seed: ' + str(seed))
        #     epsilon_frames = append_row(epsilon_frames, dataset, seed, use_test, bb_model, metrics, mitigator,
        #                                 explainer, min_grp, epsilon, predictor, labeler, sensitive, thresholds, n)
        # # for each set of results (coresponding to results made under different IM train conditions)
        # # add data to seaborn stuff
        # print(epsilon_frames)

        for key in key_metrics:
            preds, trues = _threshold_detector_preds(epsilon_frames[key])
            detector[key].append([trues, preds])
            # result_frames dont have detector stuff
            # epsilon_frames[key] = epsilon_frames[key].drop('Detectors', axis=1)
            for (big, small) in cv_frames[key].columns:
                if big == 'Detectors':
                    cv_frames[key] = cv_frames[key].drop((big, small), axis=1)
            frame = epsilon_frames[key].copy()
            frame['epsilon'] = [epsilon] * len(frame)
            to_seaborn[key].append(frame)

        cv_frames = average_result_frames(cv_frames, epsilon_frames, epsilon)

    for key in key_metrics:
        frames = pd.concat(to_seaborn[key], ignore_index=True)
        frames.index.rename('seed')
        to_seaborn[key] = frames
        detector_metrics = {
            'accuracy': skm.accuracy_score,
            'precision': skm.precision_score,
            'recall': skm.recall_score,
            'log_loss': skm.log_loss,
            'confusion_matrix': skm.confusion_matrix,
            'f1': skm.f1_score
        }
        string += ('\n\n' + key)
        string += detector_analysis(detector[key], detector_metrics)

    return cv_frames, to_seaborn, string


def _threshold_detector_preds(frame):
    '''
    For entire epsilon-frame, get the ground truth labels, and a list of predictions made by each detector threshold
    (stored in a Dict[float, list])
    Each epsilon gives us
    :param frame:
    :return:
    '''
    gt_labels = list(frame[('Detectors', 'GT_washed')])
    threshold_preds = defaultdict(list)
    thresh_series = frame[('Detectors', 'threshold_detector')]
    for index, model_preds in thresh_series.items():
        for threshold in list(model_preds.keys()):
            threshold_preds[threshold].append(model_preds[threshold])
    return threshold_preds, gt_labels


# run for each metric
def detector_analysis(detector, detector_metrics):
    preds = defaultdict(list)
    trues = []
    for model in detector:
        trues.extend(model[0])
        pred_dict = model[1]
        for threshold in list(pred_dict.keys()):
            preds[threshold].extend(pred_dict[threshold])
    string = ''

    for threshold in preds.keys():
        string += '\nThreshold {}\n'.format(threshold)
        predictions = preds[threshold]
        for metric in list(detector_metrics.keys()):
            callable = detector_metrics[metric]
            if metric == 'accuracy':
                value = (callable(trues, predictions))
            else:
                value = callable(trues, predictions, labels=[True, False])
            string += '\t{}: {}'.format(metric, value)
    return string


def get_memberships(data: pd.DataFrame, sensitive: list) -> npt.ArrayLike:
    """
    Undos data one-hot encodings for sensitive feature and returns the grop memberships
    :param data:
    :param sensitive:
    :return:
    """
    return data[sensitive].idxmax(axis=1)


def make_index(metrics: List[Dict[str, Callable]], sensitive: list) -> pd.MultiIndex:
    """
    Build indexes for cv_frames and epsilon frames
    :param metrics:
    :param sensitive:
    :return:
    """
    # get group names, put those as highest level
    skm_metrics = list(metrics[0].keys())
    fair_metrics = list(metrics[1].keys())
    our_metrics = list(metrics[2].keys())
    detectors = list(metrics[3].keys())
    highest = [sensitive[0]] * len(skm_metrics)
    highest.extend([sensitive[1]] * len(skm_metrics))
    highest.extend(['Across groups'] * (len(skm_metrics) + len(fair_metrics) + len(our_metrics)))
    highest.extend(['Detectors'] * len(detectors))
    second = skm_metrics.copy()  # min
    second.extend(skm_metrics)  # maj
    second.extend(skm_metrics)  # overall
    second.extend(fair_metrics)  # fair
    second.extend(our_metrics)  # our metrics
    second.extend(detectors)  # detectors
    tuples = list(zip(highest, second))
    index = pd.MultiIndex.from_tuples(tuples, names=['Group', 'Metric'])
    return index


def weighted_score(dataset, algorithm):
    # method from harvineet here https://github.com/fairlearn/fairlearn/issues/65
    # algorithm = mitigator (esp. ExponentiatedGradient)
    pred_all = np.zeros((dataset.shape[0], len(algorithm._hs)))
    for i, clf in enumerate(algorithm.weights_):
        pred_all[:, i] = clf.predict_proba(dataset)[:, 1]
    score = pred_all.dot(algorithm._expgrad_result._weights)
    return score


def old_unf_metrics():
    metrics = {
        'dp': Metric.statistical_parity}  # ,  # ,
    #    'fpr': Metric.predictive_equality}   # FPP
    #    'tpr': Metric.equal_opportunity,    # TPP
    #    'eodds': Metric.equalized_odds
    # }
    return metrics


def get_metrics(dataset: str) -> List[Dict[str, Callable]]:
    """
    Construct metrics wanting to test, alter this for different results. Returns list of dicts from str (metric name) to
    Callable. The first elem is all sklearn metrics, second fairlearn metircs, third are our mterics.
    :param dataset:
    :param mitigator:
    :return:
    """
    # want performance metrics for both groups, and in general, want fairness metrics across groups, our metrics across
    # groups as well
    labels = get_possible_labels(dataset)
    precision_alt = functools.partial(skm.precision_score, labels=labels, zero_division=0)
    recall_alt = functools.partial(skm.recall_score, labels=labels, zero_division=0)
    log_loss_alt = functools.partial(skm.log_loss, labels=labels)
    cm_alt = functools.partial(skm.confusion_matrix, labels=labels,
                               normalize=None)  # normalize='true')  # sets nomlaization for KL divs
    performance_metrics = {
        'count': count,
        'accuracy': skm.accuracy_score,
       # 'precision': precision_alt,
       # 'recall': recall_alt,
        'conf_matrix': cm_alt#,
      #  'log_loss': log_loss_alt
    }

    fair_metrics = old_unf_metrics()
    our_metrics = {
        'kl_conf_matrix': kl_conf_matrix,
        'kl_tp_fp': kl_tp_fp#,
    #    'kl_ce': kl_ce,
    #    'wasserstein_dist': wasserstein_dist,
    #    'kl_ce_samples': kl_ce_samples,
   #     'kl_ce_ad_lab': kl_ce_ad_lab,
   #     'ttest': ttest,
   #     'ttest_stat': None
    }
    detectors = {
        'bb_unfairness': gt_washed_unfairness,
        'GT_washed': GT_washed,
        'threshold_detector': threshold_detector
    }
    metrics = [performance_metrics, fair_metrics, our_metrics, detectors]
    return metrics


def main(run, dataset, epsilons, title):
    np.set_printoptions(threshold=sys.maxsize)
    print('Epsilons:' + str(epsilons))
    dataset, _, maj_grp, min_grp = get_data(dataset)
    # more_eps = np.arange(.701, 1.0, .001)
    # epsilons = np.around(epsilons + list(more_eps), 3)  # fix any weird errors here
    #epsilons = [0.0, .15, .2, .3, .5, .8, .9, .92, .95, .96, .97, .99, .999, .9995, 1.0]
    seeds = list(range(10))
    use_test = False

    [mit, exp, bb_model] = run

    thresholds = [.005, .01, .03, .05, .1, .15, .2, .5, 1, 1.5, 2, 3]
    n = 5  # in percentages

    sensitive = [maj_grp, min_grp]

    metrics = get_metrics(dataset)

    labeler = 'BB'
    predictor = 'IM'

    # Make results dataframes
    results_frames, to_seaborn, string = generate_model_frames(seeds, epsilons, predictor, labeler, mit, exp,
                                                               bb_model, dataset,
                                                               sensitive, metrics, use_test, thresholds, n)
    frame = to_seaborn['dp']
    frame = add_seeds(frame)
    # print(frame)
    frame = compute_front(frame, 'dp', 'accuracy')
    # sb_metric_v_unf(frame, 'dp', 'kl_conf_matrix', title,
    #                 '../../sample_results/ckl_plots/{}_paretoed_{}_{}_{}.png'.format(title, dataset, bb_model, exp))
    seed_metric_v_unf(frame, 'dp', 'kl_conf_matrix',
                      '../../sample_results/ckl_plots/{}_paretoed_{}_{}_{}.png'.format(title, dataset, bb_model, exp),
                      title, True, [0, 1], [0, 1])

    save_to = '../../sample_results/ckl_plots/sb_{}_paretoed_{}_{}_{}.png'.format(title, dataset, bb_model, exp)
    main_plots(frame, 'dp', 'epsilon', ['kl_conf_matrix', 'dp'], bb_model, exp, dataset, save_to)

    print('Saved plot here: ../../sample_results/ckl_plots/{}_paretoed_{}_{}_{}.png'.format(title, dataset, bb_model,
                                                                                            exp))
    print('Also saved plot here: ' + save_to)


if __name__ == '__main__':
    # epsilons = [0.0, 0.05, .1, .3, .4, .5, .6, .7]
    # more_eps = np.arange(.701, 1.0, .001)
    # epsilons = np.around(epsilons + list(more_eps), 3)
    epsilons = [0.0, .15, .2, .3, .5, .8, .9, .92, .95, .96, .97, .99, .999, .9995, 1.0]
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=int, help='dataset number, 1=adult_income, 2=compas, 3=Marketing')
    parser.add_argument('im', type=str, help='dt=Decision tree, lm=logistic regression')
    parser.add_argument('bb', type=str, help='DNN=Deep Neural Network, AdaBoost, RF=Random Forest, XgBoost')
    parser.add_argument('-title', type=str, default='', help='Figure title (saved in sample_results)')
    parser.add_argument('-epsilons', type=float, nargs='*',
                        default=epsilons,
                        help='Pass in epsilons between 0.0 and 1.0 in space-separated list. 308 epsilons default.')
    args = parser.parse_args()

    # fix dataset numbering (marketing is actually dataset 4)
    if args.dataset == 3:
        dataset = 4
    else:
        dataset = args.dataset

    run = ['eg', args.im, args.bb]
 
    # epsilons = np.arange((0, 1.0, .01))
    main(run, dataset, args.epsilons, args.title)
    # main(['eg', 'lm', 'AdaBoost'], 1, [0.0, .15, .99, 1.0], 'test_ish')

# ex:     python analysis.py 1 dt DNN -epsilons 0.0 0.5 .999
