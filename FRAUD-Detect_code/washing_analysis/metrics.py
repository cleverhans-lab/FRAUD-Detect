from typing import Dict, Tuple, List, Union, Callable
from collections.abc import Iterable
from collections import defaultdict, namedtuple
import random
from six.moves import xrange
import os

import pandas as pd
import numpy as np
import numpy.typing as npt
from fairlearn.metrics import *
from scipy.stats import entropy, wasserstein_distance
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats import multitest
import torch
from torch.nn.functional import cross_entropy


class ConfusionMatrix(namedtuple('ConfusionMatrix', 'minority majority label truth')):
    def get_matrix(self):
        TP = np.logical_and(self.label == 1, self.truth == 1)
        FP = np.logical_and(self.label == 1, self.truth == 0)
        FN = np.logical_and(self.label == 0, self.truth == 1)
        TN = np.logical_and(self.label == 0, self.truth == 0)

        # maj
        TP_maj = np.logical_and(TP == 1, self.majority == 1)
        FP_maj = np.logical_and(FP == 1, self.majority == 1)
        FN_maj = np.logical_and(FN == 1, self.majority == 1)
        TN_maj = np.logical_and(TN == 1, self.majority == 1)

        nTP_maj = np.sum(TP_maj)
        nFN_maj = np.sum(FN_maj)
        nFP_maj = np.sum(FP_maj)
        nTN_maj = np.sum(TN_maj)

        nPPV_maj = float(nTP_maj) / max((nTP_maj + nFP_maj), 1)
        nTPR_maj = float(nTP_maj) / max((nTP_maj + nFN_maj), 1)

        nFDR_maj = float(nFP_maj) / max((nFP_maj + nTP_maj), 1)
        nFPR_maj = float(nFP_maj) / max((nFP_maj + nTN_maj), 1)

        nFOR_maj = float(nFN_maj) / max((nFN_maj + nTN_maj), 1)
        nFNR_maj = float(nFN_maj) / max((nFN_maj + nTP_maj), 1)

        nNPV_maj = float(nTN_maj) / max((nTN_maj + nFN_maj), 1)
        nTNR_maj = float(nTN_maj) / max((nTN_maj + nFP_maj), 1)

        # min
        TP_min = np.logical_and(TP == 1, self.minority == 1)
        FP_min = np.logical_and(FP == 1, self.minority == 1)
        FN_min = np.logical_and(FN == 1, self.minority == 1)
        TN_min = np.logical_and(TN == 1, self.minority == 1)

        nTP_min = np.sum(TP_min)
        nFN_min = np.sum(FN_min)
        nFP_min = np.sum(FP_min)
        nTN_min = np.sum(TN_min)

        nPPV_min = float(nTP_min) / max((nTP_min + nFP_min), 1)
        nTPR_min = float(nTP_min) / max((nTP_min + nFN_min), 1)

        nFDR_min = float(nFP_min) / max((nFP_min + nTP_min), 1)
        nFPR_min = float(nFP_min) / max((nFP_min + nTN_min), 1)

        nFOR_min = float(nFN_min) / max((nFN_min + nTN_min), 1)
        nFNR_min = float(nFN_min) / max((nFN_min + nTP_min), 1)

        nNPV_min = float(nTN_min) / max((nTN_min + nFN_min), 1)
        nTNR_min = float(nTN_min) / max((nTN_min + nFP_min), 1)

        matrix_maj = {
            'TP' : nTP_maj,
            'FP' : nFP_maj,
            'FN' : nFN_maj,
            'TN' : nTN_maj,
            'PPV' : nPPV_maj,
            'TPR' : nTPR_maj,
            'FDR' : nFDR_maj,
            'FPR' : nFPR_maj,
            'FOR' : nFOR_maj,
            'FNR' : nFNR_maj,
            'NPV' : nNPV_maj,
            'TNR' : nTNR_maj}

        matrix_min = {
            'TP' : nTP_min,
            'FP' : nFP_min,
            'FN' : nFN_min,
            'TN' : nTN_min,
            'PPV' : nPPV_min,
            'TPR' : nTPR_min,
            'FDR' : nFDR_min,
            'FPR' : nFPR_min,
            'FOR' : nFOR_min,
            'FNR' : nFNR_min,
            'NPV' : nNPV_min,
            'TNR' : nTNR_min}

        return matrix_min, matrix_maj


class Metric(namedtuple('Metric', 'cm_minority cm_majority')):
    def statistical_parity(self):
        statistical_parity_maj = float(self.cm_majority['TP'] + self.cm_majority['FP']) / max((self.cm_majority['TP'] + self.cm_majority['FP'] + self.cm_majority['FN'] + self.cm_majority['TN']), 1)
        statistical_parity_min = float(self.cm_minority['TP'] + self.cm_minority['FP']) / max((self.cm_minority['TP'] + self.cm_minority['FP'] + self.cm_minority['FN'] + self.cm_minority['TN']), 1)
        return np.fabs(statistical_parity_maj - statistical_parity_min)
    
    def predictive_parity(self):
        return np.fabs(self.cm_majority['PPV'] - self.cm_minority['PPV'])

    def predictive_equality(self):
        return np.fabs(self.cm_majority['FPR'] - self.cm_minority['FPR'])

    def equal_opportunity(self):
        return np.fabs(self.cm_majority['TPR'] - self.cm_minority['TPR'])

    def equalized_odds(self):
        return np.max([np.fabs(self.cm_majority['TPR'] - self.cm_minority['TPR']), np.fabs(self.cm_majority['FPR'] - self.cm_minority['FPR'])])

    def conditional_use_accuracy_equality(self):
        return np.max([np.fabs(self.cm_majority['PPV'] - self.cm_minority['PPV']), np.fabs(self.cm_majority['NPV'] - self.cm_minority['NPV'])])

    def fairness_metric(self, id):

        if id == 1:
            return self.statistical_parity()

        if id == 2:
            return self.predictive_parity()

        if id == 3:
            return self.predictive_equality()

        if id == 4:
            return self.equal_opportunity()

        if id == 5:
            return self.equalized_odds()

        if id == 6:
            return self.conditional_use_accuracy_equality()

    def fairness_metric_alt(self, id):
        if id == 1:
            return self.statistical_parity()    # DP

        if id == 2:
            return self.predictive_equality()   # FPP

        if id == 3:
            return self.equal_opportunity()     # TPP

        if id == 4:
            return self.equalized_odds()        # EOdds

    def fairness_metric_alt_alt(self, id):
        if id == 'dp':
            return self.statistical_parity()    # DP

        if id == 'fpr':
            return self.predictive_equality()   # FPP

        if id == 'tpr':
            return self.equal_opportunity()     # TPP

        if id == 'eodds':
            return self.equalized_odds()        # EOdds


def get_sklearn_metrics(epsilon_frame: pd.DataFrame, metrics: List[Dict[str, Callable]], trues: npt.ArrayLike,
                        preds: npt.ArrayLike, membership_data: npt.ArrayLike, seed: int, sensitive: List[str]) \
        -> pd.DataFrame:
    """
    Measure metrics passed (must be sklearn-implemented metrics) into dataframe at passed seed row
    :param epsilon_frame:
    :param metrics:
    :param trues:
    :param preds:
    :param membership_data:
    :param seed:
    :param sensitive:
    :return:
    """
    multi_metric = MetricFrame(metrics=metrics[0], y_true=trues, y_pred=preds, sensitive_features=membership_data)
    sklearn_results = multi_metric.by_group
    epsilon_frame.loc[seed, sensitive[0]] = sklearn_results.loc[sensitive[0]].values
    epsilon_frame.loc[seed, sensitive[1]] = sklearn_results.loc[sensitive[1]].values
    epsilon_frame.loc[seed, ('Across groups', list(metrics[0].keys()))] = multi_metric.overall.values
    return epsilon_frame


def get_fair_metric_score(epsilon_frame: pd.DataFrame, metric: Callable, metric_name: str, trues: npt.ArrayLike,
                          preds: npt.ArrayLike, membership_data: npt.ArrayLike, seed: int) -> pd.DataFrame:
    """
    Get fairlearn's fairness metrics, store at row seed in epsilon frame
    :param epsilon_frame:
    :param metric:
    :param metric_name:
    :param trues:
    :param preds:
    :param membership_data:
    :param seed:
    :return:
    """
    # get fairness metric the IM was trained on
    if metric in [demographic_parity_ratio, equalized_odds_ratio]:
        epsilon_frame.loc[seed, ('Across groups', metric_name)] = metric(trues, preds,
                                                                         sensitive_features=membership_data)
    else:  # TPR, FPR
        epsilon_frame.loc[seed, ('Across groups', metric_name)] = metric(trues, preds)
    return epsilon_frame


def get_old_unf(epsilon_frame: pd.DataFrame, metric_name: str, trues: npt.ArrayLike, preds: npt.ArrayLike,
                X_data: npt.ArrayLike, grp_names: List[str], seed: int) -> pd.DataFrame:
    maj_features_train = X_data[grp_names[0]]
    min_features_train = X_data[grp_names[1]]
    cm_train = ConfusionMatrix(min_features_train, maj_features_train, preds, trues)
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)
    unf_train = fm_train.fairness_metric_alt_alt(metric_name)
    epsilon_frame.loc[seed, ('Across groups', metric_name)] = unf_train
    return epsilon_frame


def wasserstein_dist(a, b):
    """
    Compute wasserstein over groups
    :param a: Confusion matricies for one group and ...
    :param b: ... for the other group
    :return:
    """
    dist = wasserstein_distance(a.ravel(), b.ravel())
    return dist


def kl_conf_matrix(maj_matr: npt.ArrayLike, min_matr: npt.ArrayLike) -> float:
    """
    Returns kl-divergence over groups given their confusion matrix values. Conf matrix values for each group are found
    between IM predictions and BB labels as follows:
        TN: IM=BB=1
        FP: IM!=BB, BB=1
        FN: IM!=BB, BB=0
        TP: IM=BB=0
    These values are normalized w.r.t. true (BB) labels.
    The conf matricies fot each group are flattened. These 1D arrays are then compared to given the kl-divergence.
    :param maj_matr:    Scipy's conf matrix for major group
    :type maj_matr:     Array
    :param min_matr:    Scipy's conf matrix for minor group
    :type min_matr:     Array
    :return:            Float, KL-divergence over
    """
    # entropy normalizes vectors first like so: vector/(sum(vector))
    ''' # figuring out kl manually for R code reproduction
    names = ['TN', 'FP', 'FN', 'TP']
    majss = maj_matr.ravel()  # tn, fp, fn, tp
    minss = min_matr.ravel()
    cm_majority = dict(zip(names, majss))
    cm_minority = dict(zip(names, minss))
    mins = [0] * 4
    majs = [0] * 4
    mins[0] = float(cm_minority['TP']) / (cm_minority['TP'] + cm_minority['FN'])
    mins[1] = float(cm_minority['FN']) / (cm_minority['TP'] + cm_minority['FN'])
    mins[2] = float(cm_minority['TN']) / (cm_minority['TN'] + cm_minority['FP'])
    mins[3] = float(cm_minority['FP']) / (cm_minority['TN'] + cm_minority['FP'])
    minss = 1.0*np.asarray(mins)/np.sum(mins, axis=0, keepdims=True)

    majs[0] = float(cm_majority['TP']) / (cm_majority['TP'] + cm_majority['FN'])
    majs[1] = float(cm_majority['FN']) / (cm_majority['TP'] + cm_majority['FN'])
    majs[2] = float(cm_majority['TN']) / (cm_majority['TN'] + cm_majority['FP'])
    majs[3] = float(cm_majority['FP']) / (cm_majority['TN'] + cm_majority['FP'])
    majss = 1.0*np.asarray(majs)/np.sum(majs, axis=0, keepdims=True)

    kl_TP = majss[0] * np.log(majss[0]/minss[0])
    kl_FP = majss[1] * np.log(majss[1]/minss[1])
    kl_TN = majss[2] * np.log(majss[2]/minss[2])
    kl_FN = majss[3] * np.log(majss[3]/minss[3])
    kldiv = kl_TP + kl_FP + kl_TN + kl_FN
   '''
    majs = maj_matr.ravel()  # tn, fp, fn, tp
    mins = min_matr.ravel()
    # print(majs)
    kldiv = entropy(majs, mins)
    return kldiv


def kl_tp_fp(maj_matr: npt.ArrayLike, min_matr: npt.ArrayLike) -> float:
    maj_matr = maj_matr.ravel()
    min_matr = min_matr.ravel()
    maj_important = [maj_matr[1], maj_matr[3]]
    min_important = [min_matr[1], min_matr[3]]
    ### FP and FN (when washed, m has highFP,lowFN, opposite for M):
    #maj_important = [maj_matr[1], maj_matr[2]]
    #min_important = [min_matr[1], min_matr[2]]
    kl = entropy(maj_important, min_important)
    return kl


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def add_seeds(df):
    # add column showing seed number assuming every seed is present and inorder
    epsilon_counts = df['epsilon'].value_counts()
    num_seeds = epsilon_counts[0]
    seeds = np.array(list(range(num_seeds)) * len(epsilon_counts))
    df['seed'] = seeds
    return df


def compute_front(df, unf, to_front):
    """
    Gets the pareto-efficient subset (between to_front and unf) of dataframe.
    :param df:          Must have seed column
    :param unf:         Unfairness metric name
    :param to_front:    Metric to id pareto by (e.g. accuracy)
    :return:            Pareto subset dataframe
    """
    pareto_df = pd.DataFrame(columns=df.columns)  # strores pareto-subset of inout frame
    for seed in range(max(df['seed'])+1):
        # for each seed, keep only the pareto-efficient subset
        rows = df.loc[df['seed'] == seed]
        errors = 1.0 - rows[('Across groups', to_front)]
        unfairness = rows[('Across groups', unf)]
        pareto_input = [[error, unf] for (error, unf) in zip(errors, unfairness)]
        pareto_input = np.array(pareto_input)
        msk = is_pareto_efficient(pareto_input)
        rows = rows[msk]
        pareto_df = pd.concat([pareto_df, rows])
    return pareto_df


def kl_ce(maj_CE: float, min_CE: float) -> float:
    """
    Get kl div over cross entropy losses of each group. Compares each groups' average CE loss.
    :param maj_CE:  Average CE loss (IM prediction, BB labels) of samples in major group.
    :type maj_CE:   Float
    :param min_CE:  Average CE loss (IM prediction, BB labels) of samples in minor group.
    :type min_CE:   Float
    :return:        Float, KL-divergence over average CE loss of each group.
    """
    # given gender cross entropy/ log losses, get KL over those
    kldiv = entropy([maj_CE, min_CE])
    return kldiv


def ttest(preds: npt.ArrayLike, trues: npt.ArrayLike, membership_data: npt.ArrayLike, partition: bool,
          n_subsamples: int, n_samples_per_subsample: int) -> Tuple[
    npt.ArrayLike, Union[float, np.ndarray, Iterable, int]]:
    """
    Performs student ttest to check if means between two groups (sensitives) are related multivariately over
     tpr, fpr etc. rates. for multiple sample sized of the test set
    :param preds:
    :param trues:
    :param membership_data:
    :param partition:               whether or not to partition (the samples will be disjoint, i.e. NO correlation)
    :param n_subsamples:            if partition == False, the number of subsamples to randomly draw
    :param n_samples_per_subsample: number of samples to draw for each subsample OR if partition == True, the
                                    number of samples per partition (n_suing MUST be divisible by this number)
    :return:
    """

    def _tpr(x):
        return x["true_positives"] / x["positives"]

    def _tnr(x):
        return x["true_negatives"] / x["negatives"]

    def _fpr(x):
        return x["false_positives"] / x["negatives"]

    def _fnr(x):
        return x["false_negatives"] / x["positives"]

    # n_suing = 10000  ## Number of suing samples
    n_suing = len(preds)
    sa = membership_data
    bb_y = trues
    im_y = preds
    if partition:
        assert n_suing % n_samples_per_subsample == 0, "To equally partition samples, the number of samples per " \
                                                       "partition must divide the number of suing samples."
        shuffle_idx = np.random.permutation(n_suing)
        bb_y = trues[shuffle_idx]
        im_y = preds[shuffle_idx]
        sa = sa[shuffle_idx]
        n_experiments = n_suing // n_samples_per_subsample
    else:
        n_experiments = n_subsamples
    tprs, tnrs, fprs, fnrs = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    # draw n_experiments subsamples, each with a crap tonne of samples
    for i in range(n_experiments):
        # Subsample (either disjoint partition or single random subsample of suing set)
        if partition:
            subsample = {"bb": bb_y[i * n_samples_per_subsample:(i + 1) * n_samples_per_subsample],
                         "im": im_y[i * n_samples_per_subsample:(i + 1) * n_samples_per_subsample],
                         "sa": sa[i * n_samples_per_subsample:(i + 1) * n_samples_per_subsample]}
        else:
            subsample_index = random.choices(range(n_suing), k=n_samples_per_subsample)
            subsample = {"bb": bb_y[subsample_index],
                         "im": im_y[subsample_index],
                         "sa": sa[subsample_index]}

        if np.unique(subsample["bb"]).shape[0] == 1:
            continue

        for unique_sa in np.unique(subsample["sa"]):
            # For each subsample, calculate the tpr, tnr, fpr, and fnr for each value of the sensitive attribute
            # (assumedly 0 and 1)
            grp_bb = subsample["bb"][subsample["sa"] == unique_sa]
            grp_im = subsample["im"][subsample["sa"] == unique_sa]
            counts = {
                "positives": sum(grp_bb),
                "negatives": n_samples_per_subsample - sum(grp_bb),
                "true_positives": np.logical_and(grp_bb == 1, grp_im == 1).sum(),
                "true_negatives": np.logical_and(grp_bb == 0, grp_im == 0).sum(),
                "false_positives": np.logical_and(grp_bb == 0, grp_im == 1).sum(),
                "false_negatives": np.logical_and(grp_bb == 1, grp_im == 0).sum()
            }
            tpr = _tpr(counts)
            tnr = _tnr(counts)
            fpr = _fpr(counts)
            fnr = _fnr(counts)
            tprs[unique_sa].append(tpr)
            tnrs[unique_sa].append(tnr)
            fprs[unique_sa].append(fpr)
            fnrs[unique_sa].append(fnr)
    # Stack the tprs, tnrs and fprs for each sensitive attribute in order to calculate a multiple variable
    # t - test on the tpr, tnr, fpr and fnr (t - test between tpr, tnr, fpr, and fnr respectively)
    stats_dict = {}
    # get conf matrix rates for each subgroup
    for unique_sa in np.unique(sa):
        stats_dict[unique_sa] = np.vstack((tprs[unique_sa], tnrs[unique_sa], fprs[unique_sa], fnrs[unique_sa]))
    sa_keys = list(stats_dict.keys())
    # Obtain p - values across groups
    stats, p_vals = ttest_ind(stats_dict[sa_keys[0]], stats_dict[sa_keys[1]], equal_var=False, axis=1)
    # p_vals = ttest_results.pvalue
    # stat = ttest_results.statistic
    # stat, _p_val = ttest_ind(stats_dict[sa_keys[0]], stats_dict[sa_keys[1]], equal_var=False, axis=None)
    # Use multiple testing correction method to adjust p-values and print out p-values for each rate
    # can choose from bonferroni, sidak, holm, fdr_bh and fdr_tsbh
    reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(p_vals, method="bonferroni")
    # format is tp, fp, fn, tn
    data = np.asarray([[pvals_corrected[0], pvals_corrected[2]], [pvals_corrected[3], pvals_corrected[1]]])
    # print(dict(zip(["tpr", "tnr", "fpr", "fnr"], pvals_corrected)))
    return data, stats


def get_divisors(n: int) -> List[int]:
    """
    Get list of divisors of a number
    :param n:
    :return:
    """
    divisors = []
    i = 1
    while i <= n:
        if n % i == 0:
            divisors.append(i)
        i += 1
    return divisors


def ttest_set_data(preds: npt.ArrayLike, trues: pd.Series, membership_data: pd.Series) \
        -> Tuple[int, npt.ArrayLike, pd.Series, pd.Series]:
    """
    Since the number of subsamples for ttest needs to evenly divide the length of the data we're testing, we might need
    to finagle the number of datapoints, (AKA drop one sample randomly (seed controlled)) until we have a not-prime
    number of samples. Then we can compute the divisors of the set size and select one that usually will divide the set
    in half.
    :param preds:
    :param trues:
    :param membership_data:
    :return:                Subsample size, altered trues preds and membership data to use for the ttest only.
    """
    divisors = get_divisors(len(preds))
    while len(divisors) <= 2:  # prime number of samples to test
        # decrease data to test by one
        ind_to_drop = np.random.randint(0, high=len(preds))
        preds = np.delete(preds, [ind_to_drop], axis=0)
        trues = trues.drop(ind_to_drop, axis=0)
        membership_data = membership_data.drop([ind_to_drop], axis=0)
        divisors = get_divisors(len(preds))
    return divisors[-2], trues, preds, membership_data  # get value about half (or a third etc) the size of entire set


def kl_ce_samples(logits: npt.ArrayLike, labels: npt.ArrayLike, memberships: npt.ArrayLike, normalize: bool, nbins: int,
                  sensitive: list, seed: int, dataset: str, epsilon: float, metric_name: str, mit: str, exp: str,
                  bb_model: str) -> float:
    """
    Plot kl divergence over sample-wise cross entropy values. CE calculated for each sample from IM probas and BB
    labels. CE values are split to groups. Make histograms of nbins to describe sample frequencies of CE values.
    Histograms for each group share bins and are made with density normalization to nullify group imbalance. Histogram
    frequencies from each group are compared via KL.

    ALSO: currently plots each groups histograms together.

    :param logits:          IM probas
    :param labels:          BB labels
    :param memberships:     Sample groupings.
    :param nbins:           Number of bins to consider
    :param sensitive:       Sensitive gorup names
    :param normalize:       If true, area under histogram integrates to 1. (More like density than normalization)
    :param seed:            Seed computing for
    :param dataset:         Name of dataset computing for
    :param epsilon:         Value of epsilon computing for
    :param metric_name:     Name of fairness metric IM was fairwashed with
    :param mit:             Name of mitigator
    :param exp:             Name of explainer/interpretable model
    :param bb_model:        Name of BB model
    :return:                Float, KL over histogram frequencies
    """

    array = []
    for logit in logits:
        array.append([1 - logit, logit])
    array = np.array(array)

    logits = torch.tensor(array)
    logits = logits.double()
    labels = torch.tensor(labels.to_numpy())
    ce = cross_entropy(logits, labels, reduction='none')
    ce = ce.numpy()
    x_range = (min(ce), max(ce))

    cross_entropy_values_M = ce[memberships == sensitive[0]]
    cross_entropy_values_m = ce[memberships == sensitive[1]]

    min_range = min([min(cross_entropy_values_M), min(cross_entropy_values_m)])
    max_range = max([max(cross_entropy_values_M), max(cross_entropy_values_m)])
    x_range = (min_range, max_range)

#    fig, axs = plt.subplots()
#    plt.yscale('log', nonposy='clip')
#    plt.xlabel('CE values')
#    plt.ylabel('Frequency (norm. via Density)') if normalize else plt.ylabel('Frequency')
#    axs.hist(cross_entropy_values_M, bins=nbins, range=x_range, density=normalize, alpha=0.5, label='Major group',
#              color='red')
#    axs.hist(cross_entropy_values_m, bins=nbins, range=x_range, density=normalize, alpha=0.5, label='Minor group',
#              color='blue')
#    plt.ylim(0, 100)
#    plt.xlim(0, 1.6)

    # take kl_divergece between histogram bin counts
    hist0 = np.histogram(cross_entropy_values_M, range=x_range, bins=nbins, density=normalize)
    hist1 = np.histogram(cross_entropy_values_m, range=x_range, bins=nbins, density=normalize)
    # hist0[0][hist0[0] == 0] = .000000001
    hist1[0][hist1[0] == 0] = .0000000000000001  # stop useless infinties in KL
    kl_div = entropy(hist0[0], hist1[0])
    #plt.title('kl-div=' + str(kl_div))
    #axs.legend()
    #save_path = '../2-11-21/distributions_{}_{}_{}_{}/{}_{}/{}_{}.png'.format(dataset, mit, exp, nbins,
     #                                                                                         metric_name,
      #                                                                                        bb_model, epsilon, seed)
    #os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    #fig.savefig(save_path, bbox_inches='tight')

    return kl_div


def kl_ce_ad_lab(logits: npt.ArrayLike, predictions: npt.ArrayLike, labels: npt.ArrayLike, memberships: npt.ArrayLike,
                 sensitive: list, nbins: int, normalize: bool) -> npt.ArrayLike:
    """
    Get sample-wise cross-entropy loss between logits and labels. Then, partition into group, agree/disagree and the
    label. (For disagree, the label is given by the labels)

    CURRENTLY: Concats all the conf matrix sample ce histogram values to one array per group and does entropy across
    them.
    :param logits:          Logit array (1D)
    :param predictions:     Predicitions (from IM or whomever's labels we happen to be using
    :param labels:          Labels (from BB or whomever's labels we happen to be using)
    :param memberships:     Group membership info
    :param sensitive:       Group names, major then minor.
    :param nbins:           Number bins for histogram
    :param normalize:       If true, density normalization (integrates to 1)
    :return:
    """
    # KL divergence over CE of samples partitioned into group agree/disagree and labels
    array = []
    for logit in logits:
        array.append([1 - logit, logit])
    array = np.array(array)

    logits = torch.tensor(array)
    logits = logits.double()
    labels = torch.tensor(labels.to_numpy())
    ce = cross_entropy(logits, labels, reduction='none')
    ce = ce.numpy()
    x_range = (min(ce), max(ce))

    # M_a_0, M_d_0, M_a_1, M_d_1 for each group (M/m)
    group_data = []
    for group in sensitive:
        for label in [0, 1]:
            g = np.where(memberships == group)
            a = np.where(np.equal(predictions, labels))
            d = np.where(np.not_equal(predictions, labels))
            idx_a = np.intersect1d(g, a)
            idx_d = np.intersect1d(g, d)
            l = np.where(labels == label)
            idx_a_l = np.intersect1d(idx_a, l)
            idx_d_l = np.intersect1d(idx_d, l)
            ce_a_label = ce[idx_a_l]
            group_data.append([ce_a_label])
            ce_d_label = ce[idx_d_l]
            group_data.append([ce_d_label])
    major_data = group_data[0: 4]
    minor_data = group_data[4:]

    kls = []
    M_histos = []
    m_histos = []
    for i in range(len(major_data)):
        M_histo = np.histogram(np.array(major_data[i]), bins=nbins, range=x_range, density=normalize)[0]
        M_histos.extend(M_histo)
        m_histo = np.histogram(np.array(minor_data[i]), bins=nbins, range=x_range, density=normalize)[0]
        m_histos.extend(m_histo)
        kls.append(entropy(M_histo, m_histo))
    # take kl of sample-wise conf metric values
    m_histos = np.array(m_histos)
    m_histos[m_histos == 0] = .000000000000001
    kl_div = entropy(M_histos, m_histos)

    # a_0, d_0, a_1, d_1 format of KLs taken across groups
    return kl_div  # np.array(kls)


def GT_washed(n: float, metric_name: str, X_data, grp_names, gt_labels, bb_preds, im_preds) -> bool:
    """
    Determine if IM model was fairwashed within an n% drop in unfairness from the BB_model. Acts as an oracle, as we
    dont have ground truths at real time. Both unfairnesses compared are w.r.t. the ground truths
    :param n:                   Hyperparaeter, in %s
    :param metric:              Metric to use to get bb_unfairness against ground truth labels
    :param membership_data:
    :param gt_labels:           Ground truth dataset labels, ORACLE HERE
    :param bb_preds:            BB_predictions on set
    :param im_preds:            IM predictions on set
    :return:                    True if IM is too low and washed (can include unintentional based on n),
                                    false if no washing.
    """

    bb_unfairness = gt_washed_unfairness(X_data, grp_names, bb_preds, gt_labels, metric_name)
    im_unfairness = gt_washed_unfairness(X_data, grp_names, im_preds, gt_labels, metric_name)
    lower_bound_acceptable = bb_unfairness - (n/100.0 * bb_unfairness)
    if im_unfairness < lower_bound_acceptable:
        return True
    else:
        return False


def gt_washed_unfairness(X_data, grp_names, preds, trues, metric_name):
    maj_features_train = X_data[grp_names[0]]
    min_features_train = X_data[grp_names[1]]
    cm_train = ConfusionMatrix(min_features_train, maj_features_train, preds, trues)
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)
    unf_train = fm_train.fairness_metric_alt_alt(metric_name)
    return unf_train


def threshold_detector(thresholds: list, value: float) -> Dict[float, bool]:
    """
    Outputs dictionary of detection values for each metric provided, given the value from the metric's output.
    True = washing detected, False = no washing detected
    :param thresholds:
    :param pred_metric:
    :return:
    """
    results = {}
    for threshold in thresholds:
        if value >= threshold:
            results[threshold] = True
        else:
            results[threshold] = False
    return results
