
import pandas as pd
import numpy as np
import numpy.typing as npt
from faircorels import CorelsClassifier
from fairlearn.reductions import DemographicParity, FalsePositiveRateParity, \
    TruePositiveRateParity, EqualizedOdds
from config_fairlearn import *

from my_types import *
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

from typing import Dict, Tuple, List, Union, Callable

######################
# IM
######################


def get_IM_path(dataset: str, mit: str, exp: str, metric: Callable, epsilon: float, seed: int, use_test: bool,
                bb_model: str) -> str:
    """
    Get path of IM predictions to read them in
    :param dataset:     Dataset name
    :param mit:         Name of mitigator
    :param exp:         Name of IM
    :param metric:      Callable metric
    :param epsilon:
    :param seed:
    :param use_test:    If true, returns path to test set, otherwise suing set
    :param bb_model:    Name of BB model
    :return:
    """
    if use_test:
        path = '../models/labels/{}/{}/{}/{}/{}/{}_{}_test.pkl'.format(dataset, mit, exp, metric, epsilon, bb_model,
                                                                       seed)
    else:
        path = '../models/labels/{}/{}/{}/{}/{}/{}_{}_suing.pkl'.format(dataset, mit, exp, metric, epsilon, bb_model,
                                                                        seed)
    return path


def get_IM_path_probas(dataset: str, mit: str, exp: str, metric: Callable, epsilon: float, seed: int,
                       use_test: bool, bb_model: str) -> str:
    """
    Get path of IM predictions to read them in
    :param dataset:
    :param mit:
    :param exp:
    :param metric:
    :param epsilon:
    :param seed:
    :param use_test:
    :param bb_model:
    :return:
    """
    if use_test:
        path = '../models/labels/{}/{}/{}/{}/{}/{}_{}_test_probas.pkl'.format(dataset, mit, exp, metric, epsilon,
                                                                              bb_model, seed)
    else:
        path = '../models/labels/{}/{}/{}/{}/{}/{}_{}_suing_probas.pkl'.format(dataset, mit, exp, metric, epsilon,
                                                                               bb_model, seed)
    return path


def get_IM_predictions(mit: str, exp: str, dataset: str, min_grp: str, epsilon: float, seed: int,
                       suing_X: npt.ArrayLike, bb_model: str, metric: Callable, use_test: bool, save: bool) \
        -> npt.ArrayLike:
    """
    Gets IM predictions on passed data, if save, will save preds so future calculations can just read in our previous
    work, pickle for speeed.
    :param mit:
    :param exp:
    :param dataset:
    :param min_grp:
    :param epsilon:
    :param seed:
    :param suing_X:
    :param bb_model:
    :param metric:
    :param use_test:
    :param save:
    :return:
    """
    # get filepath to save IM preds to, and check if we've already generated them
    pred_path = get_IM_path(dataset, mit, exp, metric, epsilon, seed, use_test, bb_model)
    proba_path = get_IM_path_probas(dataset, mit, exp, metric, epsilon, seed, use_test, bb_model)
    if os.path.isfile(pred_path) and os.path.isfile(proba_path) and False:  # turning off reading premade data
        # print('reading')
        preds = read_pickle(pred_path)
        probas = read_pickle(proba_path)
    else:  # train IM
        suing_y = get_BB_preds(dataset, bb_model, seed, use_test=False)
        test_X = get_datasets(dataset, seed, 'testing', exp) if use_test else suing_X
        if exp != 'rl':
            # difference bound bounds difference between groups,
            # so epsilon = 1 means no difference is permitted between groups, eps = 0 means no constraint
            constraint = metric(difference_bound=1.0-epsilon)  # , ratio_bound=1-epsilon)
            mitigator, explainer = get_mit_exp(mit, exp)
            mitigator = mitigator(explainer, constraint)
            # get data and BB labels to train IM
            min_grp_data = suing_X[min_grp]

            # fit to BB labels on suing set
            mitigator.fit(suing_X, suing_y, sensitive_features=min_grp_data)
            preds = mitigator.predict(test_X, random_state=seed)
            probas = expgrad_predict_proba(mitigator, test_X)
        else:  # IM == rule list
            N_ITER = 300000
            strategy, bfsMode, strategy_name = get_strategy_rl(1)
            fairness, dataset_num = get_fairness_number_rl(metric, dataset)
            ulb = False
            dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data_rl(dataset_num)
            clf = CorelsClassifier(n_iter=N_ITER, min_support=0.01, c=1e-3, max_card=1, policy=strategy,
                                   bfs_mode=bfsMode, mode=3, useUnfairnessLB=ulb, forbidSensAttr=True,
                                   fairness=fairness, epsilon=epsilon, maj_pos=maj_pos, min_pos=min_pos, verbosity=[])
            clf.fit(suing_X.to_numpy(), suing_y, features=list(suing_X.columns), prediction_name=prediction_name)
            preds = clf.predict(test_X)
            probas = preds
        if save:
            # print('saving')
            save_pickle(preds, pred_path)
            save_pickle(probas, proba_path)
    return preds, probas


def get_fairness_number_rl(metric: Callable, dataset: str) -> Tuple[int, int]:
    # faircorels takes an int to designate fairness:
    # https://snyk.io/advisor/python/faircorels#package-footer
    fairness_map = {
        DemographicParity: 1,  # AKA statistical parity
        EqualizedOdds: 5,
        TruePositiveRateParity: 4,  # Equal Opp
        FalsePositiveRateParity: 3  # AKA predictive equality
    }
    dataset_map = {
        'adult_income': 1,
        'compas': 2,
        'default_credit': 3,
        'marketing': 4
    }
    return fairness_map[metric], dataset_map[dataset]


def expgrad_predict_proba(mitigator: ExponentiatedGradient, X: np.ndarray) -> npt.ArrayLike:
    # gets positive class probabilities
    probs = [None] * len(mitigator.predictors_)
    for t in range(len(mitigator.predictors_)):
        if mitigator.weights_[t] == 0:
            probs[t] = 0. * mitigator.predictors_[t].predict_proba(X)
        else:
            probs[t] = mitigator.predictors_[t].predict_proba(X)

        if probs[t].shape[1] == 2:
            probs[t] = probs[t][:, 1]
    probs = np.stack(probs)
    probs = np.tensordot(mitigator.weights_.values, probs[mitigator.weights_.index], axes=1)
    return probs


##########################
# BB
##########################


def get_BB_preds(dataset: str, bb_model: str, seed: int, use_test: bool) -> npt.ArrayLike:
    """
    Get bb models preds
    :param dataset:
    :param bb_model:
    :param seed:
    :param use_test:
    :return:
    """
    # can return BB predictions for preds or for GTs
    if use_test:  # get BB preds on test set
        path = '../models/labels/{}/{}_test_{}.csv'.format(dataset, bb_model, seed)
        # if not os.path.isfile(path):
        #     path = '../models/labels/{}/{}_test_{}.csv'.format(dataset, bb_model, seed)
        #     data = pd.read_csv(path)
        #     data = data['prediction']
        #     data = pd.Series(data)
        #     data.to_pickle('../models/labels/{}/{}_test_{}.pkl'.format(dataset, bb_model, seed))
        # else:
            # print('reading bb')
        #data = pd.read_pickle(path)
    else:  # just get BB preds on suing set
        path = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, bb_model, seed)
        # if not os.path.isfile(path):
        #     path = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, bb_model, seed)
        #     data = pd.read_csv(path)
        #     data = data['prediction']
        #     data = pd.Series(data)
        #     data.to_pickle('../models/labels/{}/{}_sg_{}.pkl'.format(dataset, bb_model, seed))
        # else:
            # print('reading bb')
        #data = pd.read_pickle(path)
    # preds = pd.read_pickle(path)  # ['prediction']
    data = pd.read_csv(path)['prediction']
    return data


##########################
# Data and Labels
##########################

def get_labels(dataset: str, seed: int, get_test: bool) -> npt.ArrayLike:
    """
    Get ground truth labels of testing or suing
    :param dataset:
    :param seed:
    :param get_test:
    :return:
    """
    if get_test:
        # get and return the GT for test
        path = '../models/true_labels/{}/label_test_{}.csv'.format(dataset, seed)
        # if not os.path.isfile(path):
        #     path = '../models/true_labels/{}/label_test_{}.csv'.format(dataset, seed)
        #     data = pd.read_csv(path)
        #     # data.to_pickle('../models/true_labels/{}/label_test_{}.pkl'.format(dataset, seed))
        # else:
        #     data = pd.read_pickle(path)
    else:
        # get and return the GT for suing
        path = '../models/true_labels/{}/label_sg_{}.csv'.format(dataset, seed)
        # if not os.path.isfile(path):
        #     path = '../models/true_labels/{}/label_sg_{}.csv'.format(dataset, seed)
        #     data = pd.read_csv(path)
        #     #data.to_pickle(path = '../models/true_labels/{}/label_sg_{}.pkl'.format(dataset, seed))
        # else:
        #     data = pd.read_pickle(path)
    # gts = pd.read_pickle(path)  # ['prediction']
    # gts = pd.read_csv(path)['prediction']
    data = pd.read_csv(path)['prediction']
    return data


def get_possible_labels(dataset: str):
    # return possible label outcomes
    return [0, 1]


def get_datasets(dataset: str, seed: int, sets: str, exp: str) -> pd.DataFrame:
    """
    Get X data of specified dataset
    :param dataset:
    :param seed:
    :param sets:    If ='suing' gets the suing set, otherwise gets the test set
    :return:
    """
    if sets == 'suing':
        if exp == 'rl' and False:
            filename = '../preprocessing/preprocessed/{}/{}_attackRules_{}.pkl'.format(dataset, dataset, seed)
        else:
            filename = '../preprocessing/preprocessed/{}/{}_attackOneHot_{}.pkl'.format(dataset, dataset, seed)
        if not os.path.isfile(filename):
            print('here')
            filename_csv = '.'.join(filename.split('.')[:-1]) + '.csv'
            # filename = '../preprocessing/preprocessed/{}/{}_attackOneHot_{}.csv'.format(dataset, dataset, seed)
            data = pd.read_csv(filename_csv)
            # data.to_pickle('../preprocessing/preprocessed/{}/{}_attackOneHot_{}.pkl'.format(dataset, dataset, seed))
            data.to_pickle(filename)
        else:
            data = pd.read_pickle(filename)
    else:  # sets == 'test'
        if exp == 'rl':
            filename = '../preprocessing/preprocessed/{}/{}_testRules_{}.pkl'.format(dataset, dataset, seed)
        else:
            filename = '../preprocessing/preprocessed/{}/{}_testOneHot_{}.pkl'.format(dataset, dataset, seed)
        if not os.path.isfile(filename):
            filename = '../preprocessing/preprocessed/{}/{}_testOneHot_{}.csv'.format(dataset, dataset, seed)
            data = pd.read_csv(filename)
            data.to_pickle('../preprocessing/preprocessed/{}/{}_testOneHot_{}.pkl'.format(dataset, dataset, seed))
        else:
            data = pd.read_pickle(filename)
    # data = pd.read_pickle(filename)
    return data


def get_mit_exp(mit: str, exp: str) -> Tuple[Mit_type, Explainer]:
    """
    Get mitigator and explainer classes/models from strings
    :param mit:     Mitigator string
    :param exp:     Explainer string
    :return:        Mitigator and explainer
    """
    explainer_map = {
        'lm': LogisticRegression(solver='liblinear', fit_intercept=True),
        'dt': DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    }
    mitigator_map = {
        'eg': ExponentiatedGradient,
        'gs': GridSearch
    }
    return mitigator_map[mit], explainer_map[exp]


def get_epsilons(metric: str, dataset: str) -> list:
    """
    Get tuned epsilons for each dataset and metric.
    :param metric:
    :param dataset:
    :return:
    """
    # for Demographic Parity
    dp_epsilons_map = {
        'adult_income': [0.0, .95, .999],
        'compas': [0.0, .92, .96],
        'default_credit': [0.0, .999, .9995],
        'marketing': [0.0, .97, .99]
    }

    # original models already have low unfairness
    fpp_epsilons_map = {
        'adult_income': [0.0, .95, .99],
        'compas': [0.0, .95, .99],
        'default_credit': [0.0, .2, .8],
        'marketing': [0.0, .15, .2]
    }

    # already low unairness
    tpp_epsilons_map = {
        'adult_income': [0.0, .2, .3],
        'compas': [0.0, .5, .9],
        'default_credit': [0.0, .5, .9],
        'marketing': [0.0, .5, .9]
    }

    eodds_epsilons_map = {
        'adult_income': [0.0, .5, .9],
        'compas': [0.0, .5, .9],
        'default_credit': [0.0, .5, .9],
        'marketing': [0.0, .5, .9]
    }

    epsilons_map = {
        'dp': dp_epsilons_map,
        'fpr': fpp_epsilons_map,
        'tpr': tpp_epsilons_map,
        'eodds': eodds_epsilons_map
    }
    return epsilons_map[metric][dataset]


##########################
# Unconstrained, (old stuff)
#########################


def get_unc_IM_path(dataset: str, mit: str, exp: str, metric: Callable, epsilon: float, seed: int, use_test: bool) \
        -> str:
    if use_test:
        path = '../models/labels_unc_im/{}/{}/{}/{}/{}/{}_test.pkl'.format(dataset, mit, exp, metric, epsilon, seed)
    else:
        path = '../models/labels_unc_im/{}/{}/{}/{}/{}/{}_suing.pkl'.format(dataset, mit, exp, metric, epsilon, seed)
    return path


def get_unc_IM_proba_path(dataset: str, mit: str, exp: str, metric: Callable, epsilon: float, seed: int,
                          use_test: bool) -> str:
    if use_test:
        path = '../models/labels_unc_im/{}/{}/{}/{}/{}/{}_test_probas.pkl'.format(dataset, mit, exp, metric, epsilon,
                                                                                  seed)
    else:
        path = '../models/labels_unc_im/{}/{}/{}/{}/{}/{}_suing_probas.pkl'.format(dataset, mit, exp, metric, epsilon,
                                                                                   seed)
    return path


def no_constraint_IM(mit: str, exp: str, dataset: str, epsilon: float, seed: int,
                       train_X: npt.ArrayLike, bb_model, metric: Callable, use_test: bool, save: bool) -> npt.ArrayLike:
    # train normal interpretable model
    pred_path = get_unc_IM_path(dataset, mit, exp, metric, epsilon, seed, use_test)
    proba_path = get_unc_IM_proba_path(dataset, mit, exp, metric, epsilon, seed, use_test)
    if os.path.isfile(pred_path) and os.path.isfile(proba_path):
        preds = read_pickle(pred_path)
        probas = read_pickle(proba_path)
    else:
        im = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=seed)
        # fit to BB labels of suing set
        train_y = get_BB_preds(dataset, bb_model, seed, use_test=False)
        im.fit(train_X, train_y)
        # get set to predict
        test_X = get_datasets(dataset, seed, 'testing', exp) if use_test else train_X
        preds = im.predict(test_X)
        probas = im.predict_proba(test_X)
        if save:
            print('saving')
            save_pickle(preds, pred_path)
            save_pickle(probas, proba_path)
    return preds, probas


#########################
# File IO
#########################


def read_pickle(path: str) -> npt.ArrayLike:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data: npt.ArrayLike, path: str) -> str:
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def write_results_frames(frames: Dict[str, pd.DataFrame], folder_path: str) -> str:
    """
    Writes frames to given folder + key in pickle format (speeed)
    :param frames:
    :param folder_path:
    :return:
    """
    # write frame to disk, folder path is most basic root of things
    os.makedirs(folder_path, exist_ok=True)
    for key in list(frames.keys()):
        store_to = os.path.join(folder_path, key)
        frames[key].to_pickle(store_to)
    return folder_path


def read_results_frames(folder_path: str, keys: Union[None, List[str]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reads results frames in folder_path corresponding to given keys
    :param folder_path:
    :param keys:
    :return:
    """
    if keys is not None:
        frames = dict.fromkeys(keys)
        for key in keys:
            read_from = os.path.join(folder_path, key)
            frames[key] = pd.read_pickle(read_from)
        return frames
    else:
        return pd.read_pickle(folder_path)


# def weighted_score(dataset, algorithm):
#     pred_all = np.zeros((dataset.shape[0],len(algorithm._classifiers)))
#     for i,clf in enumerate(algorithm._classifiers):
#         pred_all[:, i] = clf.predict_proba(dataset)[:, 1]
#
#     score = pred_all.dot(algorithm._expgrad_result._weights)
#
#     return score


def print_series(series):
    keys = list(series.keys())
    vals = list(series.values)
    for i in range(len(series)):
        key = keys[i]
        elem = vals[i]
        print(str(key) + '\t' + str(format(float(elem), ".4f")))
