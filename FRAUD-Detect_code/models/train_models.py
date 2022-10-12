from __future__ import print_function

from functools import partial

# hyper-opt
from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from hyperopt import space_eval
from hyperopt.pyll import scope

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential

# utils
import pickle
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset_dict = {
    1 : 'adult_income',
    2 : 'compas',
    3 : 'default_credit',
    4 : 'marketing'
}

data_dict = {
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'two_year_recid'),
    'default_credit'    : ('default_credit', 'DEFAULT_PAYEMENT'),
    'marketing'         : ('marketing', 'subscribed')      
}  

space_DNN = {
    'units_l1'      : hp.choice('units_l1', [25, 50, 75, 100, 150, 200]),
    'dropout_l1'    : hp.uniform('dropout_l1', 0, 0.5),
    'units_l2'      : hp.choice('units_l2', [25, 50, 75, 100, 150, 200]),
    'dropout_l2'    : hp.uniform('dropout_l2', 0, 0.5),
    'nbr_layers'    : hp.choice('nbr_layers',
                    [
                        {'layers':'two',              
                        },
                        {'layers':'three',
                        'units_l3_a': hp.choice('units_l3_a', [25, 50, 75, 100, 150, 200]),
                        'dropout_l3_a': hp.uniform('dropout_l3_a', 0, 0.5),     
                        }, 
                        {'layers':'four',
                        'units_l3_b': hp.choice('units_l3_b', [25, 50, 75, 100, 150, 200]),
                        'dropout_l3_b': hp.uniform('dropout_l3_b', 0, 0.5), 
                        'units_l4': hp.choice('units_l4', [25, 50, 75, 100, 150, 200]),
                        'dropout_l4': hp.uniform('dropout_l4', 0, 0.5),     
                        },
                    ]),
    'optimizer'     : hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
    'epochs'        : 30,
    }

space_RF = {
    'max_depth'         : scope.int(hp.uniform('max_depth', 1, 11)),
    'max_features'      : hp.choice('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'n_estimators'      : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'criterion'         : hp.choice('criterion', ["gini", "entropy"]),
    'min_samples_split' : hp.choice('min_samples_split', [2, 5, 10]),
    'min_samples_leaf'  : hp.choice('min_samples_leaf', [1, 2, 4]),
    'bootstrap'         : hp.choice('bootstrap', [True, False]),
}

space_AdaBoost = {
    'n_estimators'          : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'learning_rate'         : hp.lognormal('learning_rate', np.log(0.01), np.log(10.0)),
    'algorithm'             : hp.choice('algorithm', ['SAMME', 'SAMME.R']),
}

space_XgBoost = {
    'max_depth'             : scope.int(hp.uniform('max_depth', 1, 11)),
    'n_estimators'          : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'learning_rate'         : hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'gamma'                 : hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'min_child_weight'      : scope.int(hp.loguniform('min_child_weight', np.log(1), np.log(100))),
    'subsample'             : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree'      : hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel'     : hp.uniform('colsample_bylevel', 0.5, 1),
    'reg_alpha'             : hp.loguniform('reg_alpha', np.log(0.0001), np.log(1)) - 0.0001,
    'reg_lambda'            : hp.loguniform('reg_lambda', np.log(1), np.log(4))
}


def prepare_data(data, rseed):

    dataset, decision = data_dict[data]
    datadir = '../preprocessing/preprocessed/{}/'.format(dataset)    

    #filenames
    suffix = 'OneHot'
    train_file      = '{}{}_train{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    test_file       = '{}{}_test{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    sg_file         = '{}{}_attack{}_{}.csv'.format(datadir, dataset, suffix, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)
    df_sg       = pd.read_csv(sg_file)

    # prepare the data
    scaler = StandardScaler()
    ## training set
    y_train = df_train[decision]
    X_train = df_train.drop(labels=[decision], axis = 1)
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    ## test set
    y_test = df_test[decision]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

     ## sg set
    y_sg = df_sg[decision]
    X_sg = df_sg.drop(labels=[decision], axis = 1)
    X_sg = scaler.fit_transform(X_sg)
    ### cast
    X_sg = np.asarray(X_sg).astype(np.float32)
    y_sg = np.asarray(y_sg).astype(np.float32)

    return X_train, y_train, X_test, y_test, X_sg, y_sg

# DNN
def obj_func__DNN(params, data, rseed):

    X_train, y_train, _, _, _, _ = prepare_data(data, rseed)

    model = Sequential()

    # first layer
    model.add(Dense(params['units_l1'], input_dim = X_train.shape[1]))
    model.add(Dropout(params['dropout_l1']))
    model.add(Activation('relu'))
    
    # second layer
    model.add(Dense(params['units_l2']))
    model.add(Dropout(params['dropout_l2']))
    model.add(Activation('relu'))

    if params['nbr_layers']['layers']== 'three':
        # third layer
        model.add(Dense(params['nbr_layers']['units_l3_a']))
        model.add(Dropout(params['nbr_layers']['dropout_l3_a']))
        model.add(Activation('relu'))

    if params['nbr_layers']['layers']== 'four':
        # third layer
        model.add(Dense(params['nbr_layers']['units_l3_b']))
        model.add(Dropout(params['nbr_layers']['dropout_l3_b']))
        model.add(Activation('relu'))
        # fourth layer
        model.add(Dense(params['nbr_layers']['units_l4']))
        model.add(Dropout(params['nbr_layers']['dropout_l4']))
        model.add(Activation('relu'))

    # final layers
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # loss  function and optimizers
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=params['optimizer'])

    result = model.fit(X_train, y_train,
              epochs=params['epochs'],
              verbose=0,
              validation_split=0.1)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.mean(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


best = 0
# Random Forest
def obj_func__RF(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train, y_train, _, _, _, _ = prepare_data(data, rseed)
        model = RandomForestClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# AdaBoost
def obj_func__AdaBoost(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train, y_train, _, _, _, _ = prepare_data(data, rseed)
        model = AdaBoostClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# XgBoost
def obj_func__XgBoost(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train, y_train, _, _, _, _ = prepare_data(data, rseed)
        model = xgboost.XGBClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
    


if __name__ == '__main__':

    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
    parser.add_argument('--model_class', type=str, default='DNN', help='DNN, RF, AdaBoost, XgBoost')
    parser.add_argument('--nbr_evals', type=int, default=25, help='Number of evaluations for hyperopt')

    # get input
    args = parser.parse_args()
    dataset = args.dataset
    rseed = args.rseed
    model_class = args.model_class
    nbr_evals = args.nbr_evals

    # create directory to save outputs
    outdir = './pretrained/{}/'.format(dataset)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    
    
    # filenames of saved objects
    model_name = '{}{}_{}.h5'.format(outdir, model_class, rseed)
    stats_name = '{}{}_{}.txt'.format(outdir, model_class, rseed)

    if model_class == 'DNN':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__DNN = partial(obj_func__DNN, data=dataset, rseed=rseed)
        best = fmin(obj_func__DNN, space_DNN, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_DNN, best)
        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
        acc_train = best_model.evaluate(X_train, y_train)[1]
        acc_test = best_model.evaluate(X_test, y_test)[1]
        acc_sg = best_model.evaluate(X_sg, y_sg)[1]

        # save the best model as bbox
        best_model.save(model_name)
        


        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Accuracy sg: {}\n'.format(acc_sg))
            myFile.write('Model params: {}\n'.format(best_params))
        
    if model_class == 'RF':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__RF = partial(obj_func__RF, data=dataset, rseed=rseed)
        best = fmin(obj_func__RF, space_RF, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_RF, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))
        acc_sg      = accuracy_score(y_sg, best_model.predict(X_sg))


        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))


        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Accuracy sg: {}\n'.format(acc_sg))
            myFile.write('Model params: {}\n'.format(best_params))

    if model_class == 'AdaBoost':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__AdaBoost = partial(obj_func__AdaBoost, data=dataset, rseed=rseed)
        best = fmin(obj_func__AdaBoost, space_AdaBoost, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_AdaBoost, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))
        acc_sg      = accuracy_score(y_sg, best_model.predict(X_sg))

        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))

        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Accuracy sg: {}\n'.format(acc_sg))
            myFile.write('Model params: {}\n'.format(best_params))
   
    if model_class == 'XgBoost':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__XgBoost = partial(obj_func__XgBoost, data=dataset, rseed=rseed)
        best = fmin(obj_func__XgBoost, space_XgBoost, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_XgBoost, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))
        acc_sg      = accuracy_score(y_sg, best_model.predict(X_sg))

        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))

        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Accuracy sg: {}\n'.format(acc_sg))
            myFile.write('Model params: {}\n'.format(best_params))