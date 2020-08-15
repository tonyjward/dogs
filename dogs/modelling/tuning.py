# Evaluation of the model
from sklearn.model_selection import KFold

import csv
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import numpy as np
import pandas as pd
import os
import lightgbm as lgb

from timeit import default_timer as timer

import ast

from dogs.modelling.utils import accuracy_error


MAX_EVALS = 200

# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 #{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}


def train_tune(X_train, Y_train, nfold, early_stopping_rounds = 20, max_evals = 200):
    """ 
    Perform hyperparmater tuning using cross validation and bayesian optimisation
    to find the set of parmaters and boosting iterations that maximises hold out  
    accuracy.  
    Then trains a model using all of the training data with the optimal 
    hyperparameters and boosting iterations

    Arguments:
        X_train - pandas dataframe with training features
        Y_train - pandas dataframe with training labels

    Returns:
        a trained lightgbm model
    """

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, Y_train)

    # optimization algorithm
    tpe_algorithm = tpe.suggest

    # Keep track of results
    bayes_trials = Trials()

    # store tuning results file
    out_file = 'gbm_trials.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
    of_connection.close()

    # Global variable
    global  ITERATION

    ITERATION = 0
    
    def objective(params, data = lgb_train, n_folds = nfold, early_stopping_rounds = early_stopping_rounds, max_evals = max_evals):
        """
        Objective function for Gradient Boosting Machine Hyperparameter Optimization
        We use accuracy as a custom evaluation function
        """
        
        # Keep track of evals
        global ITERATION
        
        ITERATION += 1
        
        # Retrieve the subsample if present otherwise set to 1.0
        subsample = params['boosting_type'].get('subsample', 1.0)
        
        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        params['subsample'] = subsample
        
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            params[parameter_name] = int(params[parameter_name])
        
        start = timer()
        
        params['num_class'] = 6
        params["objective"] = "multiclass"
        
        # Perform n_folds cross validation
        cv_results = lgb.cv(params, data, num_boost_round = 10000, nfold = n_folds, 
                            early_stopping_rounds = early_stopping_rounds, feval = accuracy_error, metrics = 'None', seed = 42)
        
        run_time = timer() - start
        
        # Extract the best score
        best_score = np.max(cv_results['accuracy-mean'])
        loss = 1- best_score
        
        # Boosting rounds that returned the highest cv score
        n_estimators = int(np.argmax(cv_results['accuracy-mean']) + 1)

        # Write to the csv file ('a' means append)
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, ITERATION, n_estimators, run_time])
        
        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'iteration': ITERATION,
                'estimators': n_estimators, 
                'train_time': run_time, 'status': STATUS_OK}  
    
    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, 
                max_evals = max_evals, trials = bayes_trials, rstate = np.random.RandomState(42))

    # grab results from file
    results = pd.read_csv('gbm_trials.csv')

    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending = True, inplace = True)
    results.reset_index(inplace = True, drop = True)

    # Convert from a string to a dictionary
    print("The best hyperparms are")
    ast.literal_eval(results.loc[0, 'params'])

    # Extract the ideal number of estimators and hyperparameters
    best_bayes_estimators = int(results.loc[0, 'estimators'])
    best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
    print(best_bayes_params)

    # Re-create the best model and train on the training data
    model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1, random_state = 42, **best_bayes_params)
    # Fit on the training data
    model.fit(X_train, Y_train)

    return model



