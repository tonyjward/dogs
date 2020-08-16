import numpy as np
import pandas as pd
import lightgbm as lgb
# import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from datetime import date, datetime, timezone
from dateutil.relativedelta import relativedelta
from dateutil import parser
import traceback

from dogs.modelling.utils import create_features, features_used, accuracy_error
from dogs.modelling.tuning import train_tune


def train_default(X_train, Y_train):
    """ Train a lightgbm model using the default settings

     Arguments:
        X_train - pandas dataframe with training features
        Y_train - pandas dataframe with training labels

    Returns:
        a trained lightgbm model
    """
    model = lgb.LGBMClassifier(objective = 'multiclass', num_class = "6", random_state = 42)
    model.fit(X_train, Y_train)
    return model


def train_cv(X_train, Y_train, nfold = 5, early_stopping_rounds = 20):
    """ 
    First trains a model using cross validation and early stopping to identify
    the optimal number of boosting iterations. Then trains a model using
    all of the data with the optimal number of boosting iterations

    Arguments:
        X_train - pandas dataframe with training features
        Y_train - pandas dataframe with training labels

    Returns:
        a trained lightgbm model
    """
    # model params
    params = {  "objective" : "multiclass",
            "num_class" : 6,
            "verbosity" : -1 }

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, Y_train)
    
    # cross validate to find optimal no of iterations
    r = lgb.cv(params, 
           lgb_train, 
           10000,
           early_stopping_rounds = early_stopping_rounds,
           nfold = nfold,
           feval = accuracy_error,
           metrics = 'None',
           verbose_eval = True,
           seed = 42)

    # Highest score
    r_best = np.max(r['accuracy-mean'])

    # best number of estimators
    best_estimators = np.argmax(r['accuracy-mean']) + 1
    print(best_estimators)

    print(f'The maxium accuracy on the validation set was {r_best:.5f}')
    print(f'The ideal number of iterations was {best_estimators}.')

    # Fit on all of the training data using the ideal number of iterations
    model = lgb.LGBMClassifier(n_estimators=best_estimators, n_jobs = -1,
                                       **params, random_state = 42)    
    model.fit(X_train, Y_train)

    return model


def modelling(df,
              training_months = 120,
              test_start = 2018,
              test_end = 2019,
              method = 'default',
              nfold = 5,
              max_evals = 200,
              feature_importance = False):

    start = timer()

    # define hold out periods
    years = list(range(test_start, test_end))
    months = list(range(1, 13))
    data_starts_at = df['date_time'].min()
    data_starts_at = parser.parse(data_starts_at)
    print(f'data starts at {data_starts_at}')
    
    # arrays to store accuracy indicator for all races across all time periods
    model_correct_all = np.array([])
    benchmark_correct_all = np.array([])

    # monthly summary statistics
    test_date = []
    model_accuracy = []
    benchmark_accuracy = []
    
    for year in years:
        for month in months:

            print(f"Building model for {year}:{month}")

            try:

                test_start = datetime(year, month,1, tzinfo = timezone.utc)
                test_end = test_start + relativedelta(months = +1)
                train_start = max(test_start + relativedelta(months = -training_months), data_starts_at)

                # partition data
                train_idx = (df.date_time >= str(train_start)) & (df.date_time < str(test_start))
                test_idx = (df.date_time >= str(test_start)) & (df.date_time < str(test_end))
                X_train, Y_train = create_features(df.loc[train_idx])
                X_test, Y_test = create_features(df.loc[test_idx])
                print('Train shape: ', X_train.shape)
                print('Test shape: ', X_test.shape)
                print(f"The following columns will be used in the modelling \n {X_train.columns.values}")

                # create dataset for lightgbm
                lgb_train = lgb.Dataset(X_train, Y_train)
                lgb_test = lgb.Dataset(X_test, Y_test, reference = lgb_train)

                # train model
                if method == 'default':
                    model = train_default(X_train, Y_train)
                    
                elif method == 'cv':
                    model = train_cv(X_train, Y_train, nfold = nfold)
                
                elif method == 'tune':
                    model = train_tune(X_train, Y_train, nfold = nfold, max_evals = max_evals)

                if feature_importance:
                    print(lgb.plot_importance(model))
                    
                # obtain predictions
                predictions = model.predict_proba(X_test)
                prediction_class = np.array([np.argmax(line) for line in predictions])

                # obtain benchmark
                benchmark = df.loc[test_idx, 'benchmark'].values

                # calculate monthly accuracy
                model_correct = (prediction_class == Y_test)
                benchmark_correct = (benchmark == Y_test)
                accuracy_model = model_correct.mean()
                accuracy_benchmark =  benchmark_correct.mean()

                # store for later
                model_correct_all = np.append(model_correct_all, model_correct)
                benchmark_correct_all = np.append(benchmark_correct_all, benchmark_correct)

                # print results
                print(f"Model correct: {accuracy_model:.3f} Benchmark correct {accuracy_benchmark:.3f}")
                
                # store results
                test_date.append(test_start)
                model_accuracy.append(accuracy_model)
                benchmark_accuracy.append(accuracy_benchmark)

            except Exception as Error:
                msg = f'''perform_modelling encountered an error for {year}: {month}'''
                print(msg)
                traceback.print_exc()

    # calculate results
    results = pd.DataFrame(list(zip(test_date, model_accuracy, benchmark_accuracy)), 
                    columns = ['test_date', 'model_accuracy', 'benchmark_accuracy'])
    results['columns_used'] = features_used(df.columns)
    model_accuracy_overall = model_correct_all.mean()
    benchmark_accuracy_overall = benchmark_correct_all.mean()
    no_eligible_races = len(model_correct_all)

    # display results
    print(results)
    print(f"We used the following features {features_used(df.columns)}")
    print(f"Overall Model Accuracy: {model_accuracy_overall:.3f} Overall Benchmark Accuracy: {benchmark_accuracy_overall:.3f}")

    total_time = timer() - start
    print(f'It took {total_time}')

    return model_accuracy_overall, benchmark_accuracy_overall, no_eligible_races, features_used(df.columns), total_time

if __name__ == '__main__':
    modelling()