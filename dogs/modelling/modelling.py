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

from greyhound.modelling.utils import create_features, features_used

def modelling(data_path = '/home/d14xj1/repos/greyhound/modelling_data',
              file_name = 'modelling_data',
              training_months = 120,
              test_start = 2017,
              test_end = 2019,
              method = 'default'):

    start = timer()

    years = list(range(test_start, test_end))
    months = list(range(1, 13))

    # read data
    df = pd.read_csv(os.path.join(data_path, file_name + '.csv'))

    # target must start from 0
    df['winning_box'] = df['winning_box'] - 1
    df['favourite'] = df['favourite'] - 1
    
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
                    model = lgb.LGBMClassifier(objective = 'multiclass', num_class = "6", random_state = 42)
                    model.fit(X_train, Y_train)
                elif method == 'cv':
                    params = {  "objective" : "multiclass",
                                "num_class" : 6,
                                "verbosity" : -1 }
                    
                    # cross validate to find optimal no of iterations
                    r = lgb.cv(params, 
                            lgb_train, 
                            10000,
                            early_stopping_rounds = 100,
                            nfold = 5,
                            metrics = 'multi_logloss',
                            verbose_eval = True,
                            seed = 42)
                    
                    # Highest score
                    r_best = np.min(r['multi_logloss-mean'])

                    # Standard deviation of best score
                    r_best_std = r['multi_logloss-stdv'][np.argmin(r['multi_logloss-mean'])]

                    # best number of estimators
                    best_estimators = np.argmin(r['multi_logloss-mean']) + 1

                    print('The minimum multi-log_loss on the validation set was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
                    print(f'The ideal number of iterations was {best_estimators}.')

                    model = lgb.LGBMClassifier(n_estimators=best_estimators, n_jobs = -1, **params, random_state = 42)
                    # Fit on the training data
                    model.fit(X_train, Y_train)

                # obtain predictions
                predictions = model.predict_proba(X_test)
                prediction_class = np.array([np.argmax(line) for line in predictions])

                # obtain benchmark
                favourite = df.loc[test_idx, 'favourite'].values

                # calculate monthly accuracy
                model_correct = (prediction_class == Y_test)
                benchmark_correct = (favourite == Y_test)
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
    # results.to_csv(os.path.join(data_path, 'runs.csv'), mode = 'a', index = False)

    total_time = timer() - start
    print(f'It took {total_time}')

    return model_accuracy_overall, benchmark_accuracy_overall, no_eligible_races, features_used(df.columns)

if __name__ == '__main__':
    modelling()