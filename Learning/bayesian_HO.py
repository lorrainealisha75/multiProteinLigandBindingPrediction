#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script uses Bayesian optimization to find the 
best values for the LGBM classifier hyperparameters.
"""
import pandas as pd
import numpy as np
import lightgbm as lgbm
import argparse

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import KFold


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


def main(args):

    #Read in entire csv file
    pcba_with_rdkit_df = pd.read_csv(args.dpath,  sep=',', header=None, error_bad_lines=False)

    #Replace all infinity values to NaN
    pcba_with_rdkit_df.replace(to_replace=[np.inf], value=np.nan, inplace=True)

    X = pcba_with_rdkit_df.iloc[:, 0:450]
    y = pcba_with_rdkit_df.iloc[:, 450]

    #Impute NaN values with column median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = imputer.fit_transform(X, y)

    # Create a pandas df again from the output of the imputer
    X = pd.DataFrame(data=X_imputed[0:, 0:])

    bayes_cv_tuner = BayesSearchCV(
        estimator=lgbm.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt'),
        search_spaces={
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'num_leaves': Integer(100, 300),
            'max_depth': Integer(100, 300),
            'min_child_samples': Integer(0, 100),
            'bagging_freq': Integer(0, 20),
            'min_child_weight': Integer(0, 20),
            'lambda_l1': Real(0, 50),
            'max_bin': Integer(100, 500),
            'num_iterations': Integer(200, 500),
            'num_threads': Integer(10, 200)
        },
        scoring='neg_mean_squared_log_error',
        #scoring='roc_auc',
        cv=KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),
        n_jobs=10,
        n_iter=30,
        verbose=0,
        refit=True,
        random_state=42
    )

    print("\nFit the model\n")
    # Fit the model
    result = bayes_cv_tuner.fit(X, y, callback=status_print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True, help='Path to the final created dataset to be used for BO')

    args = parser.parse_args()
    main(args)