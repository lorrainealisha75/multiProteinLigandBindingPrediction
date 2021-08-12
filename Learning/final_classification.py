#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script uses the final dataset created after random undersampling
for classifcation wih LGBM classifier
"""
import pandas as pd
import numpy as np

import csv
import gc
import argparse

import lightgbm as lgbm
from sklearn.ensemble import GradientBoostingClassifier

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score


#Data Cleaning

#Simple Imputer
def get_simple_imputer():
    return SimpleImputer(missing_values=np.nan, strategy='median')

def process_feature_importance(feature_import):
    # Sort features according to importance
    feature_import = feature_import.sort_values('importance', ascending=False).reset_index(drop=True)

    # Normalize the feature importances to add up to one
    feature_import['normalized_importance'] = feature_import['importance'] / feature_import[
        'importance'].sum()
    return feature_import


def zero_importance_features(feature_importance):
    zero_imp_features = feature_importance[feature_importance['importance'] == 0.0]
    return set(zero_imp_features['feature'])


def low_importance_feature(cumulative_imp, feature_importance):

    # Make sure most important features are on top
    feature_importance = feature_importance.sort_values('cumulative_importance')

    # Identify the features not needed to reach the cumulative_importance
    low_importance_features = feature_importance[feature_importance['cumulative_importance'] > cumulative_imp]

    return set(low_importance_features['feature'])


def remove_columns(X, columns):
    # Get the indices for the given column names
    to_drop_index = [X.columns.get_loc(col) for col in columns]

    # Drop relevant columns
    X.drop(X.columns[to_drop_index], axis=1, inplace=True)
    return X

def get_gb_classifier_model(X, y):
    
    print("\nStarting GBM. Train shape: {}\n\n".format(X.shape))
    gc.collect()

    # Cross validation model
    folds = StratifiedKFold(n_splits=5, shuffle=True)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(X.shape[0])
    auc_score = {}

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx, :], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx, :], y.iloc[valid_idx]

        # GBM parameters
        clf = GradientBoostingClassifier(n_estimators=192, learning_rate=0.41, max_features=93, max_depth=37, random_state=0)

        clf.fit(train_x, train_y)

        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
       
        score = roc_auc_score(valid_y, oof_preds[valid_idx])
        print('\nFold %2d AUC : %.6f\n' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        
        auc_score[score] = clf

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('\nFull AUC score %.6f\n\n' % roc_auc_score(y, oof_preds))

    max_auc = max(list(auc_score.keys()))

    return auc_score[max_auc]

def use_lgbm(X, y, return_classifier):

    feats = X.columns #[f for f in X.columns if f not in ['label']]

    print("Starting LightGBM. Train shape: {}".format(X.shape))
    gc.collect()
    
    #Dict for classifier and corresponding feature importance
    auc_score = {}

    # Cross validation model
    folds = StratifiedKFold(n_splits=5, shuffle=True)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(X.shape[0])
    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx, :], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx, :], y.iloc[valid_idx]        
                
        # LightGBM parameters found by Bayesian optimization
        clf = lgbm.LGBMClassifier(
            task='train',
            objective='binary',
            boosting='gbdt',
            metric='binary_logloss',
            num_threads=164,
            feature_frac=0.5,
            bagging_frac=0.4,
            learning_rate=0.046,
            num_leaves=300,
            max_depth=231,
            max_bin=100,
            min_child_weight=17,
            min_child_samples=100,
            num_iterations=500,
            early_stopping_rounds=50,
            bagging_freq=20
            ) 
        '''  
        LightGBM parameters before optimization   
        clf = lgbm.LGBMClassifier(
            task='train',
            objective='binary',
            boosting='gbdt',
            metric='binary_logloss',
            learning_rate=0.01,
            num_leaves=150,
            max_depth=130,
            max_bin=40,
            #is_unbalance=True,
            num_iterations=200,
        )
        '''                 
        
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        score = roc_auc_score(valid_y, oof_preds[valid_idx])
        print('\n\nFold %2d AUC : %.6f' % (n_fold + 1, score))
        print('\n')
        if return_classifier:
            auc_score[score] = clf
        else:
            auc_score[score] = fold_importance_df["importance"]
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('\nFull AUC score %.6f' % roc_auc_score(y, oof_preds))
    print('\n\n')

    max_auc = max(list(auc_score.keys()))

    return auc_score[max_auc]

def remove_collinear_features(X):
    correlation_matrix = X.corr()

    correlation_matrix_with_avg = correlation_matrix.copy()

    # Add a new column with the row-wise average
    correlation_matrix_with_avg['average'] = correlation_matrix_with_avg.mean(numeric_only=True, axis=1)

    columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)

    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[0]):
            # If the collinearity coefficient is too high, remove one of the features
            if correlation_matrix.iloc[i, j] >= 0.9 and columns[j]:
                # Remove the feature that has a higher collinearity coefficient with all other features
                if correlation_matrix_with_avg.iloc[j]['average'] > correlation_matrix_with_avg.iloc[i]['average']:
                    columns[j] = False
                else:
                    columns[i] = False

    selected_columns = X.columns[columns]

    return X[selected_columns], correlation_matrix

def print_prediction_metrics(y_true, y_pred):
    print("\nAccuracy is: %.3f" %(accuracy_score(y_true, y_pred)))

    print("\nPrecision is: %.3f" %(precision_score(y_true, y_pred)))

    print("\nRecall is: %.3f" %(recall_score(y_true, y_pred)))

    print("\nF1 score is: %.3f" %(f1_score(y_true, y_pred)))

    
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title("Confusion Matrix")
    plt.savefig('cm_pngs/fs_0.9.png')


def main(args):

    #Read in entire csv file
    pcba_with_rdkit_df = pd.read_csv(args.dpath,  sep=',', header=None, error_bad_lines=False)
    print(pcba_with_rdkit_df.shape)

    #Replace all infinity values to NaN
    pcba_with_rdkit_df.replace(to_replace=[np.inf], value=np.nan, inplace=True)

    X = pcba_with_rdkit_df.iloc[:, 0:450]
    y = pcba_with_rdkit_df.iloc[:, 450]

    #Impute NaN values with column median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = imputer.fit_transform(X, y)

    # Create a pandas df again from the output of the imputer
    X = pd.DataFrame(data=X_imputed[0:, 0:])

    counter = Counter(y)
    print('\n\nDistribution of classes')
    print(counter)
    print('\n\n')

    feature_importance_values = use_lgbm(X, y, False)

    # Extract feature names
    feature_names = list(X.columns)

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    feature_importances = process_feature_importance(feature_importances)
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

    # Extract the features with zero importance
    record_zero_importance = zero_importance_features(feature_importances)

    # List of all columns to be removed from the dataset
    columns_to_drop = set()

    # Add zero importance features to the list of columns to be removed
    columns_to_drop.update(record_zero_importance)

    cumulative_importance = 0.9

    # Extract the features with low importance
    record_low_importance = low_importance_feature(cumulative_importance, feature_importances)

    # Add low importance features to the list of columns to be removed
    columns_to_drop.update(record_low_importance)

    X = remove_columns(X, columns_to_drop)

    # Remove features that display high collinear coefficient
    X, corr_mat = remove_collinear_features(X)

    print("\nNumber of columns after feature selection: " + str(len(X.columns)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

    counter = Counter(y_test)
    print('\n\nDistribution of test set classes')
    print(counter)
    print('\n\n')

    # Fit the training data on a Gradient Boost classifier
    classifier_model = use_lgbm(X_train, y_train, True)
    #classifier_model = get_gb_classifier_model(X_train, y_train)

    y_pred = classifier_model.predict(X_test)

    oof_preds = classifier_model.predict_proba(X_test)

    counter = Counter(y_pred)
    print('Distribution of predicted classes')
    print(counter)
    print('\n\n')

    # Get Precision, Recall, f1 score and confusion matrix
    print_prediction_metrics(y_test, y_pred)

    print('\nLog loss score: %.3f\n' % (log_loss(y_test, oof_preds[:, 1])))
    print('\nAUC score: %.3f\n' % (roc_auc_score(y_test, oof_preds[:, 1])))

    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True, help='Path to the final created dataset to be used for classification')

    args = parser.parse_args()
    main(args)
