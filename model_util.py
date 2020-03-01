#!/usr/bin/env python

"""
Project: Kaggle 2019 Data Science Bowl (DSB).
https://www.kaggle.com/c/data-science-bowl-2019

Utils related with model development. 

Author: Zhengyang Zhao
"""

from functools import partial
import copy

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit, KFold

from xgboost.sklearn import XGBRegressor 



def acc_to_group(accuracy, bound1=0.001, bound2=0.4, bound3=0.75):
    """
    Args
        accuracy: np array with elements between 0 and 1
    Return
        group: np array with element 0, 1, 2, or 3.
    
    Accurcy:   0    b1   0.33   b2   0.5   0.75   1
    Group:     0    |     1     |     2     |     3
    """
    group = accuracy.copy()
    for i in range(len(group)):
        acc = group[i]
        if acc > bound3:
            group[i] = 3
        elif acc > bound2:
            group[i] = 2
        elif acc > bound1:
            group[i] = 1
        else:
            group[i] = 0
    return group


def plot_acc(acc_truth, acc_pred, prefix, title=None):
    plt.figure(figsize=(5, 5))
    sns.regplot(acc_truth, acc_pred)
    plt.xlabel('{} Truth'.format(prefix))
    plt.ylabel('{} Predicted'.format(prefix))
    if title != None:
        plt.title(title)


def plot_acc_group_cm(acc_group_truth, acc_group_pred):
    cm = confusion_matrix(acc_group_truth, acc_group_pred, np.arange(4))
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Accuracy_group Prediction')
    plt.ylabel('Accuracy_group Truth')
    plt.xticks(np.arange(4) + 0.5, np.arange(4), rotation=0)
    plt.yticks(np.arange(4) + 0.5, np.arange(4), rotation=0)
    plt.ioff()
    
    
class CVSplitter(object):
    
    def __init__(self, n_splits, uid_list, Xy_df):
        self.n_splits = n_splits
        self.uid_list = uid_list
        self.Xy_df = Xy_df
        
        self.uid_split_df = pd.DataFrame({'uid': self.uid_list,
                                          'split_label': [0] * len(self.uid_list)
                                        }).set_index('uid')
        
    def _split_uid(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        split_label = 0
        for train_index, test_index in kf.split(self.uid_list):
            for uid in self.uid_list[test_index]:
                self.uid_split_df.loc[uid, 'split_label'] = split_label
            split_label += 1
        
    def _split_index(self):
        self.Xy_split_df = self.Xy_df.join(self.uid_split_df, on='installation_id')
        self.Xy_split_df.loc[self.Xy_split_df['is_last_assessment'] == 0, 'split_label'] = -1
        # if a row is not the last assessment of the uid, then always keep is in train split.
        
    def get_split(self):
        self._split_uid()
        self._split_index()
        return self.Xy_split_df
#         return PredefinedSplit(self.Xy_split_df['split_label'].values)
        
    
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    Modified based on:
    https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    
    def __init__(self, regression_output, brute_search=True, brute_search_steps=20):
        """
        Args:
            regression_output: string. Either 'accuracy' or 'accgroup'
            brute_search: boolean. If false, use scipy.optimize.minimize() to do search. Else, use brute force grid search.
            brute_search_step: int. Number of grid points along each axes.
        Note:
            The result of scipy.optimize.minimize() is highly relies on the initial value, and can easily return a local miminum.
            Therefore, using brute_search=True is a better option.
        """
        self.coef_ = 0
        self.group_count = 4
        self.regression_output = regression_output
        self.brute_search = brute_search
        self.brute_search_steps = brute_search_steps
        
        # params for brute search:
        self.lb = 0
        self.hb = self.group_count - 1
        if self.regression_output == 'accuracy':
            self.lb = 0
            self.hb = 1
        self.stepsize = self.hb / self.brute_search_steps
        self.brute_search_ranges = []
        for i in range(self.group_count - 1):
            self.brute_search_ranges += [(self.lb, self.hb)]
        
        # params for optimization search:
        self.initial_coef = [0.5, 1.5, 2.5]
        if self.regression_output == 'accuracy':
            self.initial_coef = [0.2, 0.4, 0.6]

        
    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        Args:
            coef: A list of coefficients that will be used for rounding. 
            X: The raw predictions
            y: The ground truth labels
        """
        for i in range(len(coef) - 1):
            if coef[i] >= coef[i + 1]:
                return 100   # Ensure coef is sorted and no duplicate elements
        y_p = np.array(pd.cut(X, [-np.inf] + list(coef) + [np.inf], labels = [0, 1, 2, 3]))
        qwk_score = cohen_kappa_score(y, y_p, weights='quadratic')
        return -qwk_score
    
    
    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        Args:
            X: The raw predictions
            y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
            
        if self.brute_search==False:
            self.coef_ = sp.optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')['x']
        else:
            self.coef_ = sp.optimize.brute(loss_partial, ranges=self.brute_search_ranges, Ns=self.brute_search_steps)
        
        
    def predict(self, X):
        """
        Make predictions with specified thresholds
        
        Args:
            X: The raw predictions
            coef: A list of coefficients that will be used for rounding
        """
        coef=self.coef_
        assert(len(coef) == self.group_count - 1)
        return np.array(pd.cut(X, [-np.inf] + list(coef) + [np.inf], labels = [0, 1, 2, 3]))

    
    def get_coefficients(self):
        """
        Return the optimized coefficients
        """
        assert(len(self.coef_) == self.group_count - 1)
        return self.coef_
    
    
    def set_coefficients(self, my_coef):
        """
        Arbitrarily set the coefficients
        """
        self.coef_ = np.array(my_coef)
    

class RegressorModel(object):
    """
    A wrapper class for regression models.
    It can be used for training and prediction.
    Can plot feature importance and training progress (if relevant for model).
    """

    def __init__(self, columns=None, model_wrapper=None):
        """
        Args: 
            columns (list): 
            model_wrapper: 
        """
        self.columns = columns
        self.model_wrapper = model_wrapper
        self.result_dict = {}
        self.train_one_fold = False
        self.preprocesser = None
        self.transformers = {}
        self.trained_transformers = {}
        self.scores = {}
        
        
    def save_model(self, save_path):
        joblib.dump(self, save_path, compress = 1)
        
        
    def transform_(self, datasets, cols_to_drop):
        """
        Do transformation on X before training.

        Args:
            datasets (dict): datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_holdout, 'y_train': y_train}
            cols_to_drop (list): columns in X that won't be used for training.
            
        Return: 
            X_train, X_valid, X_holdout.
        
        """
        for name, transformer in self.transformers.items():
            transformer.fit(datasets['X_train'], datasets['y_train'])
            datasets['X_train'] = transformer.transform(datasets['X_train'])
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = transformer.transform(datasets['X_valid'])
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = transformer.transform(datasets['X_holdout'])
            self.trained_transformers[name].append(transformer)
        if cols_to_drop is not None:
            cols_to_drop = [col for col in cols_to_drop if col in datasets['X_train'].columns]
            datasets['X_train'] = datasets['X_train'].drop(cols_to_drop, axis=1)
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = datasets['X_valid'].drop(cols_to_drop, axis=1)
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = datasets['X_holdout'].drop(cols_to_drop, axis=1)
        self.cols_to_drop = cols_to_drop

        return datasets['X_train'], datasets['X_valid'], datasets['X_holdout']
        

    def fit(self, X, y,
            X_holdout=None, y_holdout=None,
            train_one_fold=False, 
#             folds=None,
            params=None,
            eval_metric='rmse',
            cols_to_drop=None,
            preprocesser=None,
            transformers=None,
            adversarial=False,
            plot=True,
            plot_title=None,
            verbose=1):
        
        """
        Training the model.

        Args:
            X (pd.DataFrame), y: training data. 
                Note: if train_one_fold == False, X should contain a column 'split_label', where the folds is predefined.
                split_label == k (k >= 0) means this data would be put into the validation set on the k-th fold;
                split_label == -1 means this data would be always be hold in train set.
            X_holdout (pd.DataFrame), y_holdout: data for holdout validation. This part of data is excluded from X and y.
            train_one_fold (bool): if true, train on the whole (X, y). 
            folds: folds to split the data. If not defined, then model will be trained on the whole X + X_holdin.
            params (dict): training parameters. Including hyperparameters and:
                params['objective'] (str): 'regression' or 'classification',
                params['verbose'] (bool),
                params['cat_cols'] (list): categorical_columns, only used in LGB and CatBoost wrappers.
                params['early_stopping_rounds'] (int).
            eval_metric (str): metric for validataion.
            cols_to_drop (list): list of columns to drop (for example ID).
            preprocesser: preprocesser class.
            transformers (dict): transformer to use on folds.
            adversarial (bool): to do. If true, do adversarial validation.
            plot (bool): if true, plot 'feature importance', 'training curve', 'distribution of prediction', 'distribution of error'.
        
        Note:
            fit() function contains the following steps:
                1. do preprocessing on X and X_holdout, if preprocesser is not None.
                2. do the for loop for fords.
                3. In each loop, first split X_train, y_train, X_valid, y_valid, X_holdout, y_holdout.
                4. then do transformation on X_train, X_valid, X_holdout (transformation includes column selection).
                5. model.fit(X_train, y_train, X_valid, y_valid, X_hold, y_holdout, params=params)
                6. record the prediction, trained model, feature_importance of current fold.
                7. After all folds finished, plotting results.
        """

        if train_one_fold == True:
            folds = KFold(n_splits=3, random_state=42) # actually (X, y) won't be split in the for loop.
            self.train_one_fold = True
        else:
            assert('split_label' in X.columns)
            folds = PredefinedSplit(X['split_label'].values)

#         self.columns = X.columns if self.columns is None else self.columns
        self.feature_importances = pd.DataFrame(columns=['feature', 'gain'])
        if transformers is not None:
            self.trained_transformers = {k: [] for k in transformers} # the value_list store the trained_transformer from each fold.
            self.transformers = transformers
        self.models = []
        self.folds_dict = {}
        self.eval_metric = eval_metric
        n_target = 1
        self.oof = np.empty((len(X), n_target))   # OOF simply stands for "Out-of-fold" approach.
        self.oof[:] = np.NaN
        self.n_target = n_target
        self.verbose = verbose

        if preprocesser is not None:
            self.preprocesser = preprocesser
            self.preprocesser.fit(X, y)
            X = self.preprocesser.transform(X, y)
            self.columns = X.columns.to_list()
            if X_holdout is not None:
                X_holdout = self.preprocesser.transform(X_holdout)
                
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
            if adversarial:
                pass
            X_train, X_valid = X.iloc[train_index].copy(), X.iloc[valid_index].copy()
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            if self.train_one_fold:
                X_train = X.copy()
                y_train = y
                X_valid = None
                y_valid = None
            X_hold = (X_holdout.copy() if X_holdout is not None else None)
            
            if self.verbose > 1:
                print(f'Fold {fold_n} started.')

            datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_hold, 'y_train': y_train}
            X_train, X_valid, X_holdout = self.transform_(datasets, cols_to_drop)
            self.columns = X_train.columns.to_list()
            
            model = copy.deepcopy(self.model_wrapper)
            model.fit(X_train, y_train, X_valid, y_valid, X_holdout, y_holdout, params=params)
            
            self.folds_dict[fold_n] = {}
            self.folds_dict[fold_n]['columns'] = X_train.columns.tolist()
            self.folds_dict[fold_n]['scores'] = model.best_score_
            self.oof[valid_index] = model.predict(X_valid).reshape(-1, n_target)
            
            fold_importance = pd.DataFrame({
                                    'feature': X_train.columns,
                                    'gain': model.feature_importances_
                                })
            self.feature_importances = self.feature_importances.append(fold_importance)
            self.models.append(model)

        self.calc_scores_()
        
        if plot:
            # print(classification_report(y, self.oof.argmax(1)))
            fig, ax = plt.subplots(figsize=(20, 6))
            plt.subplot(1, 4, 1)
            self.plot_feature_importance(top_n=20)
            plt.subplot(1, 4, 2)
            self.plot_learning_curve()
            plt.subplot(1, 4, 3)
            errors = y.values.reshape(-1, 1) - self.oof
            errors = errors[~np.isnan(errors)]
            plt.hist(errors)
            plt.title('Distribution of errors')
            plt.subplot(1, 4, 4)
            plt.hist(self.oof[~np.isnan(self.oof)])
            plt.title('Distribution of oof predictions');
            plt.suptitle(plot_title)
    
    
    def predict(self, X_test, averaging='usual'):
        """
        Make prediction

        Args:
            X_test (pd.DataFrame): test data
            averaging: method of averaging
            
        Return:
            list: prediction of X_test
        """
        
        full_prediction = np.zeros((X_test.shape[0], self.oof.shape[1]))
        if self.preprocesser is not None:
            X_test = self.preprocesser.transform(X_test)
        for i in range(len(self.models)):
            X_t = X_test.copy()
            for name, transformers in self.trained_transformers.items():
                X_t = transformers[i].transform(X_t)
            if self.cols_to_drop is not None:
                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]
                X_t = X_t.drop(cols_to_drop, axis=1)
            y_pred = self.models[i].predict(X_t[self.folds_dict[i]['columns']]).reshape(-1, full_prediction.shape[1])

            if averaging == 'usual':
                full_prediction += y_pred
            elif averaging == 'rank':
                full_prediction += pd.Series(y_pred).rank().values

        return full_prediction / len(self.models)
        

    def calc_scores_(self):
        datasets = [k for k, v in [v['scores'] for k, v in self.folds_dict.items()][0].items() if len(v) > 0]
        for d in datasets:
            scores = [v['scores'][d][self.eval_metric] for k, v in self.folds_dict.items()]
            if self.verbose:
                print(f"CV mean score on {d}: {np.mean(scores):.3f} +/- {np.std(scores):.3f} std.")
            self.scores[d] = np.mean(scores)


    def plot_feature_importance(self, drop_null_importance=True, top_n=20):
        """
        Plot feature importance.

        Args:
            drop_null_importance (bool): drop columns with null feature importance
            top_n (int): show top n features.
        """

        top_feats = self.get_top_features(drop_null_importance, top_n)
        feature_importances = self.feature_importances.loc[self.feature_importances['feature'].isin(top_feats)]
        feature_importances['feature'] = feature_importances['feature'].astype(str)
        top_feats = [str(i) for i in top_feats]
        sns.barplot(data=feature_importances, x='gain', y='feature', orient='h', order=top_feats)
        plt.title('Feature importances')

    
    def get_top_features(self, drop_null_importance=True, top_n=20):
        """
        Get top features by importance.
        
        Args:
            drop_null_importance (bool): drop columns with null feature importance
            top_n (int): show top n features.
        """
        
        grouped_feats = self.feature_importances.groupby(['feature'])['gain'].mean()  # average over folds.
        if drop_null_importance:
            grouped_feats = grouped_feats[grouped_feats != 0]
        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]

    
    def plot_learning_curve(self):
        """
        Plot training learning curve.
        Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html
        
        An example of model.evals_result_: 
            {
                'validation_0': {'rmse': [0.259843, 0.26378, 0.26378, ...]},
                'validation_1': {'rmse': [0.22179, 0.202335, 0.196498, ...]}
            }
            
            'validation_0' represent train set;
            'validation_1' represent validation set;
        """
        
        full_evals_results = pd.DataFrame()
        for model in self.models:
            evals_result = pd.DataFrame()
            for k in model.model.evals_result_.keys():  # iterate through different sets.
                evals_result[k] = model.model.evals_result_[k][self.eval_metric]
            evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
            full_evals_results = full_evals_results.append(evals_result)

        full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,
                                                                                            'variable': 'dataset'})
        sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')
        plt.title('Train Learning-Curve')
        
    
#     def hyperparams_select(self, X, y, cols_to_drop,
#                            hyperparam_grid, 
#                            scoring,
#                            folds,
#                            gs_res_path,
#                            gs_hyperparam_path):
#         """
#         Args
#             X, y: data used for GridSearchCV.
#             cols_to_drop: cols in X_train to drop.
#             hyperparam_grid: dictionary.
#             scoring: GridSearchCV scoring.
#             folds: GridSearchCV cv. int or PredefinedSplit object.
#             gs_res_path: string, path to save the GridSearchCV csv results.
#             gs_hyperparam_path: string, path to save the best hyperparameters.
            
#         Return
#             gs_best_hyperparam: best hyperparameters.
#         """
#         gs = GridSearchCV(self.model_wrapper.model, 
#                           param_grid=hyperparam_grid, 
#                           scoring=scoring, 
#                           cv=folds, 
#                           n_jobs=cpu_count()//2, refit=False, verbose=5) 
#         X_gs = self.drop_cols_(X, cols_to_drop)
#         y_gs = y
#         gs.fit(X_gs, y_gs) 
#         print(">>>>>>> XGBoost HPO best cross-validation score:", gs.best_score_) 
#         print(">>>>>>> XGBoost HPO best params:")
#         print(gs.best_params_) 

#         # save GridSearch results df
#         gs_res = gs.cv_results_
#         gs_res_df = pd.DataFrame.from_dict(gs_res)
#         gs_res_df.to_csv(gs_res_path)

#         # save the best hyperparams
#         gs_best_hyperparam = gs.best_params_
#         gs_best_hyperparam.update({'n_jobs': cpu_count()//2}) 
#         joblib.dump(gs_best_hyperparam, gs_hyperparam_path, compress = 1)
#         self.hyperparams = gs_best_hyperparam
#         return gs_best_hyperparam
        
        
#     def hyperparams_set(self, hyperparams):
#         """
#         Set model hyperparameters.
        
#         Args
#             hyperparams: dict (hyperparameters) or string (hyperparameters path).
#         """
#         if type(hyperparams) == str:
#             assert(hyperparams[-4:] == '.pkl')
#             hyperparams = joblib.load(hyperparams)
#         self.model_wrapper.model.set_params(**hyperparams)
#         self.hyperparams = hyperparams
        
    
#################################################################
# Model Wrappers.
#################################################################

def eval_qwk_xgb(y_pred, y_true):
    """
    Fast cappa eval function for xgb.
    """
    # print('y_true', y_true)
    # print('y_pred', y_pred)
    y_true = y_true.get_label()
    y_pred = y_pred.argmax(axis=1)
    return 'cappa', -qwk(y_true, y_pred)


def eval_qwk_lgb(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """

    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3

    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True


class XGBWrapper_regr(object):
    """
    A wrapper for xgboost model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = XGBRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        self.model = self.model.set_params(**params)
        
        eval_set = [(X_train, y_train)]
        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_metric='rmse',
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])

        scores = self.model.evals_result()
        self.best_score_ = {k: {m: m_v[-1] for m, m_v in v.items()} for k, v in scores.items()}
#         self.best_score_ = {k: {m: n if m != 'cappa' else -n for m, n in v.items()} for k, v in self.best_score_.items()}

        self.feature_importances_ = self.model.feature_importances_
    
    def predict(self, X_test):
        return self.model.predict(X_test)

#     def predict_proba(self, X_test):
#         if self.model.objective == 'binary':
#             return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)[:, 1]
#         else:
#             return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)


class LGBWrapper_regr(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        if params['objective'] == 'regression':
            eval_metric = 'rmse'
        else:
            eval_metric = 'auc'

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)

    
class LGBWrapper(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = lgb.LGBMClassifier()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict_proba(self, X_test):
        if self.model.objective == 'binary':
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)


class CatWrapper(object):
    """
    A wrapper for catboost model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = cat.CatBoostClassifier()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_set = [(X_train, y_train)]
        self.model = self.model.set_params(**{k: v for k, v in params.items() if k != 'cat_cols'})

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = None
        else:
            categorical_columns = None
        
        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       cat_features=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict_proba(self, X_test):
        if 'MultiClass' not in self.model.get_param('loss_function'):
            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)


