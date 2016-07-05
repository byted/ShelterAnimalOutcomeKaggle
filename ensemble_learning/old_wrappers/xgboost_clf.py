import random
import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
import os
import sys


def _shuf(train_X, train_y):
    random.seed(1)
    index_shuf = list(range(len(train_X)))
    random.shuffle(index_shuf)
    train_X = np.array([train_X[i] for i in index_shuf])
    train_y = np.array([train_y[i] for i in index_shuf])
    return (train_X, train_y)


def _split(train_X, train_y, verify_percent=0.01):
    ten_percent = int(verify_percent * len(train_X))
    learn_train_X, verify_train_X = train_X[ten_percent:], train_X[:ten_percent]
    learn_train_y, verify_train_y = train_y[ten_percent:], train_y[:ten_percent]
    if len(np.unique(verify_train_y)) < 5:
        raise Exception("we dont have samples for every class {}".format(np.bincount(verify_train_y)))
    return (learn_train_X, learn_train_y, verify_train_X, verify_train_y)


def _cv_score(clf, X, y, train_ix, test_ix, fit_param):
        clf = _silent_fit(clf, X[train_ix], y[train_ix], fit_param)
        y_pred = clf.predict_proba(X[test_ix], ntree_limit=clf.booster().best_ntree_limit)
        return metrics.log_loss(y[test_ix], y_pred)


def _silent_fit(clf, X, y, fit_param):
    '''Wrapper to silence xgboost - silent parameter is not working'''
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        clf.fit(X, y=y, **fit_param)
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr
    return clf


def _init_xgb(train_X, train_y):
    # do this as soon as the module gets loaded
    train_X, train_y = _shuf(train_X, train_y)
    learn_train_X, learn_train_y, verify_train_X, verify_train_y = _split(train_X, train_y)

    clf_param = {
        'objective': 'multi:softprob',
        'learning_rate': 0.35,     # feature weight shrinkrage: lower -> reduce complexity
        'gamma': 0.00,   # loss reduction needed to split a leaf and add another layer: higher -> reduces complexity
        'max_depth': 8,  # maximum depth of a tree: higher -> increase complexity
        # 'min_child_weight': 12,  # minimum number of instances in a leaf: higher -> less complexity
        'max_delta_step': 1,    # in logregression a higher value might help with imbalanced classes
        'subsample': 0.5,       # randomly select instances from train set to build trees: lower -> less overfitting
        'colsample_bytree': 0.5,  # subsample colums: lower -> less overfitting
        'scale_pos_weight': 0,  # fight unbalanced classes: sum(negative cases) / sum(positive cases)
        'n_estimators': 3000
    }
    fit_param = {'eval_metric': 'mlogloss', 'eval_set': [(verify_train_X, verify_train_y)], 'early_stopping_rounds': 50}

    # sklearn interface
    clf = xgb.XGBClassifier(silent=True, **clf_param)

    return clf, fit_param, learn_train_X, learn_train_y, verify_train_X, verify_train_y


# public


def score(train_X, train_y, folds=3):
    clf, fit_param, learn_train_X, learn_train_y, _, _ = _init_xgb(train_X, train_y)
    kfold = cross_validation.StratifiedKFold(learn_train_y, n_folds=folds, random_state=0, shuffle=True)
    scores = [_cv_score(clf, learn_train_X, learn_train_y, train, test, fit_param) for train, test in kfold]
    return np.mean(scores)


def predict(train_X, train_y, test_X):
    clf, fit_param, _, _, _, _ = _init_xgb(train_X, train_y)
    _silent_fit(clf, train_X, train_y, fit_param)
    return clf.predict_proba(test_X, ntree_limit=clf.booster().best_ntree_limit)


def get_clf(train_X, train_y):
    clf, fit_param, _, _, _, _ = _init_xgb(train_X, train_y)
    return clf, fit_param
