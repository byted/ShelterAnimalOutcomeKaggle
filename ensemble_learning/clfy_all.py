import data
import pandas as pd
import numpy as np
import sys, os, json, argparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.metrics import log_loss
import xgboost as xgb
from datetime import datetime


def _filter_features(X, features_to_use):
    return X[features_to_use].as_matrix()


def to_kaggle(n, predictions, test_df, train_df, le):
    '''ugly, somehow this code survived from the beginning'''
    submission_df = pd.DataFrame(predictions)
    submission_df = pd.concat([test_df.ID, submission_df], axis=1)
    submission_df.ID = submission_df.ID.astype(int)
    submission_df.set_index('ID', inplace=True)
    submission_df.columns = le.inverse_transform(sorted(train_df.OutcomeType.unique()))  # get the string labels back
    submission_df.to_csv(n)


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

parser = argparse.ArgumentParser()
parser.add_argument('--n_folds', '-n', help='Number of folds',
                    default=3, type=int)
parser.add_argument('--random_state', '-r', help='The random state',
                    default=0, type=int)
args = parser.parse_args()

foldername = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(foldername)
params = {
    'XGB': {
        'objective': 'multi:softprob',
        'learning_rate': 0.35,     # feature weight shrinkrage: lower -> reduce complexity
        'gamma': 0.00,   # loss reduction needed to split a leaf and add another layer: higher -> reduces complexity
        'max_depth': 8,  # maximum depth of a tree: higher -> increase complexity
        # 'min_child_weight': 12,  # minimum number of instances in a leaf: higher -> less complexity
        'max_delta_step': 1,    # in logregression a higher value might help with imbalanced classes
        'subsample': 0.5,       # randomly select instances from train set to build trees: lower -> less overfitting
        'colsample_bytree': 0.5,  # subsample colums: lower -> less overfitting
        'scale_pos_weight': 0,  # fight unbalanced classes: sum(negative cases) / sum(positive cases)
        'n_estimators': 2000
    },
    'BNB': {'binarize': False, 'fit_prior': True, 'alpha': 0.7},
    'LogReg': {'C': 0.9, 'penalty': 'l1', 'fit_intercept': True, 'tol': 0.000001},
    'LDA': {}
}
features = {
    'LogReg': ['AnimalType', 'Sex', 'Intact', 'Breed', 'AgeuponOutcomeInDays'],
    'BNB': ['AnimalType', 'Sex', 'Intact', 'Breed', 'AgeuponOutcomeInYears'],
    'XGB': ['AnimalType', 'Sex', 'SexuponOutcome', 'AgeuponOutcomeInDays'],
    'LDA': ['AnimalType', 'Sex', 'Intact', 'Breed', 'AgeuponOutcomeInDays']
}

clfs = [
    ('LogReg', OneVsRestClassifier(LogisticRegression(**params['LogReg'])), features['LogReg']),
    ('BNB', BernoulliNB(**params['BNB']), features['BNB']),
    ('XGB', xgb.XGBClassifier(**params['XGB']), features['XGB']),
    ('LDA', LDA(**params['LDA']), features['LDA'])
]

test_X, y, submission_X, test_df, train_df, le = data.get_data()
clfs = [(n, c, _filter_features(test_X, f), _filter_features(submission_X, f)) for n, c, f in clfs]
skf = list(StratifiedKFold(y, args.n_folds, shuffle=True, random_state=args.random_state))

stats_per_clf = []

for j, (name, clf, X, X_submission) in enumerate(clfs):
    print(name)
    for i, (train, test) in enumerate(skf):
        # k-fold CV
        print('Fold', i)
        kfold_test_acc, kfold_subm_y_preds = [], []
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        if name == 'XGB':
            fit_param = {'eval_metric': 'mlogloss', 'eval_set': [(X_test, y_test)], 'early_stopping_rounds': 50}
            clf = _silent_fit(clf, X_train, y_train, fit_param)
        else:
            clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        kfold_test_acc.append(log_loss(y_test, y_pred))
        kfold_subm_y_preds.append(clf.predict_proba(X_submission))

    # calc avg log_loss from each k-fold round
    acc = np.mean(kfold_test_acc)
    # calc avg submission prediction
    preds = np.array(kfold_subm_y_preds).mean(axis=0)  # 3D-array -> axis 0 ishighest level
    stats_per_clf.append((name, preds, acc))
    to_kaggle('{0}/{0}_{1}.csv'.format(foldername, name), preds, test_df, train_df, le)
    print('-------------')

# reuse old script instead of rewriting without filesystem access
dfs = [pd.read_csv('{0}/{0}_{1}.csv'.format(foldername, fn)) for fn, _, _ in stats_per_clf]
grouped_df = pd.concat(dfs).groupby('ID')
grouped_df.mean().to_csv('{0}/{0}_mean.csv'.format(foldername))
grouped_df.median().to_csv('{0}/{0}_median.csv'.format(foldername))
# save params and features
with open('{}/features.json'.format(foldername), 'w') as f:
    json.dump(features, f, indent=4)
with open('{}/params.json'.format(foldername), 'w') as f:
    json.dump(params, f, indent=4)
