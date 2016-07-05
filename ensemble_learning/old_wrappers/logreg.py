from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def score(train_X, train_y):
    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.01, random_state=10)
    clf = OneVsRestClassifier(LogisticRegression(penalty='l1', fit_intercept=True, C=0.9, tol=.000001))
    y_pred = clf.fit(X_train, y_train).predict_proba(X_valid)
    return log_loss(y_valid, y_pred)


def predict(train_X, train_y, test_X):
    clf = OneVsRestClassifier(LogisticRegression(penalty='l1', fit_intercept=True, C=0.9, tol=.000001))
    clf.fit(train_X, train_y)
    return clf.predict_proba(test_X)
