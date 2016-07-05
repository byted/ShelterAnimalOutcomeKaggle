from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA
from sklearn.metrics import log_loss


def score(train_X, train_y):
    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.01, random_state=10)

    clf = LDA()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_valid)
    return log_loss(y_valid, y_pred)


def predict(train_X, train_y, test_X):
    clf = LDA()
    clf.fit(train_X, train_y)
    return clf.predict_proba(test_X)
