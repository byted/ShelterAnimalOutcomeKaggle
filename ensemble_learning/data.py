# adapted version of https://www.kaggle.com/cbrogan/titanic/xgboost-example-python/code
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def get_data():
    # Load the data
    train_df = pd.read_csv('../data/train.csv', header=0)
    test_df = pd.read_csv('../data/test.csv', header=0)

    train_df['Intact'], train_df['Sex'] = train_df.SexuponOutcome.str.split(' ').str
    train_df['Intact'] = train_df['Intact'].replace('Unknown', np.nan)
    train_df['Intact'] = train_df['Intact'].replace('Spayed', 'Neutered')

    test_df['Intact'], test_df['Sex'] = test_df.SexuponOutcome.str.split(' ').str
    test_df['Intact'] = test_df['Intact'].replace('Unknown', np.nan)
    test_df['Intact'] = test_df['Intact'].replace('Spayed', 'Neutered')

    train_df['NameShort'] = train_df['Name']
    test_df['NameShort'] = test_df['Name']

    # train_df['ID'] = train_df['AnimalID']
    # train_df.drop('AnimalID', axis=1, inplace=True)

    def calc_age_in_days(df):
        factor = {'year': 365, 'month': 31, 'week': 7, 'day': 1}
        result = []
        for age in df.AgeuponOutcome:
            if str(age) != 'nan':
                value, unit = age.split(' ')
                days = int(value) * factor[unit.replace('s', '')]  # ignore year[s], month[s], ...
                result.append(days)
            else:
                result.append(np.nan)
        df['AgeuponOutcomeInDays'] = result
        return df

    def calc_age_in_years(df):
        result = []
        for age in df.AgeuponOutcomeInDays:
            if str(age) != 'nan':
                years = int(age / 365)
                result.append(years)
            else:
                result.append(np.nan)
        df['AgeuponOutcomeInYears'] = result
        return df

    train_df = calc_age_in_days(train_df)
    train_df = calc_age_in_years(train_df)
    test_df = calc_age_in_days(test_df)
    test_df = calc_age_in_years(test_df)


    # We'll impute missing values using the median for numeric columns and the most
    # common value for string columns.
    # This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
    from sklearn.base import TransformerMixin
    class DataFrameImputer(TransformerMixin):
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)

    feature_columns_to_use = list(set(train_df) - set(['OutcomeType', 'OutcomeSubtype', 'ID', 'AnimalID']))

    nonnumeric_columns = feature_columns_to_use

    # Join the features from train and test together before imputing missing values,
    # in case their distribution is slightly different
    # Do we really wanna do this? S/b said this is bad practice for some reason
    big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
    big_X_imputed = DataFrameImputer().fit_transform(big_X)

    # See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
    # details and options
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

    # Prepare the inputs for the model
    train_X = big_X_imputed[0:train_df.shape[0]]
    test_X = big_X_imputed[train_df.shape[0]::]
    train_df.OutcomeType = le.fit_transform(train_df.OutcomeType)
    train_y = train_df['OutcomeType']

    return (train_X, train_y, test_X, test_df, train_df, le)
