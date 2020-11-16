import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def calc_age_category(age):
            if age <= 30:
                age_category = 'age_0-30'
            elif (age > 30) & (age <= 40 ):
                age_category = 'age_30-40'
            elif (age > 40) & (age <= 50 ):
                age_category = 'age_40-50'
            elif (age > 50) & (age <= 60 ):
                age_category = 'age_50-60'
            elif (age > 60):
                age_category = 'age_60'
            return age_category
        return X[self.key].apply(lambda x: calc_age_category(x), 1)

df = pd.DataFrame({'age': [23, 16, 89]})
age_encoder = AgeSelector('age')
age_encoder.fit(df)
age_encoder.transform(df)
