import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


df = pd.DataFrame({'lesson_price': [4, 2, 18, 48, 5, 9, 3, 2, 18, 48],
                   'qualification': [1, 1, 2, 3, 2, 1, 3, 4, 4, 2]})
                   
class IQRFilter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.iqr_dict = dict()
        self.q3_dict = dict()
        
    def fit(self, X):
        self.q3_dict = df.groupby('qualification')['lesson_price'].quantile(0.75).to_dict()
        self.iqr_dict = df.groupby('qualification')['lesson_price'].apply(lambda x: np.quantile(x, 0.75)-\
                                                                              np.quantile(x, 0.25)).to_dict()
        return self
    
    def transform(self, X):
        X['lesson_price_filtered'] = X[['lesson_price', 
                                          'qualification']].apply(lambda x: x[0] if x[0]>=self.q3_dict[x[1]] \
                                                                     and x[0]<=(self.q3_dict[x[1]]\
                                                                          +self.iqr_dict[x[1]]) else None, 1)
        return X[~X['lesson_price_filtered'].isnull()]

iqr_filter = IQRFilter()
iqr_filter.fit(df)
iqr_filter.transform(df)

print(iqr_filter.q3_dict)
print(iqr_filter.iqr_dict)
