import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AirbnbFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
        df["number_of_reviews"] = np.log1p(df["number_of_reviews"])
        df["availability_365"] = df["availability_365"].clip(0, 365)
        df["minimum_nights"] = df["minimum_nights"].clip(1, 30)
        df["calculated_host_listings_count"] = np.log1p(df["calculated_host_listings_count"])
        return pd.get_dummies(df, columns=["city", "neighbourhood_group_cleansed", "room_type"], drop_first=True)
