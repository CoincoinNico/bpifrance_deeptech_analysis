from sklearn.base import BaseEstimator, TransformerMixin

class NoTransformer(BaseEstimator, TransformerMixin):
    """
    Customized identity transformer
    """


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
