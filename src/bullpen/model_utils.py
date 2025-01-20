from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bullpen.data_utils import PlayerLookup

HERE = Path(__file__)
MODEL_DIR = HERE.parents[2].joinpath('models')
LOOKUP = PlayerLookup()


def sort_features_by_coefs(feature_names, coefs, print_top_n=0):
    zipped = list(zip(feature_names, coefs))
    out = sorted(zipped, key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in out[:print_top_n]:
        print(feature, round(coef, 5))
    return out


def make_processing_pipeline(categorical_features=None, numeric_features=None):
    """
    Create an sklearn ColumnTransformer object to handle transformations.
    Makes a strong assumption that *only* OHE used for categorical
    and StandardScalar for numeric.

    Parameters
    ----------
    categorical_features : Optional list of str, default=None
        The name of the categorical columns in the training dataframe.
    numeric_features : Optional list of str, default=None
        The name of the numeric columns in the training dataframe.

    Returns
    -------
    sklearn ColumnTransformer with categorical and numeric Pipelines.
    """
    transformers = []

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                # When an unknown category is encountered during transform,
                # the resulting one-hot encoded columns for this feature will be all zeros.
                # TL;DR Creates a new category for missing values.
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        transformers.append(('categorical', categorical_transformer, categorical_features))

    if numeric_features:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('numeric', numeric_transformer, numeric_features))

    processor = ColumnTransformer(transformers=transformers)
    return processor


class Baseline(BaseEstimator, RegressorMixin):
    def __init__(self, method, grouper=None, target='K%'):
        self.method = method
        self.grouper = 'PlayerId' if grouper is None else grouper
        self.target = target

    def __repr__(self):
        return f'{__class__.__name__}(method={self.method!r})'

    def fit(self, X, y):
        # Merge features and target for grouping
        data = pd.concat([X, y.rename(self.target)], axis=1)

        # Compute group-level predictions
        if self.method == 'last':
            self.best_params_ = 'return last seen K%'
            self.group_aggregates_ = data.groupby(self.grouper)[self.target].last().rename('preds')
        elif self.method == 'mean':
            self.best_params_ = 'return player avg K%'
            self.group_aggregates_ = data.groupby(self.grouper)[self.target].mean().rename('preds')
        else:
            raise ValueError(
                f"Invalid method {self.method!r}. Supported methods are 'last' and 'mean'."
            )
        self.fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError(
                f"This {self} instance is not fitted yet. Call 'fit' before using this method."
            )

        preds = X.merge(
            self.group_aggregates_,
            left_on=self.grouper,
            right_index=True,
            how='left',
        )

        if preds['preds'].isnull().any():
            raise ValueError('Some groups in X were not seen during fitting.')

        return preds['preds'].to_numpy()


class ArticleModel(BaseEstimator, RegressorMixin):
    """
    See https://fantasy.fangraphs.com/the-definitive-pitcher-expected-k-formula/.
    xK% = -0.61 + (L/Str * 1.1538) + (S/Str * 1.4696) + (F/Str * 0.9417)
    """

    def __repr__(self):
        return f'{__class__.__name__}()'

    def fit(self, X, y):
        self.best_params_ = 'return xK% from article'
        self.preds_ = -0.61 + (X['L/Str'] * 1.1538) + (X['S/Str'] * 1.4696) + (X['F/Str'] * 0.9417)
        self.fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError(
                f"This {self} instance is not fitted yet. Call 'fit' before using this method."
            )

        if self.preds_.isnull().any():
            raise ValueError('Some groups in X were not seen during fitting.')

        return self.preds_.to_numpy()


def train_baseline(model, X, y, results):
    model.fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    score = model.score(X, y)
    params = model.best_params_
    print(f'{model} {params=} {score=:.3f} {mse=:.5f}')
    results[repr(model)] = (score, mse)

    return preds, results


def train_model(processor, model, X, y, results, name):
    reg = Pipeline(steps=[('processor', processor), ('regressor', model)])

    reg.fit(X, y)
    preds = reg.predict(X)
    mse = mean_squared_error(y, preds)
    score = reg.score(X, y)
    obj = reg.named_steps['regressor']
    params = obj.best_params_ if hasattr(obj, 'best_params_') else None
    # name = reg.named_steps["regressor"].best_estimator_.__class__.__name__
    print(f'{name} {params=} {score=:.3f} {mse=:.5f}')
    results[name] = (score, mse)

    return preds, results


def find_delta_extrema(X_df, y_df, preds, extrema='max'):
    diffs = np.abs(y_df - preds)
    f = getattr(np, f'arg{extrema}')
    idx = f(diffs)
    mlb_id = X_df.iloc[idx].MLBAMID
    fangraphs_id = X_df.iloc[idx].PlayerId
    name = LOOKUP.get_name_from_id(mlb_id)
    return name, mlb_id, fangraphs_id
