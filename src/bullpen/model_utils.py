from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def sort_features_by_coefs(features, coefs):
    zipped = list(zip(features, coefs))
    return sorted(zipped, key=lambda x: abs(x[1]), reverse=True)


def make_processing_pipeline(categorical_features, numeric_features):
    """
    Create an sklearn ColumnTransformer object to handle transformations.
    Makes a strong assumption that *only* OHE used for categorical
    and StandardScalar for numeric.

    Parameters
    ----------
    categorical_features : list of str
        The name of the categorical columns in the training dataframe.
    numeric_features : list of str
        The name of the numeric columns in the training dataframe.
    """
    categorical_transformer = Pipeline(
        steps=[
            # When an unknown category is encountered during transform,
            # the resulting one-hot encoded columns for this feature will be all zeros.
            # TL;DR Creates a new category for missing values.
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    processor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numeric', numeric_transformer, numeric_features),
        ]
    )
    return processor
