from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
