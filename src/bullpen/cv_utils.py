from itertools import product

import numpy as np

from bullpen.model_utils import train_model


def make_timeseries_splits(year_list, train_df):
    splits = {'train': [], 'val': []}
    for idx, _ in enumerate(year_list[:-1]):
        train_years = year_list[: idx + 1]
        val_year = year_list[idx + 1]

        print(f'TRAIN: {train_years} VAL: {[val_year]}')

        splits['train'].append(train_df[train_df['Season'].isin(train_years)])
        splits['val'].append(train_df[train_df['Season'] == val_year])
    return splits


def pred_X_y(split, target='K%', drop_cols=None):
    drop_cols = ['Name', 'Rk', 'PAu', 'Pitu', 'Stru', target] if drop_cols is None else drop_cols

    X_df = split[[c for c in split.columns if c not in drop_cols]]
    y_df = split[target]
    return X_df, y_df


def cross_validate_model(model, param_grid, splits, processor, metric_key='mean_mse', K=2):
    """
    Manual cross-validation based on custom timeseries data
    """
    results = []
    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        print(f'Testing parameters: {param_dict}')

        split_scores = []
        for split_idx in range(K):
            # Get training and validation data
            X_df, y_df = pred_X_y(splits['train'][split_idx])
            X_val_df, y_val_df = pred_X_y(splits['val'][split_idx])
            print(f'TRAIN: {X_df.Season.unique()} VAL: {X_val_df.Season.unique()}')

            # Initialize and train the model
            preds, metrics = train_model(
                processor, model(**param_dict), X_df, y_df, results={}, name='model'
            )

            # Collect the desired metric (e.g., MSE) which is the second, or last appended
            split_scores.append(metrics['model'][-1])

        # Compute mean metric across splits
        mean_metric = np.mean(split_scores)
        results.append({**param_dict, metric_key: mean_metric})

        print(f'Mean {metric_key}: {mean_metric:.4f}')
        print()

    # Find the best hyperparameters based on the lowest metric
    best_result = min(results, key=lambda x: x[metric_key])
    return results, best_result
