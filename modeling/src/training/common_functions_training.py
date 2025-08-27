from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import random

dataset = None
cs = None
logger = None


def load_data(df, target, log_target):
    """
    Splits the dataframe into features and target variable.
    """
    X = df.drop(columns=[target], axis=1)
    y = np.log1p(df[target]) if log_target else df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    logger.info(f"Target: {target} and log_target: {log_target}")
    # logger.info(f'Columns in train set: {X_train.columns.tolist()}')
    return X_train, X_test, y_train, y_test


def save_model(pipeline, results: dict, data_name: str):
    model_name = pipeline.steps[-1][1].__class__.__name__
    filename_bucket = f'models/{model_name}_{data_name}.pkl'
    manifest_filename = 'models.json'
    local_manifest_path = f'/tmp/{manifest_filename}'

    # Lag en DataFrame med info om den nye modellen
    results['filename'] = filename_bucket
    results['model_type'] = model_name
    new_model_info = pd.DataFrame([results])

    try:
        cs.download(manifest_filename, local_manifest_path)
        all_models_df = pd.read_json(local_manifest_path, orient='records', lines=True)

        if data_name in all_models_df['dataset'].values:
            logger.info(f"Oppdaterer eksisterende modell '{data_name}' i manifestet.")
            all_models_df = all_models_df[all_models_df['dataset'] != data_name]
        all_models_df = pd.concat([all_models_df, new_model_info], ignore_index=True)
        all_models_df.drop_duplicates(subset=['dataset'], keep='last', inplace=True)
    except Exception as e:
        logger.info(f"Kunne ikke finne '{manifest_filename}'. Oppretter en ny. Feil: {e}")
        all_models_df = new_model_info

    all_models_df.to_json(local_manifest_path, orient='records', lines=True)
    cs.upload(local_manifest_path, manifest_filename)

    local_filepath = f'/tmp/tmp_file.pkl'
    joblib.dump(pipeline, local_filepath)
    try:
        cs.upload(local_filepath, filename_bucket)
    finally:
        os.remove(local_filepath)


def calc_scores(true, pred, log_target: bool) -> tuple:
    '''

    :param true:
    :param pred:
    :param log_target:
    :return: tuple (mse,r2)
    '''
    r2 = r2_score(np.expm1(true), np.expm1(pred)) if log_target else r2_score(true, pred)
    mse = mean_squared_error(np.expm1(true), np.expm1(pred)) if log_target else mean_squared_error(true, pred)
    return mse, r2


def get_cat(df):
    cat_cols = []
    cat_idx = []
    for idx, (col, dtype) in enumerate(df.dtypes.items()):
        if dtype == 'object' or dtype == 'category' or dtype == 'string':
            cat_cols.append(col), cat_idx.append(idx)
    return cat_cols, cat_idx


def train(df, params, data_name, model, target, save_to_gc=True, categorical=False, log_target=None,logger = None):
    logger.info(f'\n \nTRAINING {model().__class__.__name__} model for {data_name.upper()}')
    if not log_target:
        log_target = log_target
    X_train, X_test, y_train, y_test = load_data(df=df, target=target, log_target=log_target)

    cat_cols, cat_idx = get_cat(X_train)
    num_cols = X_train.select_dtypes(include=[np.number, 'bool', 'boolean']).columns.tolist()

    num_trans = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        # ('scaler', StandardScaler())
    ])
    cat_trans = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        # ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    if categorical:
        pre_processer = ColumnTransformer(
            transformers=[
                ('num', num_trans, num_cols),
                ('cat', cat_trans, cat_cols)
            ], remainder='passthrough')

        cat_features_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        pipeline = Pipeline([
            ('pre_processor', pre_processer),
            ('model', CatBoostRegressor(cat_features=cat_features_indices, verbose=0)),  # Changed to cat_features
        ])

        pipeline_params = {'model__' + key: value for key, value in params.items()}
        pipeline.set_params(**pipeline_params)
    else:
        pre_processer = ColumnTransformer(
            transformers=[
                ('num', num_trans, num_cols),
                # ('cat', cat_trans, cat_cols)
            ], remainder='passthrough')
        pipeline = Pipeline([
            ('pre_processor', pre_processer),
            ('model', model()),
        ])
        pipeline.set_params(**params)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    mse, r2 = calc_scores(y_test, y_pred, log_target=log_target)
    mse_train, r2_train = calc_scores(y_train, y_pred_train, log_target=log_target)

    training_columns = {}
    for col, dtype in X_train.dtypes.items():
        tries = 0
        sample = None
        while sample is None or sample is np.nan:
            sample_idx = random.randint(0, len(X_train[col]) - 1)
            sample = X_train[col].iloc[sample_idx]
            tries += 1
            if tries > 1000:
                break
        training_columns[col] = (dtype, sample)

    logger.info(
        f'MSE test: {mse},r2 test: {r2}, mse train: {mse_train}, r2 train {r2_train} '
        f'for {data_name} with target {target} and log_target {log_target}')
    results = {
        'dataset': data_name,
        'target': target,
        'log_target': log_target,
        'created_at': pd.Timestamp.now(),
        'training_columns': training_columns,
        'params': str(params),
        'r2_score': r2,
        'mse': mse,
        'mse_train': mse_train,
        'r2_train': r2_train
    }
    if save_to_gc:
        save_model(pipeline, results=results, data_name=data_name)
    return pipeline


def rm_coor_outliers(df):
    df = df[(df['lat'] > 57.5) & (df['lat'] < 71.5)
            & (df['lng'] > 4) & (df['lng'] < 31)]
    return df