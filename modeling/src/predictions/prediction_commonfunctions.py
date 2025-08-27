from sklearn.pipeline import Pipeline
import sklearn
import numpy as np
import pandas as pd

dataset = None
cs = None
logger = None
bq = None
task_name = None
replace = None


def ensure_columns(df, training_columns):
    '''Ensure that the DataFrame has all columns required by the model.
    Needs to be a sklearn pipeline with a SimpleImputer as the first step.
    '''
    for col in training_columns:
        if col not in df.columns:
            logger.warning(f'Column {col} not in dataframe. Adding it.')
            df[col] = False
    return df[training_columns]
def predict_data(dataframe ,pipeline: sklearn.pipeline ,target ,training_columns ,log_target = None):
    if not log_target:
        log_target = log_target
    X = dataframe.drop(columns=[target], axis=1)
    # y = dataframe[target]
    X = ensure_columns(df=X, training_columns=training_columns)
    y_pred = np.log1p(pipeline.predict(X)) if log_target else pipeline.predict(X)
    return y_pred

def rm_other_outliers(df):
    if 'bedrooms' in df.columns and 'monthly_rent' in df.columns:
        drop_item_ids = df[(df['bedrooms'] == 0) & (df['monthly_rent'] > 20000)]
        df = df[~df['item_id'].isin(drop_item_ids['item_id'])]
    return df

def read_oslo(query):
    df = bq.read_bq(query)
    df['district_name'] = df['district_name'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['district_name'], drop_first=True)
    df = rm_other_outliers(df)
    df.drop(columns=['pre_processed_date'], errors='ignore', inplace=True)
    df.set_index('item_id', inplace=True)
    return df

def rm_coor_outliers(df):
    df = df[(df['lat'] > 57.5) & (df['lat'] < 71.5)
            & (df['lng'] > 4) & (df['lng'] < 31)]
    return df

def save_data(df,table_name):
    if task_name not in ['admin','clean','pre_processed','raw','predictions']:
        raise ValueError(f'Task name "{task_name}" is not allowed for saving data. Must be one of: "admin", "clean", "pre_processed", "raw", "predictions".')
    if replace:
        bq.to_bq(df,
                 table_name=table_name,
                 dataset_name=task_name,
                 if_exists='replace')
    else:
        bq.to_bq(df,
                 table_name=table_name,
                 dataset_name=task_name,
                 if_exists='merge',
                 merge_on=['item_id'])