import re
import pandas as pd
import numpy as np
## TRANSFORM METHODS
def extract_int(x):
    if not isinstance(x, str):
        return None
    uten_mellomrom = re.sub(r'\s', '', x)
    treff = re.search(r'[\d.]+', uten_mellomrom)
    if not treff:
        return None
    try:
        nummer_str = treff.group(0)
        nummer_float = float(nummer_str)
        if nummer_float.is_integer():
            return int(nummer_float)
        else:
            return nummer_float
    except (ValueError, TypeError):
        return None
def extract_float(x):
    if not isinstance(x, str):
        return None
    uten_mellomrom = re.sub(r'\s', '', x)
    match = re.search(r'[\d.,]+', uten_mellomrom)
    if not match:
        return None
    nummer_str = match.group(0)
    if ',' in nummer_str and '.' in nummer_str:
        if nummer_str.rfind(',') > nummer_str.rfind('.'):
            nummer_str = nummer_str.replace('.', '').replace(',', '.')
        else:
            nummer_str = nummer_str.replace(',', '')
    elif ',' in nummer_str:
        nummer_str = nummer_str.replace(',', '.')
    try:
        return float(nummer_str)
    except (ValueError, TypeError):
        return None
def extract_postnummer( x):
    if not isinstance(x, str):
        return x

    else:
        match_ = re.search(r'\d{4}', x)
        if match_:
            return match_.group()
def mk_num(df,int_cols,type = 'int'):
    if type not in ['int','float']:
        raise ValueError(f'Type "{type}" is not allowed. Must be "int" or "float".')
    for col in int_cols:
        if col in df.columns:
            if type == 'int':
                new = df[col].apply(lambda x: extract_int(x) if isinstance(x, str) else x)
                df[col] = pd.to_numeric(new, errors='coerce').astype('Int64', errors='ignore')
            elif type == 'float':
                new = df[col].apply(lambda x: extract_float(x) if isinstance(x, str) else x)
                df[col] = pd.to_numeric(new, errors='coerce').astype('Float64', errors='ignore')
            else:
                raise ValueError(f'Type "{type}" is not allowed. Must be "int" or "float".')
    return df
def mk_cat(df, col, valid_values):
    """
    Convert a column to a categorical type with specified valid values.
    """
    df[col] = df[col].apply(lambda x: x.lower() if  isinstance(x,str) else x)
    valid_values = [x.lower() for x in valid_values if isinstance(x, str)]
    isin = df[col].isin(valid_values)
    df = df[isin].copy()
    df.loc[:, col] = df[col].astype(str)
    df.loc[:, col] = pd.Categorical(df[col], categories=valid_values, ordered=False)
    return df
def ensure_num_types(df,num_types = None) -> pd.DataFrame:
    if num_types is None:
        num_types = ['int', 'float']
    if not isinstance(num_types, list):
        raise ValueError(f'num_types must be a list, got {type(num_types)} instead.')
    if not all(dtype in ['int', 'float'] for dtype in num_types):
        raise ValueError(f'num_types must contain only "int" and "float", got {num_types} instead.')
    if num_types  == ['int']:
        for col, dtype in df.dtypes.items():
            if dtype == 'Float64' or dtype == 'float64' or dtype == 'float32' or dtype == 'int32':
                new = df[col]
                df[col] = new.astype('Int64', errors='ignore')
    elif num_types == ['int','float']:
        for col, dtype in df.dtypes.items():
            if dtype == 'Float64' or dtype == 'float64' or dtype == 'float32':
                new = df[col]
                df[col] = new.astype('Float64', errors='ignore')
                # logger.debug(f'Column {col} changed from {dtype} to {df[col].dtypes}')
            elif dtype == 'int32' or dtype == 'int64':
                new = df[col]
                df[col] = new.astype('Int64', errors='ignore')
                # logger.debug(f'Column {col} changed from {dtype} to {df[col].dtypes}')
    elif num_types == ['float']:
        for col, dtype in df.dtypes.items():
            if dtype == 'Float64' or dtype == 'float64' or dtype == 'float32' or dtype == 'int32':
                new = df[col]
                df[col] = new.astype('Float64', errors='ignore')
    return df
def transform_nan(df):
    #logger.debug(f'Length of df before cleaning: {len(df)}')
    df = df.drop_duplicates(subset='item_id')
    values_to_replace_map = {
        'nan': np.nan, 'None': np.nan, '': np.nan, 'null': np.nan,
        'NA': np.nan, 'np.nan': np.nan, '<NA>': np.nan, 'NaN': np.nan,
        'NAType': np.nan
    }
    #null_val = ['nan', 'None', '', 'null', 'NULL', 'NA', 'np.nan', '<NA>', 'NaN', 'NAType', np.nan]
    df.replace(values_to_replace_map, inplace=True)
    return df
def fill_na(df,feature,fill_value):
    if feature in df.columns:
        df.loc[:,feature] = df[feature].fillna(fill_value)
    else:
        df[feature] = fill_value
    return df

def rm_empty_features(df,threshold = 0.9):
    for col in df.columns:
        if df[col].isna().sum() / len(df) > threshold:
            df.drop(col, axis=1, inplace=True)
    #logger.debug(f'Length: {len(df)} | after removing columns with >90% missing values')
    return df
def add_missing_features(df: pd.DataFrame,missing_features : list) -> pd.DataFrame:
    for col in missing_features:
        if col not in df.columns:
            df[col] = None
    return df
def mk_fractions(df,new_feat_name,numerator,denominator):

    df[new_feat_name] = df.apply(
        lambda x: x[numerator] / x[denominator]
        if pd.notna(x[denominator]) and pd.notna(x[numerator]) and x[denominator] > 0
        else np.nan,
        axis=1
    )
    return df
def split_date(df,date_col:str):
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df.drop(date_col, axis=1, inplace=True)
    return df


def mk_bool_description(df, col_name, keys, source_cols=['description']):
    """
    Lager en boolsk kolonne som er True hvis noen av nøkkelordene finnes
    i en eller flere av kildekolonnene.
    """
    if isinstance(source_cols, str):
        source_cols = [source_cols]

    # Bygg regex-mønsteret
    regex = '|'.join(re.escape(key) for key in keys)
    pattern = re.compile(regex, re.IGNORECASE)

    # Start med at alt er False
    df[col_name] = False

    # Gå gjennom hver kildekolonne og oppdater resultatet
    for source_col in source_cols:
        if source_col in df.columns:
            # Finn treff i gjeldende kolonne
            matches = df[source_col].str.contains(pattern, regex=True, na=False)
            # Bruk logisk ELLER for å kombinere med tidligere resultater
            df[col_name] = df[col_name] | matches

    df[col_name] = df[col_name].astype('boolean')
    return df


# def mk_bool_features(df, col_name, key: str, source_col='features'):
#     df[col_name] = df[source_col].apply(lambda x:
#                                         True
#                                         if isinstance(x, str) and
#                                            any(key in item.strip().lower() for item in
#                                                x.replace('[', "").replace("]", "").split(','))
#                                         else False
#                                         )
#     df[col_name] = df[col_name].astype('boolean')
#     return df

def mk_bool_features(df, equipment_features, source_col = 'features'):
    # Process features once
    df.loc[:,source_col] = df[source_col].apply(
        lambda x: [item.strip().strip("'\"").lower()
                   for item in (x.replace("]","").replace("[","").split(',')
                                if isinstance(x, str) else [])
                   if isinstance(item, str) and item.strip() != '']
    )

    # Create a feature lookup dictionary for faster matching
    feature_lookup = {}
    for feature_name, keywords in equipment_features.items():
        for keyword in keywords:
            if keyword not in feature_lookup:
                feature_lookup[keyword] = []
            feature_lookup[keyword].append(feature_name)

    # Process all features in one pass
    feature_columns = {feature: np.zeros(len(df), dtype=bool) for feature in equipment_features}

    def process_row(idx, features_list):
        if not isinstance(features_list, list):
            return

        for feature in features_list:
            for keyword in feature_lookup:
                if keyword in feature:
                    for feature_name in feature_lookup[keyword]:
                        feature_columns[feature_name][idx] = True

    # Apply the function to each row
    for idx, features_list in enumerate(df[source_col]):
        process_row(idx, features_list)

    # Add the columns to the DataFrame
    for feature_name, values in feature_columns.items():
        df[feature_name] = values
    df[source_col] = df[source_col].astype(str)

    return df

def process_bool(df):
    bool_cols = []
    for col,dtype in df.dtypes.items():
        if dtype == 'bool' or dtype == 'boolean':
            bool_cols.append(col)
    for col in bool_cols:
        df[col] = df[col].astype('object').astype(str)
        df[col] = pd.Categorical(df[col], categories=['False', 'True'], ordered=False)
    df = pd.get_dummies(df, columns=bool_cols, drop_first=True)
    return df

def get_top_features(df,source_col= 'feautures'):
    '''
    Extracts and processes features from a DataFrame column containing lists of features.
    :param df:
    :param source_col:
    :return:
    '''
    feat = df[source_col].apply(lambda x: x.lower().replace("]","").replace("[","").split(',') if isinstance(x, str) else [])
    feat = feat.apply(lambda x: [i.strip().strip("'\"") for i in x if isinstance(i, str) and i.strip() != ''])
    f = feat.explode()
    return f