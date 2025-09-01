import ast
import os
from sklearn.compose import ColumnTransformer
import random
from sibr_module import Logger,CStorage
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error,r2_score
import joblib
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from src.helper_modules import CustomBigQuery
from dateutil import parser

class SibrBase:
    def __init__(self,dataset,logger = None):
        self._dataset = dataset
        self._task_name = None
        self._replace = False
        self._bucket_name = 'sibr-market'
        self._project_id = 'sibr-market'
        self.logger = logger
        self.bq = None
        self.cs = None
        self.setup()

    @property
    def dataset(self):
        return self._dataset

    @property
    def replace(self):
        return self._replace
    @replace.setter
    def replace(self, value):
        if isinstance(value, bool):
            self._replace = value
        else:
            raise ValueError("Replace must be a boolean value.")

    @property
    def task_name(self):
        return self._task_name
    @task_name.setter
    def task_name(self, value):
        if value in ['admin', 'clean', 'pre_processed', 'raw', 'predictions','train']:
            self._task_name = value
        else:
            raise ValueError("Task name must be one of: 'admin', 'clean', 'pre_processed', 'raw', 'predictions'.")

    def setup(self):
        if not self.logger:
            self.logger = Logger(f'{self.dataset.capitalize()}')
        self.bq = CustomBigQuery(project_id = self._project_id ,logger=self.logger,dataset = self.dataset)
        self.cs = CStorage(project_id = self._project_id , logger=self.logger, bucket_name=self._bucket_name)
        self.logger.debug(f'Dataset: {self.dataset} | | Replace: {self.replace}')

    def save_data(self,df,table_name):
        if self.task_name not in ['admin','clean','pre_processed','raw','predictions']:
            raise ValueError(f'Task name "{self.task_name}" is not allowed for saving data. Must be one of: "admin", "clean", "pre_processed", "raw", "predictions".')
        if self.replace:
            self.bq.to_bq(df,
                     table_name=table_name,
                     dataset_name=self.task_name,
                     if_exists='replace')
        else:
            self.bq.to_bq(df,
                     table_name=table_name,
                     dataset_name=self.task_name,
                     if_exists='merge',
                     merge_on=['item_id'])

class Clean(SibrBase):
    def __init__(self,dataset,logger = None,df = None):
        super().__init__(dataset=dataset,logger=logger)
        self.logger.info('Clean class initialized.')
        self.df = df
        self.geo = None
        #self.salgs = None

    ## TRANSFORM METHODS
    import re


    def extract_int(self, x):
        """
        Extracts an integer from a string.

        Args:
            x (str): The input value from which to extract the number.
        Returns:
            int or None: The extracted integer, or None if no valid number can be extracted.

        Examples
            "200.000" : str -> 200000 : int
            "200,000" : str -> 200000 : int
            "200m2" : str -> 200 : int
            "2000,0 : str -> 2000000 : int
            "435,5 kr : srt -> 435500 : int
        """
        if not isinstance(x, str):
            return None
        uten_mellomrom = re.sub(r'\s', '', x)
        match = re.search(r'[\d.,]+', uten_mellomrom)
        if not match:
            return None
        nummer_str = match.group(0)
        has_separator = '.' in nummer_str or ',' in nummer_str
        # Parse to float using similar logic as extract_float
        if ',' in nummer_str and '.' in nummer_str:
            if nummer_str.rfind(',') > nummer_str.rfind('.'):
                nummer_str_parsed = nummer_str.replace('.', '').replace(',', '.')
            else:
                nummer_str_parsed = nummer_str.replace(',', '')
        elif ',' in nummer_str:
            nummer_str_parsed = nummer_str.replace(',', '.')
        else:
            nummer_str_parsed = nummer_str
        try:
            nummer_float = float(nummer_str_parsed)
        except (ValueError, TypeError):
            return None
        if not has_separator:
            if nummer_float.is_integer():
                return int(nummer_float)
            else:
                return None
        else:
            # Multiply by 1000 and round to int
            multiplied = nummer_float * 1000
            return int(round(multiplied))

    def extract_float(self, x):
        """
        Extracts  a float number from a string.

        Args:
            x (str): The input value from which to extract the number.
        Returns:
            float or None: The extracted floating number, or None if no valid number can be extracted.

        Examples
            "200.000" : str -> 200.0 : float
            "200,000" : str -> 200.0 : float
            "200m2" : str -> 200.0 : float
            "2000,0 : str -> 2000.0 : float
            "435,5 kr : srt -> 435.5 : float
                """
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
    def extract_postnummer(self, x):
        if not isinstance(x, str):
            return x

        else:
            match_ = re.search(r'\d{4}', x)
            if match_:
                return match_.group()

    def extract_datetime(self,x):
        """
        Extracts a datetime object from a string, handling both English and Norwegian month names.

        Args:
            x (str): The input string containing the date and time.
        Returns:
            datetime or None: The extracted datetime, or None if parsing fails.

        Examples:
            'July 12, 2025, at 12:23\u202fPM' -> datetime(2025, 7, 12, 12, 23)
            '2024-05-28 13:35:00' -> datetime(2024, 5, 28, 13, 35)
            '4. februar 2025, 17:12' -> datetime(2025, 2, 4, 17, 12)
            'August 23, 2025, at 06:08\u202fAM' -> datetime(2025, 8, 23, 6, 8)
        """
        if not isinstance(x, str):
            return None
        x = x.replace('\u202f', ' ')
        nor_to_eng = {
            'januar': 'January',
            'februar': 'February',
            'mars': 'March',
            'april': 'April',
            'mai': 'May',
            'juni': 'June',
            'juli': 'July',
            'august': 'August',
            'september': 'September',
            'oktober': 'October',
            'november': 'November',
            'desember': 'December',
        }
        x_trans = x
        for nor, eng in nor_to_eng.items():
            x_trans = re.sub(r'\b' + re.escape(nor) + r'\b', eng, x_trans, flags=re.IGNORECASE)
        try:
            dt = parser.parse(x_trans, fuzzy=True)
            return dt
        except (ValueError, TypeError):
            return None

    def mk_num(self,df,int_cols,type = 'int'):
        if type not in ['int','float']:
            raise ValueError(f'Type "{type}" is not allowed. Must be "int" or "float".')
        for col in int_cols:
            if col in df.columns:
                if type == 'int':
                    new = df[col].apply(lambda x: self.extract_int(x) if isinstance(x, str) else x)
                    df[col] = pd.to_numeric(new, errors='coerce').astype('Int64', errors='ignore')
                elif type == 'float':
                    new = df[col].apply(lambda x: self.extract_float(x) if isinstance(x, str) else x)
                    df[col] = pd.to_numeric(new, errors='coerce').astype('Float64', errors='ignore')
                else:
                    raise ValueError(f'Type "{type}" is not allowed. Must be "int" or "float".')
            else:
                self.logger.warning(f'Column "{col}" not found in dataframe.')
        return df
    def mk_cat(self,df, col, valid_values):
        """
        Convert a column to a categorical type with specified valid values.
        """
        df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        valid_values = [x.lower() for x in valid_values if isinstance(x, str)]
        isin = df[col].isin(valid_values)
        df = df[isin].copy()
        df.loc[:, col] = df[col].astype(str)
        df.loc[:, col] = pd.Categorical(df[col], categories=valid_values, ordered=False)
        return df
    def ensure_num_types(self,df,num_types = None) -> pd.DataFrame:
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
                    # self.logger.debug(f'Column {col} changed from {dtype} to {df[col].dtypes}')
                elif dtype == 'int32' or dtype == 'int64':
                    new = df[col]
                    df[col] = new.astype('Int64', errors='ignore')
                    # self.logger.debug(f'Column {col} changed from {dtype} to {df[col].dtypes}')
        elif num_types == ['float']:
            for col, dtype in df.dtypes.items():
                if dtype == 'Float64' or dtype == 'float64' or dtype == 'float32' or dtype == 'int32':
                    new = df[col]
                    df[col] = new.astype('Float64', errors='ignore')
        return df
    def transform_nan(self,df):
        self.logger.debug(f'Length of df before cleaning: {len(self.df)}')
        df = df.drop_duplicates(subset='item_id')
        values_to_replace_map = {
            'nan': np.nan, 'None': np.nan, '': np.nan, 'null': np.nan,
            'NA': np.nan, 'np.nan': np.nan, '<NA>': np.nan, 'NaN': np.nan,
            'NAType': np.nan
        }
        #null_val = ['nan', 'None', '', 'null', 'NULL', 'NA', 'np.nan', '<NA>', 'NaN', 'NAType', np.nan]
        df.replace(values_to_replace_map, inplace=True)
        return df
    def fill_na(self,df,feature,fill_value):
        if feature in df.columns:
            try:
                if pd.api.types.is_any_real_numeric_dtype(df[feature]):
                    fill_value = float(fill_value)
                elif pd.api.types.is_bool_dtype(df[feature]):
                    fill_value = bool(fill_value)
                elif pd.api.types.is_string_dtype(df[feature]):
                    fill_value = str(fill_value)
                elif pd.api.types.is_object_dtype(df[feature]):
                    fill_value = str(fill_value)
            except (ValueError, TypeError):
                self.logger.warning(f'Tried converting {fill_value} to the correct dtype {df[feature].dtype}, but failed.')
                pass
            df.loc[:,feature] = df[feature].fillna(fill_value)
        else:
            df[feature] = fill_value
        return df
    def rm_empty_features(self,df,threshold = 0.9):
        for col in df.columns:
            if df[col].isna().sum() / len(df) > threshold:
                df.drop(col, axis=1, inplace=True)
        self.logger.debug(f'Length: {len(df)} | after removing columns with >90% missing values')
        return df
    def add_missing_features(self,df: pd.DataFrame,missing_features : list) -> pd.DataFrame:
        for col in missing_features:
            if col not in df.columns:
                df[col] = None
        return df
    def mk_fractions(self,df,new_feat_name,numerator,denominator):

        df[new_feat_name] = df.apply(
            lambda x: x[numerator] / x[denominator]
            if pd.notna(x[denominator]) and pd.notna(x[numerator]) and x[denominator] > 0
            else np.nan,
            axis=1
        )
        return df
    def split_date(self,df,date_col:str):
        df[date_col] = pd.to_datetime(df[date_col],errors="coerce")
        df['day'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df.drop(date_col, axis=1, inplace=True)
        return df

    def mk_bool_description(self,df : pd.DataFrame, col_name : str, keys : list, source_cols=['description']) -> pd.DataFrame:
        """
        Takes inn the full dataframe, makes a new boolean feature based on the key inputs and outputs a full dataframe again.
        :param df
        :param col_name
        :param keys
        :param source_cols
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

    def mk_bool_features(self, df : pd.DataFrame, equipment_features : dict, source_col='features') -> pd.DataFrame:
        """
        Takes in a full dataframe, makes a boolean feature based on the input dictionary of feature names keys and search values, and outputs a full dataframe again.
        :param df: 
        :param equipment_features: 
        :param source_col: 
        :return: 
        """

        df.loc[:, source_col] = df[source_col].apply(
            lambda x: [item.strip().strip("'\"").lower()
                       for item in (x.replace("]", "").replace("[", "").split(',')
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
    def process_bool(self,df):
        bool_cols = []
        for col, dtype in df.dtypes.items():
            if dtype == 'bool' or dtype == 'boolean':
                bool_cols.append(col)
        for col in bool_cols:
            df[col] = df[col].astype('object').astype(str)
            df[col] = pd.Categorical(df[col], categories=['False', 'True'], ordered=False)
        df = pd.get_dummies(df, columns=bool_cols, drop_first=True)
        return df

    def get_top_features(self,df, source_col='feautures'):
        '''
        Extracts and processes features from a DataFrame column containing lists of features.
        :param df:
        :param source_col:
        :return:
        '''
        feat = df[source_col].apply(
            lambda x: x.lower().replace("]", "").replace("[", "").split(',') if isinstance(x, str) else [])
        feat = feat.apply(lambda x: [i.strip().strip("'\"") for i in x if isinstance(i, str) and i.strip() != ''])
        f = feat.explode()
        return f

    
    def read_in_data(self):
        if self.task_name == 'clean':
            self.df = self.bq.read_raw(replace = self.replace)
            self.geo = self.bq.read_geonorge()
            if self.df is None or self.df.empty:
                self.logger.error(f'No data found, exiting {self.task_name} task.')
                raise ValueError("No data found in BigQuery for the 'clean' task.")
        elif self.task_name == 'pre_processed':
            self.df = self.bq.read_clean(replace = self.replace)
        else:
            raise ValueError(f'Task name "{self.task_name}" is not allowed for reading data. Must be "clean" or "pre_processed".')

    def clean_cars(self) -> pd.DataFrame:

        df = self.transform_nan(self.df)

        df = self.rm_empty_features(df)
        self.logger.debug(f'Length: {len(df)} | after merge with sales time')

        int_cols = [
            'model_year', 'mileage', 'transfer_fee', 'price_excl_transfer', 'power',
            'co2', 'weight', 'seats', 'prev_owners', 'doors', 'trailer_weight',
            'last_eu', 'next_eu', 'range', 'battery', 'liens', 'total_price', 'rn'
             'engine_volume', 'cargo_space'
        ]

        df = self.mk_num(df, int_cols,type='int')
        #df = self.mk_num(df, float_cols,type='float')

        df['transfer_fee'] = df['transfer_fee'].fillna(0)

        df = df[
            (df['total_price'] > 10000) & (df['total_price'] < 3000000)
            & ((df['mileage'] >= 0) & (df['mileage'] < 1000000))
            & ((df['weight'] >= 0) & (df['weight'] < 10000))
            & ((df['seats'] >= 0) & (df['seats'] < 10))
            & ((df['doors'] >= 0) & (df['doors'] < 10))
            & ((df['model_year'] >= 1900) & (df['model_year'] < 2030))
            & ((df['power'] >= 0) & (df['power'] < 5000))
            ]
        self.logger.debug(f'Length: {len(df)} | after filter price and usable_area')

        df = self.mk_fractions(df, "price_pr_km", "total_price", "mileage")

        df['scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce', utc=True)
        df['clean_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length: {len(df)} | after datetime conversion')

        df['postal_code'] = df['address'].apply(self.extract_postnummer)
        df = pd.merge(df, self.geo[['postal_code', 'municipality', 'county', 'region']], how='left', on='postal_code')
        self.logger.debug(f'Length: {len(df)} | after geo')

        fuel_mapping = {
            'diesel': 'diesel',
            'bensin': 'bensin',
            'el': 'el',
            'elektrisitet': 'el',
            'el + bensin': 'el + bensin',
            'hybrid bensin': 'el + bensin',
            'plug-in bensin': 'el + bensin',
            'el + diesel': 'el + diesel',
            'hybrid diesel': 'el + diesel',
            'plug-in diesel': 'el + diesel',
            'hydrogen': 'hydrogen'
        }
        brand_mapping = {'tesla motors': 'tesla',
                         'bmw i': 'bmw',
                         'alfa': 'alfa romeo',
                         'jaguar land rover limited': 'land rover',
                         'automobili lamborghini s.p.a.': 'lamborghini',
                         'land': 'land rover',
                         'rover': 'land rover',
                         'range rover': 'land rover',
                         'mercedes sprinter / kegger': 'mercedes-benz',
                         'mercedes-amg': 'mercedes-benz',
                         'ford-cng-technik': 'ford',
                         'daimler': 'mercedes-benz',
                         'kg mobility': 'kgm',
                         'mitsubishi fuso': 'mitsubishi',
                         'jaguar cars limited': 'jaguar'}

        for col in ['fuel', 'brand']:
            # logger.debug(f'Processing column: {col}')
            df.loc[:, col] = df[col].str.lower()
            if col == 'fuel':
                df.loc[:, col] = df[col].map(fuel_mapping).fillna('annet')
            elif col == 'brand':
                df.loc[:, col] = df[col].replace(brand_mapping)

        self.logger.debug(f'Length: {len(df)} | after category mapping')

        df = self.fill_na(df,feature='gearbox', fill_value='Automat')
        df = self.fill_na(df,feature='body_type', fill_value='SUV/Offroad')
        df = self.fill_na(df,feature='sales_type', fill_value='Bruktbil til salgs')
        df = self.fill_na(df,feature='dealer', fill_value='private')
        df = self.fill_na(df,feature='known_issues', fill_value='False')
        df = self.fill_na(df,feature='major_repairs', fill_value='False')
        df = self.fill_na(df,feature='engine_tuned', fill_value='False')
        df = self.fill_na(df,feature='liens', fill_value=False)
        df.loc[:, 'liens'] = df['liens'].astype('object').astype(str)

        self.logger.debug(f'Length: {len(df)} | after fill_na')

        keys_carparts = ['delebil', 'rep objekt', 'reparasjonsobjekt','reparasjonsobjekt', 'repobjekt', 'restaureringsobjekt',
                         'reparasjons objekt', 'motorhavari', 'motor havari', 'motorhavart',
                         'motor starter ikke', 'motoren starter ikke', 'motorhavarert', 'motor havarert',
                         'motorfeil', 'motor feil', 'chippet', 'defekt motor', 'rådebank',"rep-objekt"]
        keys_regreim = ['regreim', 'reg.reim', 'reg reim', 'reg.reim byttet', 'registerreim',
                        'registerreim byttet', ]

        df = self.mk_bool_description(df, 'car_for_parts', keys_carparts, source_cols=['description'])
        df = self.mk_bool_description(df, 'timing_belt', keys_regreim, source_cols=['description'])

        # features from equipment
        equipment_features = {
            'eq_cruise_control': ['cruisekontroll', 'cruisekontroll adaptiv', 'cruise controll adaptive',
                                  'adaptiv cruisekontroll', 'adaptive cruise control', 'adaptiv cruise control'],
            'eq_parking_sensor_behind': ['parkeringsensor bak', 'avstandsføler bak', 'parkeringssensorer: bak',
                                         'parkeringssensor bak'],
            'eq_parking_sensor_front': ['parkeringsensor foran', 'avstandsføler foran', 'parkeringssensorer: foran',
                                        'parkeringssensor foran'],
            'eq_parking_assistant': ['parkeringsassistent'],
            'eq_driving_computer': ['kjørecomputer'],
            'eq_rear_view_camera': ['ryggekamera'],
            'eq_bluetooth': ['bluetooth', 'bluetooth: mobil', 'bluetooth handsfree', 'bluetooth®-forbindelse',
                             'bluetooth telefon og streaming'],
            'eq_tow_hitch': ['hengerfeste', 'henger feste', 'avtagbart hengerfeste', 'tilhengerfeste'],
            'eq_360_camera': ['360 kamera', '360-kamera', '360 graders overvåkning'],
            'eq_internet': ['internett', 'internet', 'internettilkobling'],
            'eq_app': ['app-tilkobling', 'app'],

            'eq_skin_interior': ['skinninteriør', 'skinn seter', 'skinnseter', 'skinn', 'seter i helskinn', 'skinn'],
            'eq_air_conditioning': ['klimaanlegg', 'aircondition', 'air condition', 'klimaanlegg bak',
                                    'klimaanlegg foran', 'automatisk klimaanlegg', 'ventilasjon'],
            'eq_dab': ['dab-radio', 'dab radio', 'dab'],
            'eq_apple_carplay': ['apple carplay'],
            'eq_TCS': ['antispinn'],
            'eq_ESC': ['antiskrens'],
            'eq_sunroof': ['soltak', 'takluke', 'soltak/glasstak', 'panorama glasstak'],
            'eq_traffic_sign_recognition': ['trafikkskiltgjenkjenning', 'veiskiltinformasjon'],

            'eq_led_lights': ['led hovedlys', 'ledlys', 'led lys', 'lys led kjørelys'],
            'eq_xenon_lights': ['xenonlys', 'bi-xenon', 'xenon'],
            'eq_alarm': ['alarm', 'tyverialarm'],
            'eq_high_beam_assistant': ['fjernlysassistent'],

            'eq_keyless_start': ['nøkkelløs start', 'nøkkelfri lås/start'],
            'eq_keyless_go': ['nøkkelløs adgang', 'keyless go', 'nøkkelløs åpning'],

            'eq_abs': ['abs-bremser', 'abs'],
            'eq_isofix': ['isofix', 'isofix barnesetefesting', 'isofix: bak', 'isofix-monteringspunkter'],
            'eq_roof_railing': ['takrails', 'takstativ', 'takreling'],
            'eq_roof_box': ['takboks'],
            'eq_pollen_filter': ['pollenfilter', 'pollen filter'],
            'eq_airbag': ['airbags', 'kollisjonspute', 'kollisjonsputer', 'airbag foran side', 'airbag bak side'],
            'eq_power_steering': ['servostyring'],
            'eq_metallic_paint': ['metallisk lakk', 'metallakk'],
            'eq_handsfree': ['handsfree', 'handsfree opplegg', 'bluetooth handsfree'],
            'eq_alloy_wheels_summer': ['lettmet. felg sommer'],
            'eq_alloy_wheels_winter': ['lettmet. felg vinter'],
            'eq_navigation': ['navigasjonssystem'],

            'eq_charging_cable': ['laderkabel', 'lader kabel', 'ladekabel', 'ladeledning'],

            'eq_central_locking': ['sentrallås', 'fjernstyrt sentrallås'],
            'eq_rain_sensor': ['regnsensor'],

            'eq_heated_seats': ['oppvarmede seter', 'oppvarmede seter foran', 'oppvarmede seter bak',
                                'oppvarmede forseter', 'oppvarmede bakseter', 'varme i seter', 'setevarme foran',
                                'setevarme bak', 'setevarmer foran', 'setevarmer bak'],
            'eq_heated_steering_wheel': ['oppvarmet ratt', 'varmeratt', 'ratt oppvarmet'],
            'eq_electric_tailgate': ['elektrisk bakluke', 'el. bakluke', 'el bakluke', 'elektrisk bakdør'],
            'eq_electric_mirrors': ['el.mirroer', 'el speil', 'elektriske speil'],
            'eq_electric_windows': ['el.vinduer', 'el vinduer', 'elektriske vinduer',
                                    'vindusheiser elektriske foran og bak'],
            'eq_electric_seats': ['elektrisk seteregulering foran', 'elektrisk seteregulering bak',
                                  'el. seter foran', 'el. seter bak', 'elektriske seter', 'elektrisk sete'],
            'eq_heated_windshield': ['oppvarmet frontrute', 'oppvarmet vindu', 'oppvarmet vindusvisker'],

            'eq_winter_tires': ['vinterhjul', 'vinterdekk'],
            'eq_summer_tires': ['sommerhjul', 'sommerdekk'],
            'eq_multi_function_stearing_wheel': ['multifunksjonsratt'],
            'eq_block_heater': ['motorvarmer']
        }
        df = self.mk_bool_features(df, equipment_features, source_col='features')

        date_fields = ["last_eu", "next_eu","last_updated"]
        for field in date_fields:
            df.loc[:,field] = df[field].apply(lambda x: self.extract_datetime(x) if isinstance(x, str) else x)
            df[field] = pd.to_datetime(df[field])
        add_if_missing = ['web', 'email', 'warranty', 'color_interior', 'gearbox_type', 'warranty_length',
                          'condition_report']
        df = self.add_missing_features(df, add_if_missing)

        trouble_columns = ["last_eu","next_eu"]
        for col in trouble_columns:
            df[col] = df[col].astype(str)

        self.logger.debug(f'Length: {len(df)} | before saving to BQ. Replace {self.replace}')
        self.df = df
        return df

    def clean_homes(self) -> pd.DataFrame:
        df = self.transform_nan(self.df)
        df = self.rm_empty_features(df)
        self.logger.debug(f'Length: {len(df)} | after merge with sales time')

        int_cols = [
            'price', 'balcony', 'total_price', 'bedrooms', 'rooms', 'build_year',
            'usable_area', 'internal_area', 'external_area', 'plot_size', 'fees',
            'joint_debt', 'monthly_common_cost', 'collective_assets', 'tax_value', 'floor'
        ]
        df = self.mk_num(df, int_cols, type='int')

        cadastre_fields = ["cadastral_num", "unit_num", "section_num","coop_unit_num","municipality_num","coop_org_num","leasehold_num"]

        def fix_cadastre(x):
            val = ast.literal_eval(x)
            if isinstance(val, list):
                for e in val:
                    res = self.extract_int(e)
                    if res is not None:
                        return res
            else:
                self.logger.warning(f'Unexpected dtype {type(val)} | {val} for cadastre field. Expecting list')
                return val
        for field in cadastre_fields:
            df.loc[:,field] = df[field].apply(lambda x: fix_cadastre(x) if isinstance(x, str) else x)
            df[field] = pd.to_numeric(df[field], errors='coerce')

        df['total_price'] = df['total_price'].astype('float')
        df['total_price'] = df['total_price'].fillna(df['price'].astype('float') * 1.025)
        # If you want to keep it as Int64 after filling:
        df['total_price'] = df['total_price'].round().astype('Int64')
        df.loc[:, 'bedrooms'] = df['bedrooms'].fillna(0)
        df['bedrooms'] = df['bedrooms'].fillna(0)
        df['external_area'] = df['external_area'].fillna(0)
        df['internal_area'] = df['internal_area'].fillna(df['usable_area'] - df['external_area'])
        df['usable_area'] = df['usable_area'].fillna(df['internal_area'])
        df['internal_area'] = df['internal_area'].fillna(df['usable_area'])
        df['floor'] = df['floor'].fillna(0)

        df = self.mk_fractions(df=df, new_feat_name='price_pr_sqm',numerator='price',denominator='usable_area')
        df = self.mk_fractions(df=df, new_feat_name='price_pr_i_sqm', numerator='price', denominator='internal_area')
        df = self.mk_fractions(df=df, new_feat_name='price_pr_bedroom',numerator='price',denominator='bedrooms')
        df = self.mk_fractions(df=df, new_feat_name='sqm_pr_bedroom',numerator='usable_area',denominator='bedrooms')
        df = self.mk_fractions(df=df, new_feat_name='monthly_common_cost_pr_sqm', numerator='monthly_common_cost', denominator='usable_area')
        df['sqm_pr_bedroom'] = df['sqm_pr_bedroom'].fillna(df['usable_area'])

        df.drop('primary_area', axis=1, inplace=True, errors='ignore')

        df = df[(df['price'] > 200000) & (df['price'] < 30000000)
                & ((df['usable_area'] > 0) & (df['usable_area'] < 1500))
                & ((df['internal_area'] > 0) & (df['internal_area'] < 1500))
                & ((df['bedrooms'] >= 0) & (df['bedrooms'] < 10))
                & ((df['floor'] >= 0) & (df['floor'] < 100))
                & ((df['total_price'] >= 0) & (df['total_price'] < 35000000))
                ]
        self.logger.debug(f'Length: {len(df)} | after filter price and usable_area')

        df['scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce', utc=True)
        df['clean_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length: {len(df)} | after datetime conversion')

        df['postal_code'] = df['address'].apply(self.extract_postnummer)
        df = pd.merge(df, self.geo[['postal_code', 'municipality', 'county', 'region']], how='left', on='postal_code')
        self.logger.debug(f'Length: {len(df)} | after geo')

        df['ownership_type'] = df['ownership_type'].str.replace(r'^eieform', '', case=False, regex=True).str.strip()
        df['ownership_type'] = df['ownership_type'].apply(
            lambda x: x.replace('(Selveier)', "") if isinstance(x, str) else x)
        df['property_type'] = df['property_type'].str.replace(r'^boligtype', '', case=False, regex=True)
        df['property_type'] = df['property_type'].apply(lambda x: x.replace('/', "_") if isinstance(x, str) else x).str.strip()
        df['dealer'] = df['dealer'].fillna('private').str.strip()

        equipment_features = {
            'eq_parking': ["parkering", "p-plass", "parkeringsplass", "garasje/p-plass"],
            "eq_lift": ["heis", "løfteplattform"],
            "eq_fireplace": ['peis', "peis/ildsted"],
            "eq_charging_possibility": ["lademulighet"],
            "eq_aircondition": ["aircondition"],
            "eq_garden": ["hage"],
            "eq_pool": ["badebasseng", "basseng", "boblebad", "jacuzzi"]
        }
        df = self.mk_bool_features(df=df, equipment_features=equipment_features, source_col="facilities")

        df = self.mk_bool_description(df=df,
                                   col_name="eq_parking_tmp",
                                   keys=["parkering", "p-plass", "parkeringsplass", "garasje/p-plass"],
                                   source_cols=["description", "title"])
        df = self.mk_bool_description(df=df,
                                   col_name="eq_lift_tmp",
                                   keys=["heis", "løfteplattform"],
                                   source_cols=["description", "title"])
        df = self.mk_bool_description(df=df,
                                   col_name="eq_pool_tmp",
                                   keys=["badebasseng", "basseng", "boblebad", "jacuzzi"],
                                   source_cols=["description", "title"])

        df['eq_parking'] = df.apply(lambda row: True if row['eq_parking_tmp'] or row['eq_parking'] else False, axis=1)
        df['eq_lift'] = df.apply(lambda row: True if row['eq_lift_tmp'] or row['eq_lift'] else False, axis=1)
        df['eq_pool'] = df.apply(lambda row: True if row['eq_pool_tmp'] or row['eq_pool'] else False, axis=1)
        df.drop(columns=['eq_parking_tmp', 'eq_lift_tmp', "eq_pool_tmp"], inplace=True, errors="ignore")

        keys_fixer_upper = [
            'oppussingsobjekt', 'oppussingsbehov', 'oppussingsklar', 'renoveringsobjekt', 'oppgraderingsobjekt',
            'renoveringsbehov', 'moderniseringsbehov', 'moderniseringsobjekt', 'oppgraderingsbehov',
            'rehabilitering/oppussing', 'rehabilitering', 'oppussing', 'oppgradering og vedlikehold',
            'rehabiliteringsbehov'
        ]
        keys_renovated = [
            'nylig oppusset', 'nyoppusset', 'totalrenovert', 'pusset opp', 'renovert',
            'totalrehabilitert', "oppusset"
        ]
        df = self.mk_bool_description(df=df,
                                   col_name='fixer_upper',
                                   keys=keys_fixer_upper,
                                   source_cols=['description', 'title'])
        df = self.mk_bool_description(df=df,
                                   col_name="renovated",
                                   keys=keys_renovated,
                                   source_cols=['description', 'title'])
        df = self.mk_bool_description(df=df,
                                   col_name='eq_rental_unit',
                                   keys=["utleiedel", "utleiebolig", "utleieenhet", "utleie del", "utleie bolig",
                                         "utleie enhet", "anneks"],
                                   source_cols=['description', 'title'])
        df = self.mk_bool_description(df=df,
                                   col_name="eq_west_facing",
                                   keys=["vestvendt terrasse", "vestvendt", "vestvendt balkong"],
                                   source_cols=["description", "title"])
        df = self.mk_bool_description(df=df,
                                   col_name="eq_sauna",
                                   keys=["badstue", "sauna"],
                                   source_cols=["description", "title"])

        if 'sold' in df.columns:
            df['sold'] = df['sold'].str.lower().str.strip()
            df['sold'] = df['sold'].apply(lambda x: True if pd.notna(x) and x == 'solgt' else False)
            df['sold'] = df['sold'].fillna(False)
            df['sold'] = df['sold'].astype('boolean')
        self.logger.debug(f'Length: {len(df)} | after boolean')

        df.loc[:, "last_updated"] = df["last_updated"].apply(
            lambda x: self.extract_datetime(x) if isinstance(x, str) else x)
        df["last_updated"] = pd.to_datetime(df["last_updated"])

        extra_cols = ['email', 'web']
        df = self.add_missing_features(df, extra_cols)

        df = self.ensure_num_types(df, num_types=['int', 'float'])

        self.logger.debug(f'Length: {len(df)} | before saving to BQ. Replace {self.replace}')
        self.df = df
        return df

    def clean_rentals(self) -> pd.DataFrame:
        # ## Remove empty feature and missing data
        df = self.transform_nan(self.df)
        df = self.rm_empty_features(df)

        #self.logger.debug(f'Num cols before transform: \n {df[["monthly_rent", "bedrooms", "floor", ]].head(10)}')
        # ## Int and Float data
        int_cols = ['monthly_rent', 'deposit', 'bedrooms', 'floor',
                    'primary_area',
                    'usable_area', 'internal_area', 'gross_area', 'primary_area',
                    'external_area'
                    ]
        df = self.mk_num(df, int_cols, type='int')

        #self.logger.debug(f'Num cols After transform: \n {df[["monthly_rent", "bedrooms", "floor", ]].head(10)}')

        df.loc[:, 'bedrooms'] = df['bedrooms'].fillna(0)
        df.loc[:, 'floor'] = df['floor'].fillna(0)

        if 'primary_area' in df.columns and 'usable_area' in df.columns:
            df.loc[:, 'primary_area'] = df['primary_area'].fillna(df['usable_area'])
        df.loc[:, 'primary_area'] = df['primary_area'].fillna(0)

        df = self.mk_fractions(df, new_feat_name='rent_pr_sqm', numerator='monthly_rent', denominator='primary_area')
        df = self.mk_fractions(df, new_feat_name='price_pr_bedroom', numerator='monthly_rent', denominator='bedrooms')
        df = self.mk_fractions(df, new_feat_name='sqm_pr_bedroom', numerator='primary_area', denominator='bedrooms')
        df['rent_pr_sqm'] = df['rent_pr_sqm'] * 12
        df['price_pr_bedroom'] = df['price_pr_bedroom'] * 12

        df = df[(df['monthly_rent'] > 1000) & (df['monthly_rent'] < 300000)
                & ((df['primary_area'] >= 0) & (df['primary_area'] < 1000))
                & ((df['bedrooms'] >= 0) & (df['bedrooms'] < 10))
                & ((df['floor'] >= 0) & (df['floor'] < 100))
                ]
        self.logger.debug(f'Length: {len(df)} | after filter price and usable_area')

        # Datetime Data
        df.loc[:, 'scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce', utc=True)
        df['clean_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length: {len(df)} | after datetime conversion')

        df.loc[:, 'postal_code'] = df['address'].apply(self.extract_postnummer)
        df = pd.merge(df, self.geo[['postal_code', 'municipality', 'county', 'region']], how='left', on='postal_code')
        self.logger.debug(f'Length: {len(df)} | after geo')
        # ## Categorical Data
        df['includes'] = df['includes'].apply(lambda x: x.replace("inkluderer", "") if isinstance(x, str) else x)

        #Categorical data
        df.loc[:, 'property_type'] = df['property_type'].str.replace(r'^boligtype', '', case=False, regex=True)
        # df.loc[:,'property_type'] = df['property_type'].str.replace(r'\s*/småbruk', '', case=False, regex=True)
        # df['property_type'] = df['property_type'].str.replace(r'\s*/flermannsbolig', '', case=False, regex=True)
        # df.loc[:,'property_type'] = df['property_type'].str.replace(r'/$', '', regex=True)
        df['property_type'] = df['property_type'].apply(lambda x: x.replace('/', "_") if isinstance(x, str) else x)
        df.loc[:, 'dealer'] = df['dealer'].fillna('private')

        def clean_includes(text):
            if not isinstance(text, str):
                return text

            text = text.lower().strip()
            # Handle specific cases first (longer strings)
            replacements = [
                ("inkludererbredbånd", "internett"),
                ("inkludererinternett", "internett"),
                ("inkludererstrøm", "strøm"),
                ("inkluderervarmtvann", "varmtvann"),
                # Then do the general replacement
                ("inkluderer", "")
            ]

            for old, new in replacements:
                text = text.replace(old, new)

            return text

        df['includes'] = df['includes'].apply(clean_includes)
        bool_dict = {'eq_power': ['strøm', 'strøm og internett'],
                     'eq_internet': ['internett', 'bredbånd', 'fiber', 'strøm og internett', 'wifi',
                                     'grunnpakke internett og grunnpakke tv er inkludert i leien.', 'internet',
                                     'tv og internett'],
                     'eq_tv': ['tv', 'kabel-tv', 'kabel-/digital-tv', 'kabel tv', 'kabeltv',
                               'grunnpakke internett og grunnpakke tv er inkludert i leien.', 'tv og internett'],
                     'eq_hot_water': ['varmtvann', 'varmt vann'],
                     'eq_water': ['vann'],
                     'eq_heating': ['oppvarming', 'oppvarming fra vannbåren varme'],
                     'eq_parking': ['parkering', 'parkeringsplass'],
                     'eq_household_appliances': ['hvitevarer', 'vaskemaskin', 'tørketrommel', 'oppvaskmaskin'],
                     'eq_furniture': ['møbler', 'møblert'], }

        df = self.mk_bool_features(df=df, equipment_features=bool_dict, source_col='includes')
        self.logger.debug(f'Length: {len(df)} | after bool features')

        df.loc[:, "last_updated"] = df["last_updated"].apply(
            lambda x: self.extract_datetime(x) if isinstance(x, str) else x)
        df["last_updated"] = pd.to_datetime(df["last_updated"])


        extra_cols = ['email', 'web', 'energy_rating']
        self.add_missing_features(df, extra_cols)

        self.df = df
        return df

    ## PRE_PROCESSING METHODS
    def pre_process_cars(self,df = None,save_to_bq = True) -> tuple:
        if df:
            self.df = df
        df = self.df.dropna(subset='item_id')
        df.drop_duplicates(subset=['item_id'], inplace=True)
        df.set_index('item_id', inplace=True)
        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on item_id')
        df.dropna(subset=['total_price', 'mileage', 'model_year'], inplace=True)

        drop = ['address', 'description', 'contact_person', 'phone', 'url', 'country', 'title', 'web', 'email',
                'municipality', 'rn', 'FIRST', 'LAST', 'postal_code', 'region', 'salgstid'
            , 'vin', 'reg_num', 'co2', 'color_description', 'first_registration', 'cargo_space', 'prev_owners',
                'last_eu', 'next_eu', 'trailer_weight', 'subtitle', 'warranty_until', 'known_issues', 'engine_tuned',
                'liens', 'major_repairs', 'state', 'battery', 'price_excl_transfer', 'clean_date', 'warranty',
                'color_interior', 'gearbox_type', 'warranty_length', 'condition_report','warranty'
                ]
        df.drop(columns=drop, inplace=True, errors='ignore')

        drop_eq = [x for x in df.columns if x.startswith('eq_') if x not in
                   ['eq_rear_view_camera', 'eq_bluetooth', 'eq_tow_hitch', 'eq_360_camera', 'eq_cruise_control',
                    'eq_parking_sensor_behind',
                    'eq_parking_sensor_front', 'eq_air_conditioning', 'eq_navigation', 'eq_winter_tires',
                    'eq_summer_tires',
                    'eq_skin_interior', 'eq_apple_carplay', 'eq_led_lights', 'eq_xenon_lights', ]]
        df.drop(columns=drop_eq, inplace=True, errors='ignore')

        df = self.rm_empty_features(df)
        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on price, usable_area and bedrooms')

        # ## DUMMY VARIABLES
        dummy_cols = [
            'gearbox',
            'fuel',
            'color',
            'wheel_drive',
            'body_type',
            'sales_type',
            'category',
            'brand',
            # 'model',
            'county',
        ]
        fuel_mapping = {
            'diesel': 'diesel',
            'bensin': 'bensin',
            'el': 'el',
            'elektrisitet': 'el',
            'el + bensin': 'el + bensin',
            'hybrid bensin': 'el + bensin',
            'plug-in bensin': 'el + bensin',
            'el + diesel': 'el + diesel',
            'hybrid diesel': 'el + diesel',
            'plug-in diesel': 'el + diesel',
            'hydrogen': 'hydrogen'
        }
        body_type_mapping = {
            'suv/offroad': 'flerbruksbil (af)',
            'stasjonsvogn': 'stasjonsvogn (ac)',
            'kombi 5-dørs': 'kombikupé (ab)',
            '5': 'kombikupé (ab)',
            'kasse': 'integrert førerhus (bb)',
            'sedan': 'sedan (aa)',
            'annet': 'annet',
            'flerbruksbil': 'flerbruksbil (af)',
            'coupe': 'kupé (ad)',
            'cabriolet': 'kabriolet (ae)',
            'pickup': 'pick-up (be)',
            'kombi 3-dørs': 'kombikupé (ab)',  # endret fra 'kupé (ad)'
            '3': 'kombikupé (ab)'  # endret fra 'kupé (ad)'
        }
        brand_mapping = {'tesla motors': 'tesla',
                         'bmw i': 'bmw',
                         'alfa': 'alfa romeo',
                         'jaguar land rover limited': 'land rover',
                         'automobili lamborghini s.p.a.': 'lamborghini',
                         'land': 'land rover',
                         'rover': 'land rover',
                         'range rover': 'land rover',
                         'mercedes sprinter / kegger': 'mercedes-benz',
                         'mercedes-amg': 'mercedes-benz',
                         'ford-cng-technik': 'ford',
                         'daimler': 'mercedes-benz',
                         'kg mobility': 'kgm',
                         'mitsubishi fuso': 'mitsubishi',
                         'jaguar cars limited': 'jaguar'}
        for col in ['fuel', 'body_type', 'brand']:
            # logger.debug(f'Processing column: {col}')
            df.loc[:, col] = df[col].str.lower()
            if col == 'fuel':
                df.loc[:, col] = df[col].map(fuel_mapping).fillna('annet')
            elif col == 'body_type':
                df.loc[:, col] = df[col].map(body_type_mapping).fillna('annet')
            elif col == 'brand':
                df.loc[:, col] = df[col].replace(brand_mapping)

        df['dealer'] = df['dealer'].apply(lambda x: False if x.lower() == 'private' else True)
        df = self.process_bool(df)

        self.logger.debug(f'Length of df: {len(df)} | after mapping categories and creating dummy variables')

        col_valid_values = {
            'fuel': ['el + bensin', 'diesel', 'el', 'bensin'],
            'gearbox': ['automat', 'manuell'],
            'color': ['svart', 'grå', 'rød', 'blå', 'sølv', 'grønn', 'brun', 'hvit'],
            'wheel_drive': ['forhjulsdrift', 'firehjulsdrift', 'bakhjulsdrift'],
            'body_type': [
                'flerbruksbil (af)',
                'stasjonsvogn (ac)',
                'kombikupé (ab)',
                'integrert førerhus (bb)',
                'sedan (aa)',
                'pick-up (be)',
                'kupé (ad)',
                'annet'
            ],
            'sales_type': ['bruktbil til salgs', 'auksjon', 'nybil til salgs'],
            'brand': [
                'ford', 'skoda', 'bmw', 'volvo', 'peugeot', 'mercedes-benz',
                'volkswagen', 'kia', 'toyota', 'audi', 'porsche', 'citroen',
                'opel', 'nissan', 'renault', 'hyundai', 'mazda', 'tesla',
                'mitsubishi', 'suzuki'
            ],
            'county': [
                'rogaland', 'innlandet', 'buskerud', 'akershus', 'møre og romsdal',
                'vestfold', 'oslo', 'vestland', 'nordland', 'trøndelag', 'agder',
                'østfold', 'telemark', 'troms'
            ],
            'category': ['personbil', 'varebil']
        }
        for col, valid_values in col_valid_values.items():
            df = self.mk_cat(df, col, valid_values)

        self.logger.debug(f'Length of df: {len(df)} | after mapping and removing unwanted categories')

        df = df[df['sales_type'] != 'leasing']
        dummy_cols = [
            'gearbox',
            'fuel',
            'color',
            'wheel_drive',
            'body_type',
            'sales_type',
            'category',
            'brand',
            # 'model',
            'county',
            'dealer'
            'car_for_parts'
        ]
        # df = pd.get_dummies(df,columns=['dealer'],drop_first=True)
        self.logger.debug(f'Length of df: {len(df)} | after dummy variables')

        df['n_features'] = df['features'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        df.drop(columns=['features'], inplace=True, errors='ignore')
        # # Step 1: Find the 10 most common features
        # features_lists = df['features'].dropna().apply(lambda x: [f.strip() for f in x.split(',')])
        # feature_counts = Counter(chain.from_iterable(features_lists))
        # top_10_features = set([f for f, _ in feature_counts.most_common(10)])
        #
        # # Step 2 & 3: Filter and update the features column
        # def filter_top_features(feature_str):
        #     if pd.isna(feature_str):
        #         return feature_str
        #     features = [f.strip() for f in feature_str.split(',')]
        #     filtered = [f for f in features if f in top_10_features]
        #     return ', '.join(filtered) if filtered else np.nan
        #
        # df['features'] = df['features'].apply(filter_top_features)
        # %% md
        # ## DATE COLUMNS
        df = self.split_date(df, date_col='scrape_date')
        df = self.split_date(df, date_col='last_updated')
        df['pre_processed_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length of df: {len(df)} | after date columns')
        # ## ENSURE CORRECT DATA TYPES
        df = self.ensure_num_types(df, num_types=['int','float'])

        df_el = df[df['fuel'] == 'el']
        df_fossil = df[df['fuel'] != 'el']

        df_el.dropna(subset=['range'], inplace=True)
        df_el.drop(columns=['engine_volume'], inplace=True, errors='ignore')
        self.logger.debug(f'Length of df_el: {len(df_el)} | before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df=df_el,
                           table_name=f'{self.dataset}_el')


        df_fossil = df_fossil.drop(columns=['range'], errors='ignore')
        self.logger.debug(f'Length of df_fossil: {len(df_fossil)} | Before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df=df_fossil,
                           table_name=f'{self.dataset}_fossil')

        return df_el, df_fossil
    
    def pre_process_homes(self,df = None,save_to_bq = True):
        if df:
            self.df = df
        df = self.df.dropna(subset='item_id')
        df.drop_duplicates(subset=['item_id'], inplace=True)
        df.set_index('item_id', inplace=True)

        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on item_id')
        df.dropna(subset=['price', 'usable_area', 'bedrooms'], inplace=True)

        drop = ['district', 'address', 'title', 'sold',
                'description', 'email', 'contact_person', 'phone', 'url', 'new', 'country',
                'facilities', 'energy_rating',
                'rn', 'FIRST', 'LAST', 'postal_code', 'municipality', 'county', 'region', 'salgstid',
                'facilities', 'tax_value',
                'total_price', 'price_pr_bedroom', 'price_pr_sqm', 'web', 'cadastral_num', 'unit_num', 'section_num',
                'clean_date','price_pr_i_sqm','fees', "coop_name","apartment_num"
                #'internal_area'
                ]
        df.drop(columns=drop, inplace=True,errors='ignore')

        df = self.rm_empty_features(df)
        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on price, usable_area and bedrooms')
        # ## DUMMY VARIABLES
        df = df[df['property_type'] != 'Garasje/Parkering']
        df['dealer'] = df['dealer'].apply(lambda x: False if x.lower() == 'private' else True)

        df = self.process_bool(df)

        prop_type = ['Leilighet',
                     'Enebolig',
                     'Tomannsbolig',
                     'Rekkehus',
                     'Gårdsbruk_Småbruk',
                     'Andre',
                     'Bygård_Flermannsbolig']
        own_type = ['Eier ', 'Andel', 'Aksje', 'Annet', 'Obligasjon']

        df = self.mk_cat(df, 'property_type', prop_type)
        df = self.mk_cat(df, 'ownership_type', own_type)

        df = pd.get_dummies(df, columns=['ownership_type'], drop_first=True)
        self.logger.debug(f'Length of df: {len(df)} | after dummy variables')

        # ## DATE COLUMNS
        df = self.split_date(df, date_col='scrape_date')
        df = self.split_date(df,date_col='last_updated')
        df['pre_processed_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length of df: {len(df)} | after date columns')
        # ## ENSURE CORRECT DATA TYPES
        df = self.ensure_num_types(df, num_types=['int'])
        # ## SPLIT INTO APARTMENTS AND HOUSES AND RENTALS
        df_a = df[df['property_type'] == 'leilighet']
        df_h = df[df['property_type'] != 'leilighet']
        rental_cols = ['property_type', 'bedrooms', 'floor', 'usable_area', 'day', 'month', 'year', 'sqm_pr_bedroom']
        df_r = df[rental_cols]
        self.logger.debug(
            f'Split into apartments, houses and rentals: {len(df_a)} | {len(df_h)} | {len(df_r)}. Total length: {len(df_a) + len(df_h)}')
        # ## APARTMENTS
        df_a = pd.get_dummies(df_a, columns=['property_type'], drop_first=True)
        df_a.loc[:, 'joint_debt'] = df_a['joint_debt'].fillna(0)
        df_a.loc[:, 'collective_assets'] = df_a['collective_assets'].fillna(0)
        df_a.loc[:, 'balcony'] = df_a['balcony'].fillna(0)
        df_a.loc[:, 'floor'] = df_a['floor'].fillna(0)
        df_a.loc[:, 'rooms'] = df_a['rooms'].fillna(df_a['bedrooms'] + 1)
        df_a.loc[:, 'external_area'] = df_a['external_area'].fillna(0)
        df_a.loc[:, 'monthly_common_cost'] = df_a['monthly_common_cost'].fillna(0)
        self.logger.debug(f'Length of df_a: {len(df_a)} | before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df_a,'homes_apartments')
        # ## HOUSES
        drop = ['collective_assets', 'joint_debt', 'balcony', 'floor',
                'monthly_common_cost', 'rooms', 'external_area']
        df_h.drop(columns=drop, inplace=True)
        df_h = pd.get_dummies(df_h, columns=['property_type'], drop_first=True)
        self.logger.debug(f'Length of df_h: {len(df_h)} | before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df_h,'homes_houses')
        # %% md
        # ## RENTAL PREDICTION
        df_r.rename({'usable_area': 'primary_area'}, axis=1, inplace=True)
        order = ['bedrooms', 'floor',
                 'primary_area', 'sqm_pr_bedroom',
                 'day', 'month', 'year', 'property_type']
        df_r = df_r[order]
        prop_type = ['Enebolig',
                     'Leilighet',
                     'Tomannsbolig',
                     'Andre',
                     'Rekkehus']
        df_r = self.mk_cat(df_r,'property_type', prop_type)
        df_r = pd.get_dummies(df_r, columns=['property_type'], drop_first=True)

        self.logger.debug(f'Length of df_r: {len(df_r)} | before saving to BQ. Replace is {self.replace}')
        # %%
        if save_to_bq:
            self.save_data(df_r,'homes_rentals')

        return df_a, df_h, df_r
            
    def pre_process_rentals(self,df = None,save_to_bq = True):
        if df:
            self.df = df
        df = self.df.dropna(subset='item_id')
        df.drop_duplicates(subset=['item_id'], inplace=True)
        df.set_index('item_id', inplace=True)
        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on item_id')
        df.dropna(subset=['monthly_rent', 'primary_area'], inplace=True)
        drop = ['address', 'description', 'contact_person', 'phone', 'url', 'country', 'title',
                'facilities', 'deposit', 'includes', 'web', 'email',
                'rn', 'FIRST', 'LAST', 'postal_code', 'municipality', 'county', 'region', 'salgstid',
                'facilities'
            , 'price_pr_bedroom', 'rent_pr_sqm', 'clean_date', 'energy_rating', 'internal_area', 'gross_area',
                'usable_area', 'clean_date']
        df.drop(columns=drop, inplace=True, errors='ignore')

        df = df[(df['monthly_rent'] > 1000) & (df['monthly_rent'] < 50000)
                & ((df['primary_area'] > 0) & (df['primary_area'] < 200))
                ]

        df = self.rm_empty_features(df)
        self.logger.debug(f'Length of df: {len(df)} | after dropping NaN on price, usable_area and bedrooms')
        # ## DUMMY VARIABLES

        prop_type = ['Enebolig',
                     'Leilighet',
                     'Rom i bofellesskap',
                     'Tomannsbolig',
                     'Hybel',
                     'Andre',
                     'Rekkehus']
        df = self.mk_cat(df, 'property_type', prop_type)
        self.logger.debug(f'Unique property types: {df["property_type"].unique()}')
        df['dealer'] = df['dealer'].apply(lambda x: False if x.lower() == 'private' else True)
        self.logger.debug(f'Length of df: {len(df)} | after mapping categories and creating categorical variables')

        df = self.process_bool(df)
        # ## DATE COLUMNS
        df = self.split_date(df, date_col='scrape_date')
        df = self.split_date(df, date_col='last_updated')
        df['pre_processed_date'] = pd.Timestamp.now()
        self.logger.debug(f'Length of df: {len(df)} | after date columns')
        # ## ENSURE CORRECT DATA TYPES
        df = self.ensure_num_types(df, num_types=['int', 'float'])

        
        df_r = df[(df['property_type'] != 'rom i bofellesskap') 
                  #& (df['property_type'] != 'hybel')
                  ]
        df_k = df[(df['property_type'] == 'rom i bofellesskap')]
        

        def rm_rental_outliers(df : pd.DataFrame) -> pd.DataFrame:
            """
            :param df
            """
            if 'bedrooms' in df.columns and 'monthly_rent' in df.columns:
                # Lag en boolsk maske for radene som skal fjernes.
                # Dette identifiserer rader hvor 'bedrooms' er 0 OG 'monthly_rent' er > 20000.
                mask_to_drop = (df['bedrooms'] == 0) & (df['monthly_rent'] > 20000)

                # Bruk '~' (tilde) for å invertere masken. Dette velger alle rader
                # hvor betingelsen er USANN, og fjerner dermed outlierne.
                df = df[~mask_to_drop]

            return df
        
        
        df_r = rm_rental_outliers(df_r)

        # ## ORDINARY RENTALS
        #df_ord = df[df['property_type'] != 'rom i bofellesskap']
        #df_ord = rm_rental_outliers(df_ord)
        #df_ord = pd.get_dummies(df_ord, columns=['property_type'], drop_first=True)
        # self.logger.debug(f'Length of df_ord: {len(df_ord)} | before saving to BQ. Replace is {self.replace}')
        # if save_to_bq:
        #     self.save_data(df = df_ord, table_name = 'rentals')
        
        # ## HOUSING RENTALS
        #drop_rentals = ['dealer_True']
        #df_r = df_r.drop(columns=drop_rentals)
        prop_type = ['Enebolig',
                     'Leilighet',
                     'Tomannsbolig',
                     'Andre',
                     'Rekkehus',
                     'Hybel']
        df_r = self.mk_cat(df_r, 'property_type', prop_type)
        df_r = pd.get_dummies(df_r, columns=['property_type'], drop_first=True)
        self.logger.debug(f'Length of df_r: {len(df_r)} | Before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df = df_r, table_name = 'rentals')
        # ## Co-living rentals
        df_k = df_k.drop(columns=['dealer_True', 'property_type', 'bedrooms', 'sqm_pr_bedroom', 'primary_area'],
                         errors='ignore')
        self.logger.debug(f'Length of df_k: {len(df_k)} | Before saving to BQ. Replace is {self.replace}')
        if save_to_bq:
            self.save_data(df = df_k, table_name = 'rentals_co-living')

        return (
            #df_ord, 
            df_r, df_k)

    def run(self,task: str ,df = None, save_to_bq: bool=True,replace: bool =False):
        if task not in ['clean', 'pre_processed']:
            raise ValueError(f'Unsupported task: {task}. Supported tasks are "clean" and "pre_processed".')
        self.logger.info(f'Running task: {task} for dataset: {self.dataset} with replace={replace} and save_to_bq={save_to_bq}')
        try:
            if replace:
                self.replace = replace
            self.task_name = task
            if self.task_name == 'clean':
                if df is None:
                    self.read_in_data()
                else:
                    self.df = df
                if self.dataset == 'cars':
                    cleaned_df = self.clean_cars()
                elif self.dataset == 'homes':
                    cleaned_df = self.clean_homes()
                elif self.dataset == 'rentals':
                    cleaned_df = self.clean_rentals()
                else:
                    raise ValueError(f'Unsupported dataset: {self.dataset}')

                if save_to_bq:
                    self.save_data(df = cleaned_df,
                                   table_name = self.dataset
                                   )
                return cleaned_df
            elif self.task_name == 'pre_processed':
                if df is None:
                    self.read_in_data()
                else:
                    self.df = df
                if self.dataset == 'cars':
                    pre_processed_df = self.pre_process_cars(save_to_bq=save_to_bq)
                    return pre_processed_df
                elif self.dataset == 'homes':
                    pre_processed_df = self.pre_process_homes(save_to_bq=save_to_bq)
                elif self.dataset == 'rentals':
                    pre_processed_df = self.pre_process_rentals(save_to_bq=save_to_bq)
                else:
                    raise ValueError(f'Unsupported dataset: {self.dataset}')
        finally:
            self.logger.shutdown()

class Train(SibrBase):
    def __init__(self, dataset,logger=None,log_target = True):
        super().__init__(dataset=dataset, logger=logger)
        self.task_name = 'train'
        self.replace = False
        self.log_target = log_target

    def load_data(self,df,target,log_target):
        """
        Splits the dataframe into features and target variable.
        """
        X = df.drop(columns=[target], axis=1)
        y = np.log1p(df[target]) if log_target else df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True)
        self.logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        self.logger.info(f"Target: {target} and log_target: {log_target}")
        #self.logger.info(f'Columns in train set: {X_train.columns.tolist()}')
        return X_train, X_test, y_train, y_test

    def save_model(self, pipeline,results: dict,data_name:str):
        model_name = pipeline.steps[-1][1].__class__.__name__
        filename_bucket = f'models/{model_name}_{data_name}.pkl'
        manifest_filename = 'models.json'
        local_manifest_path = f'/tmp/{manifest_filename}'

        # Lag en DataFrame med info om den nye modellen
        results['filename'] = filename_bucket
        results['model_type'] = model_name
        new_model_info = pd.DataFrame([results])

        try:
            self.cs.download(manifest_filename, local_manifest_path)
            all_models_df = pd.read_json(local_manifest_path,
                                         orient='records',
                                         #lines=True
                                         )

            if data_name in all_models_df['dataset'].values:
                self.logger.info(f"Oppdaterer eksisterende modell '{data_name}' i manifestet.")
                all_models_df = all_models_df[all_models_df['dataset'] != data_name]
            all_models_df = pd.concat([all_models_df, new_model_info], ignore_index=True)
            all_models_df.drop_duplicates(subset = ['dataset'],keep = 'last',inplace = True)
        except Exception as e:
            self.logger.info(f"Kunne ikke finne '{manifest_filename}'. Oppretter en ny. Feil: {e}")
            all_models_df = new_model_info

        all_models_df.to_json(local_manifest_path,
                              orient='records',
                              #lines=True
                              )
        self.cs.upload(local_manifest_path, manifest_filename)

        local_filepath = f'/tmp/tmp_file.pkl'
        joblib.dump(pipeline, local_filepath)
        try:
            self.cs.upload(local_filepath, filename_bucket)
        finally:
            os.remove(local_filepath)

    def calc_scores(self,true,pred,log_target:bool) -> tuple:
        '''

        :param true:
        :param pred:
        :param log_target:
        :return: tuple (mse,r2)
        '''
        r2 = r2_score(np.expm1(true), np.expm1(pred)) if log_target else r2_score(true, pred)
        mse = mean_squared_error(np.expm1(true), np.expm1(pred)) if log_target else mean_squared_error(true,pred)
        return mse,r2

    def get_cat(self,df):
        cat_cols = []
        cat_idx = []
        for idx, (col, dtype) in enumerate(df.dtypes.items()):
            if dtype == 'object' or dtype == 'category' or dtype == 'string':
                cat_cols.append(col), cat_idx.append(idx)
        return cat_cols, cat_idx

    def train(self,df, params, data_name, model,target,save_to_gc = True,categorical = False,log_target = None):
        self.logger.info(f'\n \nTRAINING {model().__class__.__name__} model for {data_name.upper()}')
        if not log_target:
            log_target = self.log_target
        X_train, X_test, y_train, y_test = self.load_data(df = df,target = target,log_target=log_target)

        cat_cols,cat_idx = self.get_cat(X_train)
        num_cols = X_train.select_dtypes(include=[np.number,'bool','boolean']).columns.tolist()

        num_trans = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean')),
            #('scaler', StandardScaler())
        ])
        cat_trans = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            #('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            #('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        if categorical:
            pre_processer = ColumnTransformer(
                transformers=[
                    ('num', num_trans, num_cols),
                    ('cat', cat_trans, cat_cols)
                ],remainder='passthrough')

            cat_features_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
            pipeline = Pipeline([
                ('pre_processor', pre_processer),
                ('model', CatBoostRegressor(cat_features=cat_features_indices,verbose = 0)),
            ])

            pipeline_params = {'model__' + key: value for key, value in params.items()}
            pipeline.set_params(**pipeline_params)
        else:
            pre_processer = ColumnTransformer(
                transformers=[
                    ('num', num_trans, num_cols),
                    #('cat', cat_trans, cat_cols)
                ],remainder='passthrough')
            pipeline = Pipeline([
            ('pre_processor', pre_processer),
            ('model', model()),
        ])
            pipeline.set_params(**params)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)
        mse,r2 = self.calc_scores(y_test,y_pred,log_target=log_target)
        mse_train,r2_train = self.calc_scores(y_train,y_pred_train,log_target=log_target)
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
            training_columns[col] = (str(dtype), sample)
        self.logger.info(
            f'MSE test: {mse},r2 test: {r2}, mse train: {mse_train}, r2 train {r2_train} '
            f'for {data_name} with target {target} and log_target {log_target}')
        results = {
            'dataset': data_name,
            'target' : target,
            'log_target': log_target,
            'created_at': pd.Timestamp.now(),
            'training_columns' : training_columns,
            'params': str(params),
            'r2_score': r2,
            'mse': mse,
            'mse_train': mse_train,
            'r2_train': r2_train
        }
        if save_to_gc:
            self.save_model(pipeline, results=results, data_name=data_name)
        return pipeline

    def rm_coor_outliers(self, df):
        df = df[(df['lat'] > 57.5) & (df['lat'] < 71.5)
                & (df['lng'] > 4) & (df['lng'] < 31)]
        return df

    def run(self, save_to_gc=True):
        self.logger.info(f'RUNNING {self.task_name} for dataset: {self.dataset} . Save to GC is {save_to_gc}')
        try:
            if self.dataset == 'homes':
                data = self.bq.read_homes(task = "train")
                df_a = data.get("homes_apartments")
                df_h = data.get("homes_houses")
                df_o = data.get("homes_oslo")
                params_a = {'model__learning_rate': np.float64(0.02545642421255076), 'model__max_depth': 5, 'model__n_estimators': 1060, 'model__random_state': 98, 'model__subsample': np.float64(0.6478376983753207)}

                params_h = {'model__learning_rate': np.float64(0.03233791944457942), 'model__max_depth': 3, 'model__n_estimators': 1253, 'model__random_state': 52, 'model__subsample': np.float64(0.7957811041110252)}

                params_o = {'model__learning_rate': np.float64(0.03750796359625606), 'model__max_depth': 4, 'model__n_estimators': 978, 'model__random_state': 52, 'model__subsample': np.float64(0.8347004662655393)}

                pipline_a = self.train(df = df_a,
                                       params = params_a,
                                       target='price',
                                       data_name='homes_apartments',
                                       model=XGBRegressor,
                                       save_to_gc=save_to_gc,
                                       log_target = True)
                pipeline_h = self.train(df = df_h,
                                        params = params_h,
                                        target='price',
                                        data_name='homes_houses',
                                        model=XGBRegressor,
                                        save_to_gc=save_to_gc,
                                        log_target = True)
                pipeline_o = self.train(df = df_o,
                                        params = params_o,
                                        target='price',
                                        data_name='homes_apartments_oslo',
                                        model=XGBRegressor,
                                        save_to_gc=save_to_gc,
                                        log_target = True)
                return pipline_a, pipeline_h, pipeline_o

            elif self.dataset == 'rentals':
                data = self.bq.read_rentals(task = "train")
                df_a = data.get("rentals")
                df_co = data.get("rentals_co-living")
                df_o = data.get("rentals_oslo")

                df_a = self.rm_coor_outliers(df_a)
                df_co = self.rm_coor_outliers(df_co)
                df_o = self.rm_coor_outliers(df_o)
                
                params_rentals = {'model__depth': 7, 'model__iterations': 1223, 'model__l2_leaf_reg': np.float64(1.3895264494261033), 'model__learning_rate': np.float64(0.09150423569345205), 'model__random_state': 52}
                params_co = {'model__depth': 9, 'model__iterations': 809, 'model__l2_leaf_reg': np.float64(3.241509528627272), 'model__learning_rate': np.float64(0.04135867955263894), 'model__random_state': 36}
                params_o = {'model__depth': 9, 'model__iterations': 809, 'model__l2_leaf_reg': np.float64(3.241509528627272), 'model__learning_rate': np.float64(0.04135867955263894), 'model__random_state': 36}

                pipeline_rentals = self.train(df=df_a,
                                              params=params_rentals,
                                              target = 'monthly_rent' ,
                                              data_name='rentals',
                                              model=CatBoostRegressor,
                                              save_to_gc=save_to_gc,
                                              log_target=True)

                pipeline_rental_oslo = self.train(df=df_o,
                                                  params=params_o,
                                                  target = 'monthly_rent',
                                                  data_name='rentals_oslo',
                                                  model=CatBoostRegressor,
                                                  save_to_gc=save_to_gc,
                                                  log_target = True)

                pipeline_coliv = self.train(df=df_co,
                                            params=params_co,
                                            target = 'monthly_rent',
                                            data_name='rentals_co-living',
                                            model=CatBoostRegressor,
                                            save_to_gc=save_to_gc,
                                            log_target = True)

                return pipeline_rentals, pipeline_rental_oslo, pipeline_coliv

            elif self.dataset == 'cars':
                
                data = self.bq.read_cars(task = "train")
                df_el = data.get("cars_el")
                df_fossil = data.get("cars_fossil")
                
                df_el.dropna(inplace=True)
                df_fossil.dropna(inplace=True)
                self.logger.info(f'Length dataframes after dropping NaN:  df_el: {len(df_el)} | df_fossil: {len(df_fossil)}')

                params_el = {'depth': 7, 'iterations': 1223, 'l2_leaf_reg': np.float64(1.3895264494261033),
                             'learning_rate': np.float64(0.09150423569345205), 'random_state': 52}
                params_fossil = {'depth': 4, 'iterations': 1463, 'l2_leaf_reg': np.float64(1.7419090119209117),
                                 'learning_rate': np.float64(0.10215580871496355), 'random_state': 43}

                pipeline_el = self.train(df = df_el,
                                              params = params_el,
                                              target = 'total_price',
                                              data_name =  'cars_el',
                                              model = CatBoostRegressor,
                                              save_to_gc=save_to_gc,
                                              categorical=True,
                                              log_target = True)
                pipeline_fossil = self.train(df = df_fossil,
                                                   params = params_fossil,
                                                   target = 'total_price',
                                                   data_name = 'cars_fossil',
                                                   model = CatBoostRegressor,
                                                   save_to_gc = save_to_gc
                                                   ,categorical=True,
                                                   log_target = True)
                return pipeline_el,pipeline_fossil
        finally:
            self.logger.shutdown()

class Predict(SibrBase):
    def __init__(self, dataset, logger=None,log_target = True):
        super().__init__(dataset=dataset, logger=logger)
        self.task_name = 'predictions'
        self.replace = True
        self.log_target = log_target

    def ensure_columns(self,df, training_columns,data_name = None):
        '''Ensure that the DataFrame has all columns required by the model.
        Needs to be a sklearn pipeline with a SimpleImputer as the first step.
        '''
        for col in training_columns:
            if col not in df.columns:
                self.logger.warning(f'Column {col} not in dataframe for {data_name}. Adding it.')
                df[col] = False
        return df[training_columns]
    def predict_data(self,dataframe,pipeline: sklearn.pipeline,model_results):
        log_target = model_results.get('log_target')
        if not log_target:
            log_target = self.log_target
        X = dataframe.drop(columns=[model_results.get('target')], axis=1)
        #y = dataframe[target]
        X = self.ensure_columns(df=X, training_columns=list(model_results.get('training_columns').keys()),data_name=model_results.get('dataset'))
        y_pred = np.expm1(pipeline.predict(X)) if log_target else pipeline.predict(X)
        return y_pred

    def rm_other_outliers(self,df):
        if 'bedrooms' in df.columns and 'monthly_rent' in df.columns:
            drop_item_ids = df[(df['bedrooms'] == 0) & (df['monthly_rent'] > 20000)]
            df = df[~df['item_id'].isin(drop_item_ids['item_id'])]
        return df

    def rm_coor_outliers(self,df):
        df = df[(df['lat'] > 57.5) & (df['lat'] < 71.5)
                & (df['lng'] > 4) & (df['lng'] < 31)]
        return df
    
    def run(self, save_to_bq=True):
        self.logger.info(f'RUNNING {self.task_name} for dataset: {self.dataset}. Save to BQ is {save_to_bq}')
        models_json = self.cs.download('models.json', read_in_file=True)
        models = pd.DataFrame.from_dict(models_json)
        models['created_at'] = pd.to_datetime(models['created_at'], unit='ms')
        try:
            if self.dataset == 'homes':
                data = self.bq.read_homes(task = "predict")
                df_a = data.get("homes_apartments")
                df_h = data.get("homes_houses")
                df_o = data.get("homes_oslo")
                df_r = data.get("homes_rentals_oslo")
                
                df_a = self.rm_coor_outliers(df_a)
                df_h = self.rm_coor_outliers(df_h)
                df_o = self.rm_coor_outliers(df_o)
                df_r = self.rm_coor_outliers(df_r)


                add_columns = [
                    'eq_power_True',
                    'eq_internet_True',
                    'eq_tv_True',
                    'eq_fiber_True',
                    'eq_hot_water_True',
                    'eq_heating_True',
                    'parking_True'
                ]
                for col in add_columns:
                    if col == 'eq_internet_True':
                        df_r[col] = True
                    else:
                        df_r[col] = False
                res_a = models[models['dataset'] == 'homes_apartments'].iloc[0].to_dict()
                res_h = models[models['dataset'] == 'homes_houses'].iloc[0].to_dict()
                res_o = models[models['dataset'] == 'homes_apartments_oslo'].iloc[0].to_dict()
                res_r = models[models['dataset'] == 'rentals_oslo'].iloc[0].to_dict()

                m_a = self.cs.download(res_a.get('filename'), read_in_file=True)
                m_h = self.cs.download(res_h.get('filename'), read_in_file=True)
                m_o = self.cs.download(res_o.get('filename'), read_in_file=True)
                m_r = self.cs.download(res_r.get('filename'), read_in_file=True)

                y_pred_a = self.predict_data(dataframe = df_a,
                                             pipeline = m_a,
                                             model_results=res_a,
                                             )
                y_pred_h = self.predict_data(dataframe = df_h,
                                             pipeline=m_h,
                                             model_results=res_h)
                y_pred_o = self.predict_data(df_o,
                                             pipeline = m_o,
                                             model_results=res_o,
                                             )
                df_r = self.ensure_columns(df = df_r,
                                           training_columns=list(res_r.get('training_columns').keys()),
                                           data_name='rentals_oslo'
                                           )
                y_pred_r = np.log1p(m_r.predict(df_r)) if res_r.get('log_target') else m_r.predict(df_r)

                df_a_rehab = df_a[df_a['fixer_upper_True'] == True].copy()
                df_a_rehab['fixer_upper_True'] = False
                y_pred_rehab = self.predict_data(dataframe=df_a_rehab,
                                                 pipeline = m_a,
                                                 model_results=res_a,
                                                 )
                df_o_rehab = df_o[df_o['fixer_upper_True'] == True].copy()
                df_o_rehab['fixer_upper_True'] = False
                y_pred_rehab_o = self.predict_data(dataframe = df_o_rehab,
                                                   pipeline=m_o,
                                                   model_results=res_o,
                                                   )

                pred_a = pd.DataFrame({
                    'item_id': df_a.index,
                    'predicted_price': y_pred_a,
                    'model': 'apartments'
                })
                pred_h = pd.DataFrame({
                    'item_id': df_h.index,
                    'predicted_price': y_pred_h,
                    'model': 'houses'
                })
                pred_o = pd.DataFrame({
                    'item_id': df_o.index,
                    'predicted_price': y_pred_o,
                    'model': 'apartments_oslo'
                })
                pred_r = pd.DataFrame({
                    'item_id': df_r.index,
                    'predicted_price': y_pred_r,
                    'model': 'rentals_oslo'
                })
                pred_a_rehab = pd.DataFrame({'item_id': df_a_rehab.index,
                                             'predicted_price': y_pred_rehab,
                                             'model': 'homes_rehab'})
                pred_o_rehab = pd.DataFrame({'item_id': df_o_rehab.index,
                                             'predicted_price': y_pred_rehab_o,
                                             'model': 'homes_rehab_oslo'})
                pred = pd.concat([pred_a,pred_h, pred_o, pred_a_rehab,pred_o_rehab])
                pred['predict_date'] = pd.Timestamp.now()
                pred_r['predict_date'] = pd.Timestamp.now()

                if save_to_bq:
                    if not pred.empty:
                        self.save_data(df=pred, table_name=self.dataset)
                    if not pred_r.empty:
                        self.save_data(df=pred_r, table_name='homes_rentals')
                else:
                    self.logger.warning('No data saved to BQ as save_to_bq is set to False.')

            if self.dataset == 'cars':
                data = self.bq.read_cars(task = "predict")
                df_el = data.get("cars_el")
                df_fossil = data.get("cars_fossil")

                res_el = models[models['dataset']=='cars_el'].iloc[0].to_dict()
                res_fossil = models[models['dataset']=='cars_fossil'].iloc[0].to_dict()
                m_el = self.cs.download(res_el.get('filename'), read_in_file=True)
                m_fossil = self.cs.download(res_fossil.get('filename'), read_in_file=True)

                y_pred_el = self.predict_data(dataframe = df_el,
                                              pipeline = m_el,
                                              model_results=res_el,
                                              )
                y_pred_fossil = self.predict_data(dataframe = df_fossil,
                                                  pipeline = m_fossil,
                                                  model_results=res_fossil,
                                                  )

                pred_el = pd.DataFrame({
                    'item_id': df_el.index,
                    'predicted_price': y_pred_el,
                    'model': 'el'
                })
                pred_fossil = pd.DataFrame({
                    'item_id': df_fossil.index,
                    'predicted_price': y_pred_fossil,
                    'model': 'fossil'
                })
                pred = pd.concat([pred_el, pred_fossil], ignore_index=False)
                pred['predict_date'] = pd.Timestamp.now()

                if save_to_bq:
                    if not pred.empty:
                        self.save_data(df=pred, table_name=self.dataset)
                else:
                    self.logger.warning('No data saved to BQ as save_to_bq is set to False.')

        finally:
            self.logger.shutdown()


class ParamTuning:
    def __init__(self, dataset_name: str , dataframe: pd.DataFrame,target:  str, model_params:  dict,logger : Logger = None,log_target: bool = False):
        if not logger:
            logger = Logger('model_selection')
        self.logger = logger
        self.target = target
        self.model_params = model_params
        self.all_results = []
        self.df = dataframe
        self.dataset_name = dataset_name
        self.log_target = log_target


    def load_data(self):
        """
        Splits the dataframe into features and target variable.
        """
        X = self.df.drop(columns=[self.target], axis=1)
        y = np.log1p(self.df[self.target]) if self.log_target else self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            test_size=0.2,
                                                            random_state=42)
        self.logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        self.logger.info(f'Columns in train set: {X_train.columns.tolist()}')
        return X_train, X_test, y_train, y_test

    def training_step(self,model_name: str,
                      model,
                      params, data: tuple, 
                      pipe : sklearn.pipeline.Pipeline = None,
                      n_iter = 50,
                      cv = 3,
                      scoring = 'neg_mean_squared_error',
                      ):
        X_train, X_test, y_train, y_test = data
        results = {'train_loss' : 0,
                       'test_loss' : 0,
                       'r2_score' : 0,
                       'model_name' : model_name,
                       'dataset_name' : self.dataset_name}
        self.logger.info(f'Training {model_name} model with hyperparameter tuning. Dataset: {self.dataset_name}')
        if pipe is None:
            pipe = Pipeline([
                ('impute', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('model', model),
        ])
        cv = RandomizedSearchCV(pipe,
                                    param_distributions=params,
                                    n_iter=n_iter,
                                    cv=cv,
                                    scoring=scoring,
                                    verbose=1,
                                    random_state=42,
                                    n_jobs=-1,
                                refit = scoring,
                                return_train_score=True)
        try:
            cv.fit(X_train, y_train)
            y_pred = cv.predict(X_test)
            y_pred_train = cv.predict(X_train)
            mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred)) if self.log_target else mean_squared_error(y_test, y_pred)
            mse_train = mean_squared_error(np.expm1(y_train), np.expm1(y_pred_train)) if self.log_target else mean_squared_error(y_train, y_pred_train)
            r2 = r2_score(np.expm1(y_test), np.expm1(y_pred)) if self.log_target else r2_score(y_test, y_pred)
            r2_train = r2_score(np.expm1(y_train), np.expm1(y_pred_train)) if self.log_target else r2_score(y_train, y_pred_train)
            self.logger.info(f'{model_name} model best parameters: {cv.best_params_} on {self.dataset_name}')
            self.logger.info(f'Best score for {model_name}: Train: {mse_train}, Test: {mse}, R2 (test): {r2}, R2 (train): {r2_train} on {self.dataset_name} \n')
            results['dataset_name'] = self.dataset_name
            results['model_name'] = model_name
            results['train_loss'] = mse_train
            results['test_loss'] = mse
            results['r2_test'] = r2
            results['r2_train'] = r2_train
            results['params'] = cv.best_params_
            results['best_model'] = cv.best_estimator_
            self.all_results.append(results)
        except Exception as e:
            self.logger.error(f'Error training {model_name} model: {e}')

    def model_selection(self,models : dict, cv = 3, n_iter = 50,scoring = 'neg_mean_squared_error',):

        self.logger.info(f'\n \n -------- MODEL SELECTION FOR {self.dataset_name.upper()} --------')
        X_train, X_test, y_train, y_test = self.load_data()
        for model_name, (model, params) in models.items():
            self.training_step(model_name = model_name, 
                               model = model, 
                               params=params, 
                               data = (X_train, X_test, y_train, y_test), 
                               cv = cv,
                               n_iter = n_iter,
                               scoring = scoring
                               )
        self.logger.info(f' \n ===== RESULTS FOR {self.dataset_name} =====')
        for res in self.all_results:
            self.logger.info(f'Model: {res.get("model_name")} | r2_test {res.get("r2_test")} | r2_train {res.get("r2_train")} \t | \t PARAMETERS {res.get("params")}')

    def get_feat_imp(self,model_idx : int) -> tuple:
        model = self.all_results[model_idx].get('best_model')['model']
        data = model.feature_importances_
        feat_names = self.all_results[model_idx].get('best_model')['impute'].feature_names_in_

        feat_imp = pd.DataFrame(data, index=feat_names, columns=['importance']).sort_values('importance', ascending=False)
        not_important = feat_imp[feat_imp['importance'] == 0].index.tolist()
        return feat_imp, not_important

    def plot_feat_imp(self,model_index: int,show_top_n = 15):
        feat_imp,_ = self.get_feat_imp(model_idx=model_index)
        fig = plt.figure(figsize=(16,6))
        plt.bar(x = feat_imp['importance'].index[:show_top_n], height = feat_imp['importance'][:show_top_n])
        plt.xticks(rotation=45, ha='right')
        plt.show()