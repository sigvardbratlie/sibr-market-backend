import bigframes.pandas as bf
import pandas as pd
from google.cloud import bigquery
import google.auth
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import logging as cloud_logging_client
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.logging_v2.handlers import CloudLoggingHandler
import uuid
import joblib
import pandas_gbq as pbq
import numpy as np
from typing import Literal
import traceback
import logging
from pathlib import Path
import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib




class BigQuery:
    def __init__(self,logger,CREDENTIALS_PATH=None):
        self._logger = logger
        try:
            if not CREDENTIALS_PATH:
                # Lar klienten finne default credentials og project ID selv.
                self._bq_client = bigquery.Client()
                self._project_id = self._bq_client.project
                self._credentials = self._bq_client.credentials
            else:
                self._credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
                self._project_id = self._credentials.project_id
                self._bq_client = bigquery.Client(credentials=self._credentials, project=self._project_id)

            bf.reset_session()
            bf.options.bigquery.credentials = self._credentials
            bf.options.bigquery.project = self._project_id
            self._logger.info(f"BigQuery client initialized with project_id: {self._project_id}")
        except Exception as e:
            self._logger.error(f"Error initializing BigQuery client: {e}")
            self._logger.error(traceback.format_exc())
            raise ImportError(f"Error initializing BigQuery client: {e}")
    def to_bq(self, df, table_name, dataset_name, if_exists : Literal['append','replace','merge']='append',to_str=False,merge_on=None):
        dataset_id = f'{self._project_id}.{dataset_name}'
        table_id = f"{dataset_id}.{table_name}"
        if if_exists not in ['append', 'replace', 'merge']:
            raise ValueError(f"Invalid if_exists value: {if_exists}. Choose between 'append', 'replace', or 'merge'.")
        try:
            self._bq_client.get_table(table_id)
            table_exists = True
        except Exception:
            table_exists = False

        type_mapping = {
            'object': 'STRING',
            'string': 'STRING',
            'int64': 'INTEGER',
            'Int64': 'INTEGER',
            'int64[pyarrow]': 'INTEGER',
            'float64': 'FLOAT',
            'bool': 'BOOLEAN',
            'boolean': 'BOOLEAN',
            'datetime64[ns]': 'DATETIME',
            'datetime64[ns, UTC]': 'DATETIME',
            'date32[day][pyarrow]': 'DATE',
            'datetime64[us]': 'DATETIME',
            'category': 'STRING',
        }

        explicit_schema = {
            "scrape_date": "TIMESTAMP",
        }

        schema = []
        for column_name, dtype in df.dtypes.items():
            # Hvis vi har definert en eksplisitt type, bruk den
            if column_name in explicit_schema:
                bq_type = explicit_schema[column_name]
            # Ellers, bruk vår generelle mapping
            else:
                # Konverter Pandas dtype-objekt til en streng for oppslag
                dtype_str = str(dtype)
                bq_type = type_mapping.get(dtype_str, 'STRING')  # Default til STRING hvis ukjent

            schema.append(bigquery.SchemaField(column_name, bq_type))

        if if_exists in ['append','replace']:

            if if_exists == 'append':
                if not table_exists:
                    self._logger.warning(f"Table {table_id} does not exist. Creating a new table.")
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND" if table_exists else "WRITE_TRUNCATE",
                    schema=schema,  # Bruker eksplisitt schema
                    # autodetect=True,
                )
            if if_exists == 'replace':
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_TRUNCATE",
                    schema=schema,  # Bruker eksplisitt schema
                    #autodetect=True,
                )
            try:

                if to_str:
                    df = df.astype(str)
                job = self._bq_client.load_table_from_dataframe(
                    df, table_id, job_config=job_config
                )
                job.result()
                self._logger.info(f"{len(df)} rader lagret i {table_id}")
            except Exception as e:
                self._logger.error(
                    f"Error saving to BigQuery: {type(e).__name__}: {e} \n for dataframe {df.head()} with columns {df.columns}")
                self._logger.error(traceback.format_exc())
        elif if_exists == 'merge':
            
            if not merge_on or not isinstance(merge_on,list):
                raise ValueError("merge_on parameter must be provided when if_exists is 'merge' and must be a list of column names.")
            staging_table_id = f"{table_id}_staging_{uuid.uuid4().hex}"
            self._logger.info(f"Starting MERGE. Uploading data to staging table: {staging_table_id}")


            try:

                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_TRUNCATE",
                    schema = schema,
                    # autodetect=True,
                )

                if to_str:
                    df = df.astype(str)

                job = self._bq_client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
                job.result()  # Wait for the job to complete
                self._logger.info(f"Staging table {staging_table_id} created with {len(df)} rows.")

                # Dynamisk bygging av `ON`-betingelsen basert på merge_keys
                on_condition = ' AND '.join([f'T.`{key}` = S.`{key}`' for key in merge_on])

                # Dynamisk bygging av `UPDATE SET`-delen (oppdater alle kolonner unntatt nøklene)
                update_cols = [col for col in df.columns if col not in merge_on]
                update_set = ', '.join([f'T.`{col}` = S.`{col}`' for col in update_cols])

                # Dynamisk bygging av `INSERT`-delen
                insert_cols = ', '.join([f'`{col}`' for col in df.columns])
                insert_values = ', '.join([f'S.`{col}`' for col in df.columns])

                merge_query = f"""
                                MERGE `{table_id}` AS T
                                USING `{staging_table_id}` AS S
                                ON {on_condition}
                                WHEN MATCHED THEN
                                    UPDATE SET {update_set}
                                WHEN NOT MATCHED THEN
                                    INSERT ({insert_cols})
                                    VALUES ({insert_values})
                                """
                # self._logger.debug(f'Merge query: {merge_query[:1000]}... (truncated)')

                self._logger.info("Executing MERGE statement...")
                self.exe_query(merge_query)
                self._logger.info(f"MERGE operation on {table_id} complete.")

            finally:
                # --- STEG 3: Slett den midlertidige tabellen ---
                self._logger.info(f"Deleting staging table: {staging_table_id}")
                self._bq_client.delete_table(staging_table_id, not_found_ok=True)
        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}")
    def read_bq(self, query,read_type: Literal["bigframes","bq_client","pandas_gbq"] = 'bigframes'):
        '''
        Leser en BigQuery-spørring og returnerer en DataFrame.
        :param query:
        :param read_type: choose between 'bigframes', 'bq_client' and 'pandas_gbq'
        :return:
        '''
        if read_type == 'bigframes':
            df = bf.read_gbq(query).to_pandas()
        elif read_type == 'bq_client':
            df = self._bq_client.query(query).to_arrow().to_pandas()
        elif read_type == 'pandas_gbq':
            df = pbq.read_gbq(query,credentials=self._credentials)
        else:
            raise ValueError(f"Invalid read_type: {read_type}. Choose between 'bigframes', 'bq_client' and 'pandas_gbq'")
        df.replace(['nan','None','','null','NA','np.nan','<NA>','NaN','NAType','np.nan'],np.nan,inplace=True)
        self._logger.info(f"{len(df)} rader lest fra BigQuery")
        return df
    def exe_query(self, query):
        '''
        Execute a BigQuery query
        :param query:
        :return:
        '''
        job = self._bq_client.query(query)
        job.result()
        self._logger.info(f"Query executed: {query[:100]}... (truncated)")

class Logger:
    def __init__(self,log_name,root_path = None,write_type = 'a',CREDENTIALS_PATH = None):
        '''

        :param log_name:
        :param write_type: can take inn "w" or "a" for write or append
        '''
        self._logName = log_name
        if root_path:
            self._root = root_path
        elif os.getenv('DOCKER'):
            self._root = Path('/app') # Fortsatt relevant hvis du har Docker-spesifikk logikk
        else:
            self._root = Path.cwd() # Fallback til nåværende arbeidsmappe

        self._path = self._create_log_folder() / f'{self._logName}.log'
        self._write_type = write_type
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.DEBUG)
        self._path = self._create_log_folder() / f'{self._logName}.log'
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        self._create_handlers()
        self._cloud_log_name = 'sibr-market-workbench'
        self.client = None
        self.cloud_transport = None
        try:
            if not CREDENTIALS_PATH:
                credentials = google.auth.default()
            else:
                credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
            self.client = cloud_logging_client.Client(credentials=credentials)
            # self.client.setup_logging(log_level=logging.INFO, log_name=self._cloud_log_name)
            cloud_handler = CloudLoggingHandler(self.client, name=self._cloud_log_name)
            cloud_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(cloud_handler)
            self._logger.info(f'Google Cloud Logging initialized with project: {credentials.project_id}')
            self._logger.info(f'All loggs successfully initiated')
        except DefaultCredentialsError as e:
            self.client = None
            self._logger.error(f'Authentication error initializing Google Cloud Logging: {e}', exc_info=True)
            self._logger.warning('Cloud Logging will not be available due to authentication issues.')
        except GoogleAPICallError as e:
            self.client = None
            self._logger.error(f'Google API Call Error during Cloud Logging initialization: {e}', exc_info=True)
            self._logger.warning(
                'Cloud Logging might not be available due to an API error. Check permissions or network.')
        except Exception as e:
            self.client = None
            self._logger.error(f'An unexpected error occurred during Cloud Logging initialization: {e}', exc_info=True)
            self._logger.warning('Cloud Logging will not be available due to an unexpected error.')

    def shutdown(self):
        if self.cloud_transport:
            self.cloud_transport.flush()

    def _create_handlers(self):
        file_handler = logging.FileHandler(self._path, mode=self._write_type)
        console_handler = logging.StreamHandler()

        file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
        console_handler.setLevel(logging.DEBUG)  # Log only warnings and above to the console

        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def _create_log_folder(self):
        if self._root:
            root = self._find_root_folder()
            path = root / 'logfiles'
        else:
            path = Path.cwd() / 'logfiles'
        if not path.exists():
            path.mkdir()
        return path

    def _find_root_folder(self):
        if os.getenv('DOCKER'):
            return Path('/app')
        current_path = Path.cwd()
        path = current_path
        while True:
            for file in path.iterdir():
                if '.venv' in file.name:
                    return path
            if path == path.parent:
                break
            path = path.parent
        return current_path

    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)

    def set_level(self,level):
        if level not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']:
            raise ValueError(f'Invalid log level: {level}. Choose between DEBUG, INFO, WARNING, ERROR, CRITICAL')
        self._logger.setLevel(level)

class ConfigReader:
    def __init__(self):
        self._root = self._find_root_folder()
        self._configs = {}
        self._find_all_config()

    def _read_config(self, config_path):
        with open(config_path,'rb') as fc:
            cfg = tomllib.load(fc)
        return cfg

    def _find_root_folder(self):
        current_path = Path.cwd()
        path = current_path
        while True:
            for file in path.iterdir():
                if '.venv' in file.name:
                    return path
            if path == path.parent:
                break
            path = path.parent
        return current_path

    def _find_all_config(self):
        for file in self._root.rglob('*.toml'):
            self._configs[file.name] = file
        if not self._configs:
            raise FileNotFoundError('No config files found')

    def list_configs(self):
        return list(self._configs.keys())

    def get_config(self,config_name='standard.toml'):
        return self._read_config(self._configs[config_name])

class CStorage:
    '''
    A class for uploading files to Google Cloud Storage.
    :param bucket_name: The name of the Google Cloud Storage bucket.
    :param logger: An instance of Logger for logging.
    :param CREDENTIALS_PATH: Optional path to a service account key file. If not provided, it will use the default credentials.
    '''
    def __init__(self, bucket_name,logger:Logger, CREDENTIALS_PATH=None):
        self._logger = logger
        self._bucket_name = bucket_name
        if not CREDENTIALS_PATH:
            self._credentials = google.auth.default()
        else:
            self._credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        try:
            self._client = storage.Client(credentials=self._credentials, project=self._credentials.project_id)
            self._logger.info(f"Google Cloud Storage client initialized with bucket: {self._bucket_name}")
        except Exception as e:
            self._logger.error(f"Error initializing Google Cloud Storage client: {e}")
            raise ImportError(f"Error initializing Google Cloud Storage client: {e}")

    def upload(self, local_file_path, destination_blob_name = None):
        '''
        Uploads a file to Google Cloud Storage.
        :param local_file_path: Inlcudes the full path to the file with file-extension.
        :param destination_blob_name: Includes the full path to the file with file-extension.
        :return:
        '''
        if destination_blob_name is None:
            destination_blob_name = local_file_path.split('/')[-1]

        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            self._logger.info(f"File {local_file_path} uploaded to {destination_blob_name} in bucket {self._bucket_name}.")
        except Exception as e:
            self._logger.error(f"Failed to upload file {local_file_path} to bucket {self._bucket_name}: {e}")
            raise e

    def download(self, source_blob_name, destination_file_path=None,read_in_file = False ) :
        '''
        Downloads a file from Google Cloud Storage. Reads in file if file extension is csv or pkl.
        :param source_blob_name:
        :param destination_file_path:
        :param read_in_file:
        :return:
        '''
        if not read_in_file and not destination_file_path:
            raise ValueError("Either destination_file_path must be provided or read_in_file must be True to read the file content directly.")
        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(source_blob_name)
            name = os.path.basename(blob.name)
            temp_filepath = f'/tmp/{name}'
            if destination_file_path:
                blob.download_to_filename(destination_file_path)
                self._logger.info(f"Blob {source_blob_name} downloaded to {destination_file_path}.")
            if read_in_file:
                valid_ext = ['pkl','csv']
                if "." in name:
                    ext = name.split('.')[-1]
                    if ext not in valid_ext:
                        raise ValueError(f"Invalid file extension: {ext}. Valid extensions are: {valid_ext}")
                    try:
                        blob.download_to_filename(temp_filepath)
                        if ext == 'pkl':
                            output = joblib.load(temp_filepath)
                        elif ext == 'csv':
                            output = pd.read_csv(temp_filepath)
                        else:
                            output = None

                        self._logger.info(f'Read in {name}')
                        return output

                    except Exception as e:
                        self._logger.error(f"Failed to read file {name}: {e}")
                        raise e
                    finally:
                        if os.path.exists(temp_filepath):
                            os.remove(temp_filepath)
                else:
                    raise ValueError(f'File {name} does not have a valid extension. Valid extensions are: {valid_ext}')

        except Exception as e:
            self._logger.error(f"Failed to download blob {source_blob_name} from bucket {self._bucket_name}: {e}")
            raise e