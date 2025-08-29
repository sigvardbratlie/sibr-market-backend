import os
import logging
from http.client import responses
from sibr_module import BigQuery, Logger, SecretsManager
import aiohttp
import asyncio
import inspect
import pandas as pd
from urllib.parse import quote_plus
from typing import Literal
from src.settings import GOOGLE_CLOUD_PROJECT
import json
import abc
from dotenv import load_dotenv
#os.chdir("..")
#load_dotenv()

class RateLimitError(Exception):
    """Custom exception for rate limit errors."""

    def __init__(self, message="Rate limit exceeded. Please try again later."):
        self.message = message
        super().__init__(self.message)
class APIkeyError(Exception):
    """Custom exception for API key errors."""

    def __init__(self, message="API key is invalid or missing."):
        self.message = message
        super().__init__(self.message)

class GeoBase(metaclass=abc.ABCMeta):
    def __init__(self,logger_name = 'GeoBase',logger = None, proxies : dict = None):
        if logger is None:
            logger = Logger(log_name=logger_name, write_type='a', enable_cloud_logging=False)
        self.logger = logger
        self.api_key = None
        self.session = None
        self.use_proxy = True
        if proxies is None:
            proxies = {'http': 'http://sigvarbrat49411:jgytj0vcj8@154.21.32.105:21309',
            'https': 'http://sigvarbrat49411:jgytj0vcj8@154.21.32.105:21309'}
        self.proxies = proxies
        self.base_url = None
        self.ok_responses = 0
        self.fail_responses = 0


    def _ensure_fieldnames(self, df):
        new_cols = []
        for col in df.columns:
            new_cols.append(col.replace('.', '_', ))
        df.columns = new_cols

    def _mk_proxy(self, url: str) -> str:
        proxy_url = None
        if self.use_proxy:
            if url.startswith('https://'):
                proxy_url = self.proxies['https']
            else:
                proxy_url = self.proxies['http']
        return proxy_url

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _reset_session(self):
        if self.session:
            await self.session.close()
        self.session = None
        return await self._ensure_session()

    async def _fetch_single(self, url,headers = None,proxy_url = None):
        session = await self._ensure_session()
        try:
            async with session.get(url, params=headers, proxy=proxy_url) as response:
                if response.status == 429:
                    error_message = await response.text()
                    raise RateLimitError(
                        f'Rate limit exceeded. Error: {error_message}')
                if response.status == 403:
                    error_message = await response.text()
                    raise PermissionError(f'Permission denied. Error: {error_message}')
                if response.status == 401:
                    error_message = await response.text()
                    raise APIkeyError(f'Authorization error. Error: {error_message}')
                if response.status == 200:
                    response = await response.json()
                    return response
                else:
                    error_text = await response.text()
                    self.logger.error(
                        f'Error message - {inspect.currentframe().f_code.co_name}: {response.status}, {error_text}.')
                    return None
        except RateLimitError:
            raise
        except APIkeyError:
            raise
        except aiohttp.ClientError as e:
            self.logger.error(f"network failure - {e}. url {url}")
            return None
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout error.")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error in fetch_single - {e}")
            print(f'Reponse: {response}, response type : {type(response)}, url {url} ')
            return None

    @abc.abstractmethod
    def _transform_output(self,output):
        pass

    @abc.abstractmethod
    def get_item(self, item):
        pass

    @abc.abstractmethod
    def _save_func(self,results):
        pass

    async def _process_batch(self,tasks : list):

        for future in asyncio.as_completed(tasks):
            result = await future
            if isinstance(result,RateLimitError):
                self.logger.warning(f'Rate limit exceeded. Stopping code. Please try again later.')
                break
            yield result

    async def _process_tasks(self,tasks : list, save : bool, save_interval : int):

        all_results = []
        results_to_save = []

        processed_results = self._process_batch(tasks)
        count = 0
        try:
            async for result in processed_results:
                count += 1
                if result is None:
                    self.logger.warning(f'Results is None')
                    continue

                results_to_save.append(result)
                if count % 500 == 0:
                    self.logger.info(f'Processed {count} so far. Successful requests: {self.ok_responses} | failed requests {self.fail_responses}')

                if len(results_to_save) >= save_interval:
                    self.logger.info(f'Processed {count} so far. Save interval of {save_interval} reached.')
                    if save:
                        self.logger.info(f'Saving {len(results_to_save)} results')
                        self._save_func(results_to_save)
                    all_results.extend(results_to_save)
                    results_to_save.clear()

        except RateLimitError:
            self.logger.warning(f'Rate limit exceeded. Stopping code. Please try again later.')
        finally:
            if results_to_save:
                if save:
                    self._save_func(results_to_save)
                all_results.extend(results_to_save)
                results_to_save.clear()
        self.logger.info(f'Geocoding job finished. Successful requests: {self.ok_responses} | failed requests {self.fail_responses}')
        return all_results

    async def get_items_with_ids(self,
                               inputs : list | dict,
                               save: bool = False,
                               save_interval: int = 50000,
                               concurrent_requests : int = 5,
                               ) :

        if isinstance(inputs, list):
            inputs = {addr : addr for addr in inputs}
        elif isinstance(inputs, dict):
            pass
        else:
            raise TypeError(f'Invalid input type. Expected list or dict, but got {type(inputs)}')

        semaphore = asyncio.Semaphore(concurrent_requests)

        async def fetch_item_with_id(item_id, item):
            """
            Takes in both an item_id and address.
            Args:
                item_id (str):
                address (str):

            Returns (tuple): (item_id, result)

            """
            async with semaphore:
                try:
                    result  = await self.get_item(item)
                    return (item_id, result)
                except RateLimitError:
                    self.logger.warning(f'Rate limit exceeded. Stopping code. Please try again later.')
                    raise
                except Exception as e:
                    self.logger.error(f'Error fetching item {item} with item_id {item_id} - {e}')
                    return (item_id, None)

        tasks = [fetch_item_with_id(item_id=item_id, item=item) for item_id,item in inputs.items()]

        all_results = await self._process_tasks(tasks, save, save_interval)
        return all_results

    async def get_items(self,
                               inputs : list | dict,
                               save: bool = False,
                               save_interval: int = 50000,
                               concurrent_requests : int = 5,
                               ) :

        if not isinstance(inputs, list):
            raise TypeError(f'Invalid input type. Expected list, but got {type(inputs)}')

        semaphore = asyncio.Semaphore(concurrent_requests)

        async def fetch_item(item):
            async with semaphore:
                try:
                    result  = await self.get_item(item)
                    return result
                except RateLimitError:
                    self.logger.warning(f'Rate limit exceeded. Stopping code. Please try again later.')
                    raise
                except Exception as e:
                    self.logger.error(f'Error fetching item {item} with - {e}')
                    return None

        tasks = [fetch_item(item=item) for item_id,item in inputs]

        all_results = await self._process_tasks(tasks, save, save_interval)
        return all_results

    def _encode_address(self,address):
        if "/" in address:
            f = address.split("/")[0]
            l = address.split("/")[1].split(",")
            address = f"{f}, {''.join(l[1:])}"

        encoded_address = quote_plus(address)

        if not isinstance(encoded_address, str) or not encoded_address.strip():
            self.logger.warning(f"Skipping address: {address}. No valid output after encoding: {encoded_address}")
            return None

        return encoded_address



class nominatimAPI(GeoBase):
    def __init__(self, logger = None):
        super().__init__(logger_name='nominatim', logger = logger)
        self.base_url = "https://nominatim.openstreetmap.org/"
        self.bq = BigQuery(logger = self.logger)

    async def get_item(self,address):
        search_endpoint = "search"
        encoded_address = self._encode_address(address)

        if encoded_address is None:
            self.logger.warning(f'Address input is None')
            return None

        url = self.base_url + search_endpoint + f"?q={encoded_address}&format=jsonv2"
        headers = {'User-Agent': 'YourApp/1.0'}

        proxy_url = self._mk_proxy(url)
        response = await self._fetch_single(url,headers=headers,proxy_url=proxy_url)
        if response:
            self.ok_responses += 1

        if isinstance(response,dict):
            return response
        elif isinstance(response,list):
            if len(response)==1:
                return response[0]
            elif len(response) > 1 :
                self.logger.warning(f'Multiple results for address {address}')
                return response
            else:
                self.logger.warning(f'No results for address {address}')
                return None
        elif response is None:
            self.fail_responses += 1
            self.logger.warning(f'No results for address {address}')
            return None


    def _transform_output(self,response : tuple[str,dict] | dict) -> pd.DataFrame:
        required_cols = ['item_id', 'lat', 'lng', 'status', 'geocoder', 'adressetekst' ,'get_date']
        if isinstance(response,tuple):
            item_id = response[0]
            json_data = response[1]
        elif isinstance(response,dict):
            json_data = response
            item_id = None
        else:
            raise TypeError(f'Not valid response type. Expected tuple or dict, but got {type(response)}')

        if json_data is not None:
            data = pd.json_normalize(json_data)
            data.rename(columns={'lon': 'lng',
                                 'display_name': 'adressetekst',
                                 'name': 'adressenavn',
                                 "addresstype": "objtype"}, inplace=True)
            data['item_id'] = item_id
            data['status'] = "OK"
            data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
            data['lng'] = pd.to_numeric(data['lng'], errors='coerce')
            data['geocoder'] = "nominatim"
            data['get_date'] = pd.Timestamp.now()

            for col in required_cols:
                if col not in data.columns:
                    self.logger.warning(
                        f"Column {col} not found in response. Adding empty column. \t \tresponse: {response}")
                    data[col] = None
            return data[required_cols]

        elif json_data is None and item_id is not None:
            return pd.DataFrame({'item_id': [item_id],
                                 'status': 'NO_RESULTS',
                                 'geocoder': "nominatim",
                                 'get_date': pd.Timestamp.now()
                                 })


    def _save_func(self,results : list[tuple[str,dict]],table_name : str = None, dataset_name : str = None):

        if table_name is None:
            table_name = "coordinates"
        if dataset_name is None:
            dataset_name = "staging"

        dfs = [self._transform_output(result) for result in results if result is not None]
        df = pd.concat(dfs, ignore_index=True)
        self._ensure_fieldnames(df)
        self.bq.to_bq(df = df,
                      table_name = table_name,
                      dataset_name = dataset_name,
                      if_exists = 'append',
                      explicit_schema = {"get_date" : "DATETIME"})

class geonorgeAPI(GeoBase):
    def __init__(self, logger = None):
        super().__init__(logger_name='geonorge',logger = logger)
        self.base_url = "https://ws.geonorge.no/adresser/v1/"
        self.bq = BigQuery(project_id = GOOGLE_CLOUD_PROJECT, logger = logger)

    async def get_item(self,address):
        search_endpoint = "sok"
        encoded_address = self._encode_address(address)

        if encoded_address is None:
            self.logger.warning(f'Address input is None')
            return None

        url = f"{self.base_url}{search_endpoint}?sok={encoded_address}"
        headers = None
        proxy_url = self._mk_proxy(url)

        response = await self._fetch_single(url,headers=headers,proxy_url=proxy_url)


        if isinstance(response, dict):
            return response
        if isinstance(response, list):
            if len(response) == 1:
                return response[0]
            elif len(response) > 1:
                self.logger.warning(f'Multiple results for address {address}')
                return response
            else:
                self.logger.warning(f'No results for address {address}')
                return None
        elif response is None:
            self.logger.warning(f'No results for address {address}')
            return None

    async def _get_by_coordinate(self,lat : float ,lng : float ,radius : int =50):
        search_endpoint = "punktsok"
        url = f"{self.base_url}{search_endpoint}?lat={lat}&lon={lng}&radius={radius}"
        headers = None
        response = await self._fetch_single(url,headers=headers)
        if isinstance(response, dict):
            return response
        if isinstance(response, list):
            if len(response) == 1:
                return response[0]
            elif len(response) > 1:
                self.logger.warning(f'Multiple results for input {lat},{lng},{radius}')
                return response
            else:
                self.logger.warning(f'No results for input {lat},{lng},{radius}')
                return None

    def _transform_output(self,response : tuple[str,dict] | dict):
        if isinstance(response, tuple):
            item_id = response[0]
            json_data = response[1]
        elif isinstance(response, dict):
            json_data = response
            item_id = None
        else:
            raise TypeError(f'Not valid response type. Expected tuple or dict, but got {type(response)}')


        if json_data:
            addresses = json_data.get("adresser", [])
            metadata = json_data.get("metadata", {})

            all_data = []
            for addr in addresses:
                geo = addr.get("representasjonspunkt")
                merged_data = {**addr, **metadata, **geo}
                all_data.append(merged_data)
            df = pd.DataFrame(all_data)
            df['item_id'] = item_id
            df['get_date'] = pd.Timestamp.now()
            df["status"] = "OK"
            df["geocoder"] = "geonorge"
            if "representasjonspunkt" in df.columns:
                df["representasjonspunkt"] = df["representasjonspunkt"].astype(str)
            df.rename(columns={"lon" : "lng"},inplace=True)
            if not df.empty:
                self.ok_responses += 1
                return df
            elif df.empty:
                self.fail_responses += 1
                self.logger.warning(f'No results {metadata.get("sokeStreng")}') if metadata else self.logger.warning(f'No results for input {item_id}')
                if item_id is not None:
                    return pd.DataFrame({"item_id": [item_id],
                                 "get_date": pd.Timestamp.now(),
                                 "status": "NO_RESULTS",
                                 "geocoder": "geonorge"})
        else:
            self.fail_responses += 1
            if item_id is not None:
                return pd.DataFrame({"item_id": [item_id],
                                     "get_date": pd.Timestamp.now(),
                                     "status": "NO_RESULTS",
                                     "geocoder": "geonorge"})

    def _save_func(self,results : list[tuple[str,dict]],table_name : str = None, dataset_name : str = None):

        if table_name is None:
            table_name = "coordinates"
        if dataset_name is None:
            dataset_name = "staging"


        dfs = [self._transform_output(result) for result in results if result is not None]
        df = pd.concat(dfs, ignore_index=True)
        self._ensure_fieldnames(df)
        self.bq.to_bq(df = df,
                      table_name = table_name,
                      dataset_name = dataset_name,
                      if_exists = 'append',
                      explicit_schema={"get_date": "DATETIME",
                                       "undernummer" : "FLOAT"}
                      )


if __name__ == '__main__':

    async def main():
        geonorge  = geonorgeAPI()
        sql  = '''
        SELECT h.item_id, h.address
                    FROM `sibr-market.clean.homes` h
                    UNION ALL -- Changed to UNION ALL to keep duplicates if desired, or UNION for unique.
                              -- If you want to ensure item_id is unique across both tables, UNION is fine.
                    SELECT r.item_id, r.address
                    FROM `sibr-market.clean.rentals` r
               '''

        df = geonorge.bq.read_bq(sql,)
        inputs = df.set_index("item_id")["address"].to_dict()

        results = await geonorge.get_items_with_ids(inputs,
                                                    save=True,
                                                    save_interval=1000,
                                                    concurrent_requests=30)
        #print(results)
    asyncio.run(main())