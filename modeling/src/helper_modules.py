from sibr_module import BigQuery
import pandas as pd
from typing import Literal


class CustomBigQuery(BigQuery):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def read_raw(self, dataset: str = None, replace=False, limit=None, ) -> pd.DataFrame:
        if not dataset:
            dataset = self.dataset
        if replace:
            sql = f'''
                        SELECT
                            t.*
                        FROM
                          (SELECT
                                *,
                                ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY scrape_date DESC) as rn
                              FROM sibr-market.raw.{dataset}) t
                        WHERE t.rn = 1
                '''
        else:
            sql = f'''
                        SELECT
                        t.*
                    FROM
                      (SELECT
                            *,
                            ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY scrape_date DESC) as rn
                          FROM sibr-market.raw.{dataset}) t
                    LEFT JOIN sibr-market.clean.{dataset} h ON t.item_id = h.item_id
                    WHERE t.rn = 1
                    AND t.scrape_date = (SELECT MAX(scrape_date) FROM sibr-market.raw.{dataset})
            '''
        if limit:
            sql += f' LIMIT {limit}'
        self._logger.info(f"Reading raw data from dataset: raw.{dataset}")
        return self.read_bq(sql)

    def read_geonorge(self):
        return self.read_bq('SELECT * FROM admin.geo_norge_old')

    def read_salestime(self, replace=False) -> pd.DataFrame:
        self._logger.warning("Method depreciated!")
        if replace:
            sql = f'''
            SELECT
              nd.item_id,
              PARSE_DATE('%Y-%m-%d', MIN(nd.scrape_date)) AS FIRST,
              PARSE_DATE('%Y-%m-%d', MAX(nd.scrape_date)) AS LAST,
              NULLIF( DATE_DIFF( PARSE_DATE('%Y-%m-%d', MAX(nd.scrape_date)), PARSE_DATE('%Y-%m-%d', MIN(nd.scrape_date)), DAY), 0) AS salgstid
            FROM
              `sibr-market.raw.{self.dataset}` AS nd
            WHERE
              nd.item_id NOT IN
              (
              SELECT item_id
                  FROM `sibr-market.raw.{self.dataset}`
                  WHERE
                    scrape_date = (SELECT MIN(scrape_date) FROM`sibr-market.raw.{self.dataset}`)
                    OR
                    scrape_date = (SELECT MAX(scrape_date) FROM`sibr-market.raw.{self.dataset}` )
                    )
            GROUP BY
              1;
              '''
        else:
            sql = f'''
            SELECT
              nd.item_id,
              PARSE_DATE('%Y-%m-%d', MIN(nd.scrape_date)) AS FIRST,
              PARSE_DATE('%Y-%m-%d', MAX(nd.scrape_date)) AS LAST,
              NULLIF( DATE_DIFF( PARSE_DATE('%Y-%m-%d', MAX(nd.scrape_date)), PARSE_DATE('%Y-%m-%d', MIN(nd.scrape_date)), DAY), 0) AS salgstid
            FROM
              `sibr-market.raw.{self.dataset}` AS nd
              LEFT JOIN sibr-market.clean.{self.dataset} c ON c.item_id = nd.item_id
            WHERE
              nd.item_id NOT IN
              (
              SELECT item_id
                  FROM `sibr-market.raw.{self.dataset}`
                  WHERE
                    scrape_date = (SELECT MIN(scrape_date) FROM`sibr-market.raw.{self.dataset}`)
                    OR
                    scrape_date = (SELECT MAX(scrape_date) FROM`sibr-market.raw.{self.dataset}` )
                    )
            AND c.item_id IS NULL
            GROUP BY
              1;
              '''
        self._logger.info(f"Reading salestime data from dataset: {self.dataset}")
        return self.read_bq(sql)

    def read_clean(self, dataset: str = None, replace=False, limit=None) -> pd.DataFrame:
        if not dataset:
            dataset = self.dataset

        if replace:
            sql = f'''
            SELECT c.* FROM sibr-market.clean.{dataset} c
            '''
        else:
            sql = f'''
            SELECT m.* FROM sibr-market.clean.{dataset} m
            WHERE DATE(m.scrape_date) = (SELECT MAX(PARSE_DATE('%Y-%m-%d',scrape_date)) FROM sibr-market.raw.{dataset})
            '''
        if limit:
            sql += f' LIMIT {limit}'
        self._logger.info(f"Reading clean data from dataset: {dataset}")
        return self.read_bq(sql)

    def read_oslo(self, query=None, dataset_name=None, task=None):
        if query is None and dataset_name is None:
            raise ValueError(f'Either query or dataset_name must be provided')
        if not query:
            if dataset_name is None and task is None:
                raise ValueError(f'Both task and datasetname must be provided')
            if task == 'train':
                query = f'''
                        SELECT p.*,
                               g.BYDELSNAVN AS district_name,
                               co.lat       AS lat,
                               co.lng       AS lng,
                        FROM `sibr-market.pre_processed.{dataset_name}` p
                                 JOIN clean.{self.dataset} c ON c.item_id = p.item_id
                                 LEFT JOIN admin.geo_oslo g ON c.postal_code = g.postnummer
                                 JOIN admin.coordinates co ON co.item_id = p.item_id
                        WHERE LOWER(c.municipality) = "oslo" \
                        '''
            elif task == 'predict':
                query = f'''
                        SELECT p.*,
                               g.BYDELSNAVN AS district_name,
                               co.lat       AS lat,
                               co.lng       AS lng,
                        FROM `sibr-market.pre_processed.{dataset_name}` p
                                 JOIN clean.{self.dataset} c ON c.item_id = p.item_id
                                 LEFT JOIN admin.geo_oslo g ON c.postal_code = g.postnummer
                                 JOIN admin.coordinates co ON co.item_id = p.item_id
                        WHERE LOWER(c.municipality) = "oslo" \
                        AND DATE(c.scrape_date) = (SELECT MAX(PARSE_DATE('%Y-%m-%d',scrape_date)) FROM sibr-market.raw.{self.dataset})
                        '''
            else:
                raise ValueError(f'Invalid task: {task}. Choose between "train" or "predict"')

        df = self.read_bq(query)
        if not 'district_name' in df.columns:
            raise ImportError(f'The imported dataframe must contain district names')
        df['district_name'] = df['district_name'].fillna('Unknown')
        df = pd.get_dummies(df, columns=['district_name'], drop_first=True)
        if 'pre_processed_date' in df.columns:
            df.drop(columns=['pre_processed_date'], errors='ignore', inplace=True)
        df.set_index('item_id', inplace=True)
        return df

    def read_preprocessed(self, table: str, municipality: str = None, limit: int = None,
                          last_scrape_date: bool = False, coordinates: bool = True, random_sample: int = None,
                          debug_query=False) -> pd.DataFrame:

        if random_sample and limit:
            raise ValueError("Du kan ikke bruke både 'random_sample' og 'limit' samtidig.")

        # 1. Grunnleggende SELECT- og FROM-deler
        select_clauses = ["a.*"]
        from_clause = f"FROM `sibr-market.pre_processed.{table}` a"
        join_clauses = []
        where_clauses = []

        # 2. Håndter 'coordinates'-parameteren
        if coordinates:
            select_clauses.extend(["c.lat", "c.lng"])
            join_clauses.append("JOIN admin.coordinates c ON c.item_id = a.item_id")
            where_clauses.append("c.lat != 0")

        # 3. Håndter 'municipality'-parameteren
        if municipality:
            municipality_lower = municipality.lower()
            # Denne joinen er uavhengig av om vi henter koordinater eller ikke
            join_clauses.append(f"JOIN clean.{self.dataset} cl ON cl.item_id = a.item_id")
            where_clauses.append(f"LOWER(cl.municipality) = '{municipality_lower}'")

        # 4. Håndter 'last_scrape_date'-parameteren
        if last_scrape_date:
            where_clauses.append(
                f"DATE(a.year, a.month, a.day) = (SELECT MAX(PARSE_DATE('%Y-%m-%d', scrape_date)) FROM sibr-market.raw.{self.dataset})")

        # --- Sett sammen den endelige spørringen ---
        sql = f"SELECT {', '.join(select_clauses)} {from_clause}"

        if join_clauses:
            sql += " " + " ".join(join_clauses)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        if random_sample:
            sql += f' ORDER BY RAND() < {random_sample} \n'
        if limit:
            sql += f' LIMIT {limit}'

        if debug_query:
            print(sql)
        df = self.read_bq(sql)
        if 'item_id' not in df.columns:
            raise ValueError(
                f"DataFrame inneholder ikke 'item_id'-kolonnen. Tilgjengelige kolonner: {df.columns.tolist()}")

        df.drop(columns=['pre_processed_date'], errors='ignore', inplace=True)
        return df.set_index('item_id')

    def read_cars(self, task: str = Literal["clean", "pre_processed", "train", "predict"], limit: int = None,
                  random_samples: float = None, last_scrape=False, replace: bool = False, debug_query=False,
                  unimportant_columns: list = None) -> dict:
        if task not in ["clean", "pre_processed", "train", "predict"]:
            raise ValueError(f"Invalid task: {task}. Choose between 'clean', 'pre_processed', 'train' or 'predict'.")

        dataset = 'cars'
        if task == "clean":
            df = self.read_raw(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'cars': df}

        elif task == "pre_processed":
            df = self.read_clean(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'cars': df}

        elif task == "train":
            df_el = self.read_preprocessed(table='cars_el',
                                           coordinates=False,
                                           limit=limit,
                                           random_sample=random_samples,
                                           debug_query=debug_query)
            df_fossil = self.read_preprocessed(table='cars_fossil',
                                               coordinates=False,
                                               limit=limit,
                                               random_sample=random_samples,
                                               debug_query=debug_query)
            if df_el.empty or df_fossil.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'cars_el': df_el, 'cars_fossil': df_fossil}

        elif task == "predict":
            last_scrape = True
            df_el = self.read_preprocessed(table='cars_el', last_scrape_date=last_scrape, coordinates=False)
            df_fossil = self.read_preprocessed(table='cars_fossil', last_scrape_date=last_scrape, coordinates=False)
            if df_el.empty or df_fossil.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'cars_el': df_el, 'cars_fossil': df_fossil}

    def read_homes(self, task: str = Literal["clean", "pre_processed", "train", "predict"], limit: int = None,
                   random_samples: float = None, last_scrape=False, replace: bool = False, debug_query=False,
                   unimportant_columns: list = None) -> dict:
        def read_query(dataset_name, limit: int = None, random_samples=None, last_scrape=False, debug_query=False):
            if random_samples and last_scrape:
                raise ValueError(
                    "Only one of the following two can be True at the same time: random_samples and last_scrape")
            query = f"""
                                 SELECT
                  a.*,
                  CASE
                      WHEN COALESCE(h.n_grunnkrets, 0) < 2 THEN h.refprice_sqm_postal
                      WHEN COALESCE(h.n_postal, 0) < 3 THEN h.refprice_sqm_municipality
                      ELSE h.refprice_sqm_grunnkrets
                  END AS ref_price_pr_i_sqm,
                  CASE
                      WHEN COALESCE(h.n_grunnkrets, 0) < 2 THEN h.salestime_postal
                      WHEN COALESCE(h.n_postal, 0) < 3 THEN h.salestime_municipality
                      ELSE h.salestime_grunnkrets
                  END AS salestime,
                  h.lat,
                  h.lng,
                  h.inntekt_Manedslonn,
                  h.inntekt_Overtid,
                  h.husholdninger_Husholdninger,
                  h.sysselsetting_Arbeidssted,
                  h.sysselsetting_Bosatt,
                  h.sysselsetting_Innpendlere,
                  h.sysselsetting_Utpendlere,
                  h.utdanning_Personer,
                FROM `sibr-market.pre_processed.{dataset_name}` a
                JOIN agent.homes h ON h.item_id = a.item_id
                            """
            if last_scrape:
                query += f"WHERE DATE(a.year, a.month, a.day) = (SELECT MAX(scrape_date) FROM sibr-market.raw.homes) \n"
            if random_samples:
                query += f"WHERE RAND() < {random_samples} \n"
            if limit:
                query += f"LIMIT {limit}"
            if debug_query:
                print(query)
            df = self.read_bq(query)
            return df

        def prep_data(df, unimportant_columns=None):
            if unimportant_columns:
                important_columns = df.columns.difference(unimportant_columns)
                df = df[important_columns].copy()
            if "item_id" in df.columns:
                df.set_index('item_id', inplace=True)
            # else:
            #     logger.warning(f"no item_id in dataframe {df.columns}")
            if 'internal_area' in df.columns:
                df['ref_price'] = df.apply(lambda row: row['ref_price_pr_i_sqm'] * row['internal_area'], axis=1)
            elif 'usable_area' in df.columns:
                df['ref_price'] = df.apply(lambda row: row['ref_price_pr_i_sqm'] * row['usable_area'], axis=1)
            else:
                self._logger.warning(f'Missing usable or internal area: {df.columns}')
            # df['fees'] = df['ref_price'] * 0.025
            df.drop(columns=['pre_processed_date'], errors='ignore', inplace=True)
            return df

        def read_data(dataset_name, limit: int = None, random_samples=None, unimportant_columns=None, last_scrape=False,
                      debug_query=False):

            df = read_query(dataset_name=dataset_name, limit=limit, random_samples=random_samples,
                            last_scrape=last_scrape, debug_query=debug_query)
            df = prep_data(df, unimportant_columns=unimportant_columns)
            return df

        def read_data_oslo(unimportant_columns: list, limit=None, random_samples=None, last_scrape=False,
                           debug_query=debug_query):

            if random_samples and last_scrape:
                raise ValueError(
                    "Only one of the following two can be True at the same time: random_samples and last_scrape")

            query_oslo = """
                         WITH OsloHomes AS (SELECT h.*,
                                                   go.BYDELSNAVN AS district_geo
                                            FROM `sibr-market.clean.homes` h
                                                     LEFT JOIN `sibr-market.admin.geo_oslo` go
                         ON go.postnummer = h.postal_code
                         WHERE LOWER (h.municipality) = 'oslo'
                             )
                         SELECT a.*,
                                CASE
                                    WHEN p.n IS NULL THEN COALESCE(m.price_pr_i_sqm, 0)
                                    WHEN p.n < 3 THEN COALESCE(d.price_pr_i_sqm, m.price_pr_i_sqm, 0)
                                    ELSE COALESCE(p.price_pr_i_sqm, d.price_pr_i_sqm, 0)
                                    END       AS ref_price_pr_i_sqm,
                                p.salgstid    AS ref_salgstid,
                                c.lat,
                                c.lng,
                                go.BYDELSNAVN AS district_name
                         FROM `sibr-market.pre_processed.homes_apartments` a
                                  JOIN OsloHomes h ON h.item_id = a.item_id
                                  JOIN admin.coordinates c ON c.item_id = a.item_id
                                  LEFT JOIN `sibr-market.api.homes_oslo_districts` d ON d.district_name = h.district_geo
                                  LEFT JOIN `sibr-market.api.homes_apartments_postal` p ON p.postal_code = h.postal_code
                                  LEFT JOIN `sibr-market.api.homes_apartments_municipality` m
                                            ON m.municipality = h.municipality
                                  LEFT JOIN `sibr-market.admin.geo_oslo` go
                         ON go.postnummer = h.postal_code
                         """
            if last_scrape:
                query_oslo += f"WHERE DATE(a.year, a.month, a.day) = (SELECT MAX(PARSE_DATE('%Y-%m-%d', scrape_date)) FROM sibr-market.raw.homes) \n "
            if random_samples:
                query_oslo += f"WHERE RAND() < {random_samples} \n"
            if limit:
                query_oslo += f"LIMIT {limit}"
            if debug_query:
                print(query_oslo)
            df_o = self.read_bq(query_oslo)
            df_o = prep_data(df_o, unimportant_columns=unimportant_columns)
            df_o = pd.get_dummies(data=df_o, columns=['district_name'])
            return df_o

        if task not in ["clean", "pre_processed", "train", "predict"]:
            raise ValueError(f"Invalid task: {task}. Choose between 'clean', 'pre_processed', 'train' or 'predict'.")

        dataset = 'homes'
        if task == "clean":
            df = self.read_raw(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'homes': df}

        elif task == "pre_processed":
            df = self.read_clean(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'homes': df}
        elif task == "train":
            df_a = read_data('homes_apartments',
                             limit=limit,
                             random_samples=random_samples,
                             last_scrape=last_scrape,
                             unimportant_columns=unimportant_columns,
                             debug_query=debug_query)
            df_h = read_data('homes_houses',
                             limit=limit,
                             random_samples=random_samples,
                             last_scrape=last_scrape,
                             unimportant_columns=unimportant_columns,
                             debug_query=debug_query)
            df_o = read_data_oslo(limit=limit,
                                  random_samples=random_samples,
                                  last_scrape=last_scrape,
                                  unimportant_columns=unimportant_columns,
                                  debug_query=debug_query)

            if df_a.empty or df_h.empty or df_o.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'homes_apartments': df_a, 'homes_houses': df_h, 'homes_oslo': df_o}
        elif task == "predict":
            last_scrape = True
            random_samples = None
            limit = None
            df_a = read_data('homes_apartments',
                             limit=limit,
                             random_samples=random_samples,
                             last_scrape=last_scrape,
                             unimportant_columns=unimportant_columns,
                             debug_query=debug_query)
            df_h = read_data('homes_houses',
                             limit=limit,
                             random_samples=random_samples,
                             last_scrape=last_scrape,
                             unimportant_columns=unimportant_columns,
                             debug_query=debug_query)
            df_o = read_data_oslo(limit=limit,
                                  random_samples=random_samples,
                                  last_scrape=last_scrape,
                                  unimportant_columns=unimportant_columns,
                                  debug_query=debug_query)
            query = """
                    WITH OsloHomesRentals AS (SELECT h.*, \
                                                     go.BYDELSNAVN AS district_geo \
                                              FROM `sibr-market.clean.homes` h \
                                                       LEFT JOIN `sibr-market.admin.geo_oslo` go \
                    ON go.postnummer = h.postal_code
                    WHERE LOWER (h.municipality) = 'oslo'
                        )
                    SELECT a.*, \
                           d.rent_pr_sqm     AS ref_rent_pr_sqm, \
                           d.rent_pr_bedroom AS ref_rent_pr_bedroom, \
                           c.lat, \
                           c.lng, \
                           go.BYDELSNAVN     AS district_name
                    FROM `sibr-market.pre_processed.homes_rentals` a
                             JOIN OsloHomesRentals h ON h.item_id = a.item_id
                             JOIN admin.coordinates c ON c.item_id = a.item_id
                             LEFT JOIN `sibr-market.api.rentals_oslo` d ON d.district_name = h.district_geo
                             LEFT JOIN `sibr-market.admin.geo_oslo` go \
                    ON go.postnummer = h.postal_code
                    WHERE DATE (a.year \
                        , a.month \
                        , a.day) = (SELECT MAX (PARSE_DATE('%Y-%m-%d' \
                        , scrape_date)) FROM sibr-market.raw.homes)
                    """
            df_r = self.read_bq(query=query)
            df_r = prep_data(df_r, unimportant_columns=unimportant_columns)
            df_r = pd.get_dummies(data=df_r, columns=['district_name'])
            if "primary_area" in df_r.columns:
                df_r["ref_rent"] = df_r.apply(lambda row: row['ref_rent_pr_sqm'] * row['primary_area'], axis=1)
            elif "bedrooms" in df_r.columns:
                df_r["ref_rent"] = df_r.apply(lambda row: row['ref_rent_pr_bedroom'] * row['bedrooms'], axis=1)
            else:
                self._logger("No bedroom or primary_area in dataframe: {df_r.columns}")
            if df_a.empty or df_h.empty or df_o.empty or df_r.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'homes_apartments': df_a, 'homes_houses': df_h, 'homes_oslo': df_o, 'homes_rentals_oslo': df_r}

    def read_rentals(self, task: str = Literal["clean", "pre_processed", "train", "predict"], limit: int = None,
                     random_samples: float = None, last_scrape=False, replace: bool = False, debug_query=False,
                     unimportant_columns: list = None) -> dict:
        def read_query(dataset_name, limit: int = None, random_samples=None, last_scrape=False, debug_query=False):
            if random_samples and last_scrape:
                raise ValueError(
                    "Only one of the following two can be True at the same time: random_samples and last_scrape")
            query = f"""
        SELECT
        a.*,
        CASE
            WHEN COALESCE(p.n, 0) < 3 THEN m.rent_pr_sqm
            ELSE p.rent_pr_sqm
        END AS ref_rent_pr_sqm,
        CASE
            WHEN COALESCE(p.n, 0) < 3 THEN m.rent_pr_bedroom
            ELSE p.rent_pr_bedroom
        END AS ref_rent_pr_bedroom,
        c.lat,
        c.lng,
        s.* EXCEPT (Kommune, Kommunenr, `År`)
        FROM `sibr-market.pre_processed.rentals` a
            JOIN clean.rentals h ON h.item_id = a.item_id
            JOIN admin.coordinates c ON c.item_id = a.item_id
            LEFT JOIN `sibr-market.api.{dataset_name}_postal` p ON p.postal_code = h.postal_code
            LEFT JOIN `sibr-market.api.{dataset_name}_municipality` m ON m.municipality = h.municipality
            LEFT JOIN `sibr-market.admin.SSB_municipality` s ON LOWER(s.Kommune) = LOWER(h.municipality)
        """
            if last_scrape:
                query += f"WHERE DATE(a.year, a.month, a.day) = (SELECT MAX(PARSE_DATE('%Y-%m-%d', scrape_date)) FROM sibr-market.raw.rentals) \n"
            if random_samples:
                query += f"WHERE RAND() < {random_samples} \n"
            if limit:
                query += f"LIMIT {limit}"
            if debug_query:
                print(query)
            df = self.read_bq(query)
            return df

        def prep_data(df, unimportant_columns=None, drop_hybel=False):
            if unimportant_columns:
                important_columns = df.columns.difference(unimportant_columns)
                df = df[important_columns].copy()
            if "item_id" in df.columns:
                df.set_index('item_id', inplace=True)
            # else:
            #     logger.warning(f"no item_id in dataframe {df.columns}")
            if 'primary_area' in df.columns:
                df['ref_rent'] = df.apply(lambda row: row['ref_rent_pr_sqm'] * row['primary_area'], axis=1)
            elif 'bedrooms' in df.columns:
                df['ref_rent'] = df.apply(lambda row: row['ref_rent_pr_sqm'] * row['bedrooms'], axis=1)
            # else:
            #     self._logger.warning(f'Missing both primary_area and bedrooms in dataframe: {df.columns}')
            if drop_hybel:
                df = df[df["property_type_hybel"] != True]
                df.drop(columns=["dealer_True"], errors="ignore", inplace=True)
            df.drop(columns=['pre_processed_date'], errors='ignore', inplace=True)
            return df

        def read_data(dataset_name, limit: int = None, random_samples=None, unimportant_columns=None, last_scrape=False,
                      drop_hybel=False, debug_query=debug_query):

            df = read_query(dataset_name=dataset_name, limit=limit, random_samples=random_samples,
                            last_scrape=last_scrape, debug_query=debug_query)
            df = prep_data(df, unimportant_columns=unimportant_columns, drop_hybel=drop_hybel)
            return df

        def read_data_oslo(query, unimportant_columns: list = [], limit=None, random_samples=None, last_scrape=False,
                           drop_hybel=False, debug_query=debug_query):

            if random_samples and last_scrape:
                raise ValueError(
                    "Only one of the following two can be True at the same time: random_samples and last_scrape")
            query_oslo = query

            if last_scrape:
                query_oslo += f"WHERE DATE(a.year, a.month, a.day) = (SELECT MAX(PARSE_DATE('%Y-%m-%d', scrape_date)) FROM sibr-market.raw.rentals) \n "
            if random_samples:
                query_oslo += f"WHERE RAND() < {random_samples} \n"
            if limit:
                query_oslo += f"LIMIT {limit}"
            if debug_query:
                print(query_oslo)
            df_o = self.read_bq(query_oslo)
            df_o = prep_data(df_o, unimportant_columns=unimportant_columns, drop_hybel=drop_hybel)
            df_o = pd.get_dummies(data=df_o, columns=['district_name'])
            return df_o

        query_co_living = """
                          WITH OsloRentals AS 
                                   (SELECT 
                                        h.*, 
                                        go.BYDELSNAVN AS district_geo 
                                    FROM `sibr-market.clean.rentals` h 
                                    LEFT JOIN `sibr-market.admin.geo_oslo` go
                                      ON go.postnummer = h.postal_code
                                      WHERE LOWER (h.municipality) = 'oslo'
                                          )
                          SELECT a.*, 
                                 d.monthly_rent AS ref_rent, 
                                 c.lat, 
                                 c.lng, 
                                 go.BYDELSNAVN  AS district_name
                          FROM `sibr-market.pre_processed.rentals_co-living` a
                               JOIN OsloRentals h ON h.item_id = a.item_id
                               JOIN admin.coordinates c ON c.item_id = a.item_id
                               LEFT JOIN `sibr-market.api.rentals_co-living_oslo` d
                                        ON d.district_name = h.district_geo
                               LEFT JOIN `sibr-market.admin.geo_oslo` go
                                    ON go.postnummer = h.postal_code
                          """

        query_oslo = f"""
        WITH OsloRentals AS (
        SELECT
            h.*,
            go.BYDELSNAVN AS district_geo  
        FROM `sibr-market.clean.rentals` h
        LEFT JOIN `sibr-market.admin.geo_oslo` go ON go.postnummer = h.postal_code
        WHERE LOWER(h.municipality) = 'oslo'
        )
        SELECT
        a.*,
        d.rent_pr_sqm AS ref_rent_pr_sqm,
        d.rent_pr_bedroom AS ref_rent_pr_bedroom,
        c.lat,
        c.lng,
        go.BYDELSNAVN AS district_name
        FROM `sibr-market.pre_processed.rentals` a
        JOIN OsloRentals h ON h.item_id = a.item_id
        JOIN admin.coordinates c ON c.item_id = a.item_id
        LEFT JOIN `sibr-market.api.rentals_oslo` d ON d.district_name = h.district_geo
        LEFT JOIN `sibr-market.admin.geo_oslo` go ON go.postnummer = h.postal_code
        """

        dataset = "rentals"
        if task not in ["clean", "pre_processed", "train", "predict"]:
            raise ValueError(f"Invalid task: {task}. Choose between 'clean', 'pre_processed', 'train' or 'predict'.")

        if task == "clean":
            df = self.read_raw(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'rentals': df}

        elif task == "pre_processed":
            df = self.read_clean(dataset=dataset, replace=replace, limit=limit)
            if df.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            return {'rentals': df}

        elif task == "train":
            df_a = read_data('rentals',
                             unimportant_columns=unimportant_columns,
                             limit=limit,
                             random_samples=random_samples,
                             last_scrape=last_scrape)
            df_co = read_data_oslo(query=query_co_living,
                                   unimportant_columns=unimportant_columns,
                                   limit=limit,
                                   random_samples=random_samples,
                                   last_scrape=last_scrape)
            df_o = read_data_oslo(query=query_oslo,
                                  unimportant_columns=unimportant_columns,
                                  limit=limit,
                                  random_samples=random_samples,
                                  last_scrape=last_scrape)
            if df_a.empty or df_o.empty or df_co.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            self._logger.info(
                f"Length's of dataframes: \t rentals {len(df_a)}, rental_oslo {len(df_o)}, rental_co-living {len(df_co)}")
            return {'rentals': df_a, 'rentals_oslo': df_o, 'rentals_co-living': df_co}

        elif task == "predict":
            last_scrape = True
            df_a = read_data('rentals',
                             unimportant_columns=unimportant_columns,
                             limit=limit, random_samples=random_samples,
                             last_scrape=last_scrape)
            df_co = read_data_oslo(query=query_co_living,
                                   unimportant_columns=unimportant_columns,
                                   limit=limit,
                                   random_samples=random_samples,
                                   last_scrape=last_scrape)
            df_o = read_data_oslo(query=query_oslo,
                                  unimportant_columns=unimportant_columns,
                                  limit=limit,
                                  random_samples=random_samples,
                                  last_scrape=last_scrape)
            if df_a.empty or df_o.empty or df_co.empty:
                raise ImportError(f"Dataframe {task} for {dataset} is empty")
            self._logger.info(
                f"Length's of dataframes: \t rentals {len(df_a)}, rental_oslo {len(df_o)}, rental_co-living {len(df_co)}")
            return {'rentals': df_a, 'rentals_oslo': df_o, 'rentals_co-living': df_co}
