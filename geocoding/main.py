import os
import asyncio
from datetime import datetime
import argparse
from dotenv import load_dotenv
from pathlib import Path
from src.GeoCodeAPI import geoApi
from sibr_module import Logger

load_dotenv()

cred_filename = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FILENAME")
if cred_filename:
    print(f'RUNNING LOCAL. ADAPTING LOADING PROCESS')
    project_root = Path(__file__).parent
    os.chdir(project_root)
    dotenv_path = project_root.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root.parent / cred_filename)

# map_geocoders = {"nominatim" : nominatimAPI,
#                 "geonorge" : geonorgeAPI}

map_conc_requests = {"nominatim" : 5,
                     "geonorge" : 30}

parser = argparse.ArgumentParser(f'Ceocoding script by SIBR')
group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--use-proxy', action='store_true', help='Use proxy for geocoding requests (default: True)')
group.add_argument('--address', type=str, help='Geocode address')
parser.add_argument('--no-save', action='store_true', help='Disable saving results')
parser.add_argument('--limit', type=int, default=None, help='Limit number of rows fetched from SQL (default: None)')
parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (default: INFO)')
parser.add_argument('--cloud-logging', action='store_true', default=False, help='Enable cloud logging (default: False)')
#parser.add_argument('--api', type=str, default='nominatim', help='Geocoding API to use (default: nominatim)')
parser.add_argument("--geocoder", choices=["geonorge", "nominatim"], default="geonorge")

if __name__ == "__main__":
    async def main():

        args = parser.parse_args()
        logger = Logger(log_name='geocoding',enable_cloud_logging=args.cloud_logging)
        starttime = datetime.now()

        geo = geoApi(logger=logger)

        if args.address:
            if args.geocoder == "geonorge":
                if not isinstance(args.address, list):
                    addresses = list(args.address)
                else:
                    addresses = args.address
                result = await geo.get_geonorge(addresses)
                res = geo.transform_single_geonorge(result).to_dict(orient='records')[0]
                print(f'Coordinates for address {args.address}: {res.get("lat")},{res.get("lng")}')
        else:
            logger.info(f'====== STARTING GEONORGE GEOCODING ======')
            sql = '''
                    WITH CombinedItems AS (
                        SELECT h.item_id, h.address
                        FROM `sibr-market.clean.homes` h
                        UNION ALL 
                        SELECT r.item_id, r.address
                        FROM `sibr-market.clean.rentals` r
                        )
                        SELECT ci.item_id, ci.address
                        FROM CombinedItems ci
                        WHERE NOT EXISTS (
                        SELECT 1
                        FROM staging.coordinates c
                        WHERE c.item_id = ci.item_id)
                        '''
            if args.limit:
                sql += f'\nLIMIT {args.limit}'

            df = geo.bq.read_bq(sql)
            inputs = df.set_index("item_id")["address"].to_dict()
            try:
                await geo.get_items_with_ids(inputs,
                                    fetcher=geo.get_geonorge,
                                  transformer=geo.transformer_geonorge,
                                  saver = geo.save_func,
                                    save_interval=5000,
                                    concurrent_requests=30,)
            finally:
                await geo.close()
            logger.info(f'====== STARTING NOMINATIM GEOCODING ======\n \tfetching those items that Geonorge missed!')
            sql = '''
                        SELECT 
                    item_id, 
                    address
                FROM (
                    SELECT item_id, address FROM `sibr-market.clean.homes`
                    UNION ALL 
                    SELECT item_id, address FROM `sibr-market.clean.rentals`
                ) AS CombinedItems
                WHERE item_id IN (
                    SELECT item_id 
                    FROM `sibr-market.staging.coordinates`
                    WHERE status = "NO_RESULTS" AND geocoder = "geonorge"
                );
            '''

            if args.limit:
                sql += f'\nLIMIT {args.limit}'
            df = geo.bq.read_bq(sql)
            inputs = df.set_index("item_id")["address"].to_dict()
            try:
                await geo.get_items_with_ids(inputs,
                                    fetcher=geo.get_nomi,
                                  transformer=geo.transformer_nomi,
                                  saver = geo.save_func,
                                    save_interval=5000,
                                    concurrent_requests=5,)
            finally:
                logger.info(f'Geocoding process completed. Time used: {datetime.now() - starttime}.')
                await geo.close()



    asyncio.run(main())