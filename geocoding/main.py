import os
import asyncio
from datetime import datetime
import argparse
from dotenv import load_dotenv
load_dotenv()
#print(f'GOOGLE {os.getenv("GOOGLE_APPLICATION_CREDENTIALS")}')
from src.GeoCodeAPI import geonorgeAPI,nominatimAPI
from sibr_module import Logger


map_geocoders = {"nominatim" : nominatimAPI,
                "geonorge" : geonorgeAPI}

map_conc_requests = {"nominatim" : 5,
                     "geonorge" : 30}

parser = argparse.ArgumentParser(f'Ceocoding script by SIBR')
group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--api', type=str, default='nominatim', help='Geocoding API to use (default: nominatim)')
parser.add_argument('--use-proxy', action='store_true', help='Use proxy for geocoding requests (default: True)')
group.add_argument('--address', type=str, help='Geocode address')
parser.add_argument('--no-save', action='store_true', help='Disable saving results')
parser.add_argument('--limit', type=int, default=None, help='Limit number of rows fetched from SQL (default: None)')
parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (default: INFO)')
parser.add_argument('--cloud-logging', action='store_true', default=False, help='Enable cloud logging (default: False)')
parser.add_argument("--geocoder", choices=map_geocoders.keys(), default="geonorge")

async def main():

    args = parser.parse_args()
    logger = Logger(log_name='geocoding',enable_cloud_logging=args.cloud_logging)
    starttime = datetime.now()

    geocoder = map_geocoders[args.geocoder](logger = logger)

    if args.address:
        result = await geocoder.get_item(args.address)
        res = geocoder._transform_output(result).to_dict(orient='records')[0]
        print(f'Coordinates for address {args.address}: {res.get("lat")},{res.get("lng")}')
    else:
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
        geonorge = geonorgeAPI(logger = logger)
        df = geonorge.bq.read_bq(sql)
        inputs = df.set_index("item_id")["address"].to_dict()
        try:
            save = not args.no_save
            logger.info(f'====== STARTING GEONORGE GEOCODING ======')
            await geonorge.get_items_with_ids(inputs,
                                                save=save,
                                                save_interval=5000,
                                                concurrent_requests=30,)
        finally:
            await geonorge.close()
        nomi = nominatimAPI(logger = logger)
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
        df = nomi.bq.read_bq(sql)
        inputs = df.set_index("item_id")["address"].to_dict()
        try:
            save = not args.no_save
            logger.info(f'====== STARTING NOMINATIM GEOCODING ======\n \tfetching those items that Geonorge missed!')
            await nomi.get_items_with_ids(inputs,
                                            save=save,
                                            save_interval=500,
                                            concurrent_requests=5)
        finally:
            nomi.logger.info(f'Geocoding process completed. Time used: {datetime.now() - starttime}.')
            await nomi.close()



asyncio.run(main())