import logging

from scrapy.crawler import CrawlerProcess
from finn_scraper.spiders.car import CarSpider
from finn_scraper.spiders.home import HomeSpider
from finn_scraper.spiders.job import JobSpider
from finn_scraper.spiders.rental import RentalSpider
from finn_scraper.spiders.mc import McSpider
from finn_scraper.spiders.boat import BoatSpider
from finn_scraper.spiders.new_home import NewHomeSpider
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from datetime import datetime
import argparse

map_spiders = {'homes': HomeSpider,
       'cars': CarSpider,
        'jobs': JobSpider,
        'rentals': RentalSpider,
        'mcs': McSpider,
        'boats': BoatSpider,
        'new_homes': NewHomeSpider}

parser = argparse.ArgumentParser(description='Run Finn scraper spiders.')
parser.add_argument('--spiders',
                    nargs='*',
                    choices=map_spiders.keys(),
                    default=list(map_spiders.keys()),)
parser.add_argument('--urls-file',
                    type=str,
                    help='Path to a text file containing a list of ad URLs to scrape, one URL per line.')

parser.add_argument('--other_urls',
                    nargs='*',
                    help='Other URLs to scrape. Needs to be front page url from finn.no')
parser.add_argument('--log_level',default='INFO',help='Set the logging level (default: INFO)')
parser.add_argument('--cloud_logging',action='store_true',help='Enable cloud logging')

args = parser.parse_args()
start = datetime.now()

configure_logging(install_root_handler=False)
settings = get_project_settings()
settings.set('LOG_LEVEL', args.log_level.upper())
settings.set('CLOUD_LOGGING_ENABLED', args.cloud_logging)
for handler in logging.getLogger().handlers:
    handler.setLevel(getattr(logging,args.log_level.upper(), logging.INFO))

process = CrawlerProcess(settings)
logging.getLogger().info(f"\n \n -------- NEW SCRAPING JOB STARTED -------- ")


if args.urls_file:
    try:
        with open(args.urls_file, 'r') as f:
            # Leser alle linjer og fjerner eventuelle tomme linjer/whitespace
            urls_to_scrape = [line.strip() for line in f if line.strip()]
        logging.getLogger().info(f"Loaded {len(urls_to_scrape)} URLs from file: {args.urls_file}")
    except FileNotFoundError:
        logging.getLogger().error(f"Error: The file '{args.urls_file}' was not found.")
        exit(1) # Avslutter skriptet hvis filen ikke finnes

spiders_to_run = [map_spiders[spider] for spider in args.spiders if args.spiders and spider in map_spiders]
for spider in spiders_to_run:
    if args.other_urls:
        logging.getLogger().info(f"Adding {spider.name} with custom URLs: {args.other_urls}")
        process.crawl(spider, other_urls=args.other_urls)
    elif args.urls_file:
        if not urls_to_scrape:
            raise TypeError(f'No urls in file {args.urls_file}')
        logging.getLogger().info(f"Adding {spider.name} with URLs from file: {len(urls_to_scrape)} single urls to get")
        process.crawl(spider, other_urls=urls_to_scrape)
    else:
        logging.getLogger().info(f"Adding {spider.name} with default URLs")
        process.crawl(spider)

logging.getLogger().info(f"Starting scraping with spiders: {', '.join(spider.name for spider in spiders_to_run)}")
process.start()

end = datetime.now()
logging.getLogger().info(f"Scraping completed in {end - start} seconds")