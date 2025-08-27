from pathlib import Path
import logging
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.oauth2 import service_account
from google.cloud.logging_v2.handlers import StructuredLogHandler
import os

# Scrapy settings for finn_scraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "finn_scraper"

SPIDER_MODULES = ["finn_scraper.spiders"]
NEWSPIDER_MODULE = "finn_scraper.spiders"


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "finn_scraper (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32
LOG_FILE = 'sibr-market-scraping.log'
LOG_LEVEL = 'INFO'
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
CLOUD_LOGGING_ENABLED = False

if CLOUD_LOGGING_ENABLED:
    logging.getLogger().info('Cloud logging is enabled. Initializing Google Cloud Logging...')
    try:
        client = cloud_logging.Client()
        client.setup_logging()
        gcloud_handler = CloudLoggingHandler(client, name='sibr-market-scraping')
        gcloud_handler.setLevel(log_level)
        # Use Scrapy's default log format
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        gcloud_handler.setFormatter(formatter)
        logging.getLogger().addHandler(gcloud_handler)
    except Exception as e:
        print(f"Failed to initialize Google Cloud Logging: {e}")
else:
    logging.getLogger().info('Cloud logging is disabled. Using local logging configuration.')

# Stream handler with same format
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
logging.getLogger().addHandler(stream_handler)


# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 16

HTTP_PROXY = 'http://sigvarbrat49411:jgytj0vcj8@154.21.32.105:21309'
HTTPS_PROXY = 'http://sigvarbrat49411:jgytj0vcj8@154.21.32.105:21309'

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "finn_scraper.middlewares.FinnScraperSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
   "finn_scraper.middlewares.FinnScraperDownloaderMiddleware": 543,
   'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   "finn_scraper.pipelines.BQPipeline": 300,
}

GOOGLE_CLOUD_PROJECT = "sibr-market"
BQ_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BQ_BATCH_SIZE = 5000
# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
