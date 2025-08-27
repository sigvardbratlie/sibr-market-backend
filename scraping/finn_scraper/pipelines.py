from google.cloud import bigquery
import pandas as pd
from itemadapter import ItemAdapter

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface


class FinnScraperPipeline:
    def process_item(self, item, spider):
        return item


class BQPipeline:
    def __init__(self,bq_path,project,batch_size=100):
        self.bq_path = bq_path
        self.batch_size = batch_size
        self.project = project
        self.buffer = []
        self._nan = {}
        self._nan['total'] = 0

    @classmethod
    def from_crawler(cls,crawler):
        return cls(
            bq_path = crawler.settings.get('BQ_PATH'),
            batch_size = crawler.settings.getint('BQ_BATCH_SIZE',2500),
            project = crawler.settings.get('GOOGLE_CLOUD_PROJECT')
        )

    def open_spider(self,spider):
        spider.logger.info('========= STARTING BQ PIPELINE =========')
        self.client = bigquery.Client(project=self.project)
        self.dataset_id = f'raw'
        self.table_id = f"{self.dataset_id}.{spider.table_name}"
        spider.logger.info(f'PIPELINE INFO: \t project: {self.project}, table: {self.table_id}')


    def close_spider(self,spider):
        if self.buffer:
            for item in self.buffer:
                self.track_nan(item)
            self._save_to_bq(spider)
        self.client.close()
        self.send_nan_report(spider)
        spider.logger.info('========= BQ PIPELINE FINISHED =========')

    def track_nan(self, item):
        for key, value in item.items():
            if key not in self._nan:
                self._nan[key] = 0
            if value is None:
                self._nan[key] += 1

    def send_nan_report(self, spider):
        if not self._nan:
            return
        nan = dict(sorted(self._nan.items(), key=lambda x: x[1], reverse=True))
        report = "NaN Report:\n"
        # report += f"Total number of items: {self._nan['total']}\n"
        for key, count in nan.items():
            report += f"{key}: {count} NaN \t {(count/self._nan['total'])*100}% \n"
        spider.logger.info(report)
        self._nan.clear()

    def process_item(self, item, spider):
        self.buffer.append(ItemAdapter(item).asdict())
        self._nan['total'] += 1
        if len(self.buffer) >= self.batch_size:
            for item in self.buffer:
                self.track_nan(item)
            # spider.logger.info('Buffer size reached, saving to BigQuery...')
            self._save_to_bq(spider)
        return item

    def _save_to_bq(self,spider):
        if not self.buffer:
            return

        try:
            df = pd.DataFrame(self.buffer)
            for col in df.columns:
                if col != "scrape_date":
                    df[col] = df[col].astype(str)
            df["scrape_date"] = pd.to_datetime(df["scrape_date"],errors="coerce").dt.date
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema = [bigquery.SchemaField("scrape_date", "DATE")],
                autodetect=True,

            )
            job = self.client.load_table_from_dataframe(df,self.table_id,job_config=job_config)
            job.result()
            spider.logger.info(f"{len(df)} rows saved to {self.table_id}")
        except Exception as e:
            spider.logger.error(f"Error saving to BigQuery: {e}")

        self.buffer.clear()
