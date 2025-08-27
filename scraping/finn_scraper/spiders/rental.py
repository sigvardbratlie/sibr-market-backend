from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import RentalItem
from datetime import datetime
import json


class RentalSpider(FinnBaseSpider):
    name = "rental"
    _table_name = "rentals"
    start_urls = [
                'https://www.finn.no/realestate/lettings/search.html?location=1.20061.20507&location=1.20061.20508&location=1.20061.20511&location=1.20061.20509&location=1.20061.20510&location=1.20061.20512&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=1.20061.20528&location=1.20061.20519&location=1.20061.20515&location=1.20061.20524&location=1.20061.20529&location=1.20061.20527&location=1.20061.20523&location=1.20061.20522&location=1.20061.20518&location=1.20061.20520&location=1.20061.20514&location=1.20061.20516&location=1.20061.20526&location=1.20061.20532&location=1.20061.20530&location=1.20061.20525&location=1.20061.20517&location=1.20061.20533&location=1.20061.20531&location=1.20061.20521&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.20003&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.22034&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.20008&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.20002&sort=PUBLISHED_DESC&stored-id=56508197',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.22042',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.20016',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.20012&location=0.20018&location=0.20015&location=0.20009&location=0.20019',
                 'https://www.finn.no/realestate/lettings/search.html?location=0.22046',
                  ]

    def __init__(self, *args, other_urls=None, **kwargs):
        super().__init__(*args, **kwargs)
        if other_urls:
            self.start_urls = other_urls
        else:
            self.start_urls = self.start_urls
    use_playwright_listing = False
    use_playwright_items = False

    custom_settings = {**FinnBaseSpider.custom_settings,
                       'LOG_LEVEL': 'INFO',
                       }

    @property
    def table_name(self):
        return self._table_name


    async def parse(self, response):

        if 'playwright_page' in response.meta:
            page = response.meta["playwright_page"]
            await page.close()

        company_data = {}
        script_tag_text = response.xpath('//script[@data-company-profile-data]/text()').get()
        try:
            company_data = json.loads(script_tag_text) if script_tag_text else {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e} in {response.url}")
        company_profile = company_data.get('companyProfile', {})

        item = RentalItem()

        # ================= COMMON FIELDS =================
        item['item_id'] = response.css('th.pr-8:contains("FINN-kode") + td.pl-8::text').get()
        item['title'] = response.css('section[aria-label="Tittel"] h1::text').get()
        item['description'] = response.css('div.description-area.whitespace-pre-wrap ::text').getall()
        item['url'] = response.url
        item['address'] = response.css('span[data-testid="object-address"]::text').get()
        item['last_updated'] = response.css('th.pr-8:contains("Sist endret") + td.pl-8::text').get()
        item['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        item['country'] = 'NO'
        item['dealer'] = company_profile.get('orgName')
        item['contact_person'] = company_profile.get('contacts', [{}])[0].get('name') if company_profile.get(
            'contacts') else company_profile.get('contact', {}).get('name')
        if company_profile.get('contacts', {}):
            if company_profile.get('contacts')[0].get('phone', {}):
                telefon_data_contacts = company_profile.get('contacts', [{}])[0].get('phone', [{}])[
                    0] if company_profile.get(
                    'contacts') else {}
                item['phone'] = telefon_data_contacts.get('phoneFormatted') or telefon_data_contacts.get('tel')
        if company_profile.get('contact', {}).get('phone', {}):
            telefon_data_contact = company_profile.get('contact', {}).get('phone', [{}])[0] if company_profile.get(
                'contact') else {}
            item['phone'] = telefon_data_contact.get('phoneFormatted')
        item['email'] = company_profile.get('contacts', [{}])[0].get('email') if company_profile.get(
            'contacts') else company_profile.get('contact', {}).get('email')
        item['web'] = company_profile.get('homepageUrl', None)

        # ================= PRICE =================
        item['property_type'] = response.css('div[data-testid="info-property-type"] dd::text').get()
        item['primary_area'] = response.css('div[data-testid="info-primary-area"] dd::text').get()
        item['usable_area'] = response.css('div[data-testid="info-usable-area"] dd::text').get()
        item['internal_area'] = response.css('div[data-testid="info-usable-i-area"] dd::text').get()
        item['external_area'] = response.css('div[data-testid="info-usable-e-area"] dd::text').get()
        item['gross_area'] = response.css('div[data-testid="info-gross-area"] dd::text').get()
        item['bedrooms'] = response.css('div[data-testid="info-bedrooms"] dd::text').get()
        item['floor'] = response.css('div[data-testid="info-floor"] dd::text').get()
        item['monthly_rent'] = response.css('div[data-testid="pricing-common-monthly-cost"] dd::text').get()
        item['deposit'] = response.css('div[data-testid="pricing-deposit"] dd::text').get()
        item['build_year'] = response.css('div[data-testid="info-construction-year"] dd::text').get()
        item['balcony'] = response.css('div[data-testid="info-open-area"] dd::text').get()
        item['energy_rating'] = response.css('div[data-testid="energy-label"] dd span::text').get()
        item['facilities'] = response.css('h2.h3[id="facilities-heading"] + div div::text').getall()
        item['includes'] = response.css('div[data-testid="pricing-common-includes"] dd::text').getall()
        item['plot_size'] = response.css('div[data-testid="info-plot-area"] dd::text').get()

        yield item