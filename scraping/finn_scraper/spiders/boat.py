from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import MCItem, BoatItem
from datetime import datetime
import re
import json
import base64


class BoatSpider(FinnBaseSpider):
    name = "boat"
    _table_name = "boats"
    start_urls = [
                'https://www.finn.no/boat/forsale/search.html?location=22042&location=20009',
                 'https://www.finn.no/boat/forsale/search.html?location=20019&location=20018&location=20020&location=20016',
                 'https://www.finn.no/boat/forsale/search.html?location=22046&location=22034&location=20007',
                 'https://www.finn.no/boat/forsale/search.html?location=20061&location=20003',
                 'https://www.finn.no/boat/forsale/search.html?location=20002&location=20008',
                 'https://www.finn.no/boat/forsale/search.html?location=20012&location=20015',
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

    def get_total_price(self, response):
        """
        Extract total price from the page using multiple methods:
        1. Direct CSS selectors
        2. Fallback to encoded data-config attribute

        Args:
            response: Scrapy response object

        Returns:
            str: Total price if found, None otherwise
        """
        # Try CSS selectors first
        total_price = (
                response.css('span.t2::text').get()
                or response.css('h2[data-testid="price"]::text').get()
        )

        # If no price found, try data-config approach
        if not total_price:
            data_config = response.css('#tjm-ad-entry::attr(data-config)').get()
            if data_config:
                try:
                    config = json.loads(base64.b64decode(data_config).decode('utf-8'))
                    total_price = (
                            config.get('model', {}).get('totalPrice')
                            or config.get('model', {}).get('totalPriceAsText')
                    )
                except:
                    return None

        return total_price

    async def parse(self, response):
        if 'playwright_page' in response.meta:
            page = response.meta["playwright_page"]
            await page.close()
        # self.logger.info(f'Processing {response.url}')
        item = BoatItem()

        company_data = {}
        script_tag_text = response.xpath('//script[@data-company-profile-data]/text()').get()
        try:
            company_data = json.loads(script_tag_text) if script_tag_text else {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e} in {response.url}")
        company_profile = company_data.get('companyProfile', {})

        # ================= COMMON FIELDS =================
        item['item_id'] = response.css(
            'p.s-text-subtle.mb-0:contains("FINN-kode") + p.font-bold.mb-0::text').get()  # ok
        item['title'] = response.css('h1.t1::text').get()  # ok
        item['description'] = response.css('div.whitespace-pre-wrap ::text').getall()
        item['url'] = response.url  # ok
        for addr in (response.css('a[role="button"]::text').getall() + response.css('p.mb-0::text').getall()):
            if re.search(r'\d{4}.*', addr):
                item['address'] = addr
                break
        item['last_updated'] = response.css(
            'p.s-text-subtle.mb-0:contains("Sist endret") + p.font-bold.mb-0::text').get()  # ok
        item['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        item['country'] = 'NO'

        item['dealer'] = company_profile.get('orgName')
        item['contact_person'] = company_profile.get('contacts', {})[0].get('name') if company_profile.get(
            'contacts') else company_profile.get('contact', {}).get('name')
        if company_profile.get('contacts',{}):
            if company_profile.get('contacts')[0].get('phone', {}):
                telefon_data_contacts = company_profile.get('contacts', [{}])[0].get('phone', [{}])[0] if company_profile.get(
                    'contacts') else {}
                item['phone'] = telefon_data_contacts.get('phoneFormatted') or telefon_data_contacts.get('tel')
        if company_profile.get('contact', {}).get('phone', {}):
            telefon_data_contact = company_profile.get('contact', {}).get('phone', [{}])[0] if company_profile.get(
                'contact') else {}
            item['phone'] = telefon_data_contact.get('phoneFormatted')
        item['email'] = company_profile.get('contacts', [{}])[0].get('email') if company_profile.get(
            'contacts') else company_profile.get('contact', {}).get('email')
        item['web'] = company_profile.get('homepageUrl', None)
        item['dealer_rating'] = None
        item['dealer_n_ratings'] = None

        # ================= PRICE =================
        item['total_price'] = self.get_total_price(response)

        item['features'] = response.css('h2.t3.mb-0:contains("Utstyr") +  div ::text').getall()

        # ================= BASIC DETAILS =================
        item['year'] = response.css("div label.s-text-subtle:contains('Modellår') + p.m-0.font-bold::text").get()
        item['length'] = response.css("div label.s-text-subtle:contains('Lengde') + p.m-0.font-bold::text").get()
        item['engine_type'] = response.css(
            "div label.s-text-subtle:contains('Type motor') + p.m-0.font-bold::text").get()
        item['seats'] = response.css("div label.s-text-subtle:contains('Seter') + p.m-0.font-bold::text").get()

        # ================= BOAT DETAILS =================
        boat_details = {
            'condition': "Tilstand",  # ok
            "brand": "Merke",  # ok
            "model": "Modell",  # ok
            "type": "Type",  # ok
            "fuel": "Drivstoff",  # ok
            "engine_included": "Motor inkludert",  # ok
            "engine_size": "Motorstørrelse",  # ok
            "engine_manufacturer": "Motorfabrikant",  # ok
            "max_speed" : "Topphastighet", # ok
            "building_material": "Byggemateriale",  # ok,
            "depth": "Dybde",  # ok
            "width": "Bredde",  # ok
            "location" : "Båtens beliggenhet",  # ok
            "sleeping_places": "Soveplasser",  # ok,
            "color": "Farge",  # ok
            "reg_num": "Registreringsnummer",  # ok
        }
        for field, label in boat_details.items():
            if response.css(f'dt:contains("{label}") + dd::text').get():
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok
            else:
                item[field] = None

        # ======== back basic details ========
        basic_details = {
            'engine_type': "Type motor",  # ok
            'year' : "Modellår",  # ok
            'length': "Lengde i fot",  # ok
            'seats': "Sitteplasser",  # ok
        }
        for field, label in basic_details.items():
            if not label in item or item[label] is None:
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok



        yield item