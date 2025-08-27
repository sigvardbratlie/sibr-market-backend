from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import MCItem
from datetime import datetime
import re
import json
import base64


class McSpider(FinnBaseSpider):
    name = "mc"
    _table_name = "mcs"
    start_urls = [
                'https://www.finn.no/mc/all/search.html?make=1481&make=1484&make=1486',
                 'https://www.finn.no/mc/all/search.html?make=1487&make=7357&make=1489',
                 'https://www.finn.no/mc/all/search.html?make=7771&make=1497&make=1500',
                 'https://www.finn.no/mc/all/search.html?make=7901&make=1502&make=8313&make=2998',
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

    def get_description(self, response):
        out = [text.strip()
               for text in response.css('div[data-testid="expandable-section"] p::text').getall()
               if text.strip()]
        if out:
            return ' '.join(out)
        out_2 = [text.strip() for text in response.css('h3.t3.mb-0:contains("Beskrivelse") + '
                                                       'div[data-testid="expandable-section"] .whitespace-pre-wrap::text').getall()
                 if text.strip()]
        if out_2:
            return ' '.join(out_2)

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
        '''
        @url https://www.finn.no/mobility/search/car?location=20061&make=0.772
        @returns items 40 60
        @returns requests 1 60
        @scrapes title year km total_price url
        '''
        if 'playwright_page' in response.meta:
            page = response.meta["playwright_page"]
            await page.close()
        # self.logger.info(f'Processing {response.url}')
        item = MCItem()

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
        item['description'] = response.css('div.whitespace-pre-wrap::text').get()
        if item['description'] is None:
            item['description'] = self.get_description(response)
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
        item['total_price'] = self.get_total_price(response)

        item['features'] = response.css('h2.t3.mb-0:contains("Utstyr") +  ul ::text').getall()

        # ================= BASIC DETAILS =================

        item['year'] = response.css("div label.s-text-subtle:contains('Modellår') + p.m-0.font-bold::text").get()
        item['mileage'] = response.css("div label.s-text-subtle:contains('Kilometer') + p.m-0.font-bold::text").get()
        item['engine_volume']  = response.css("div label.s-text-subtle:contains('Slagvolum') + p.m-0.font-bold::text").get()
        item['power'] = response.css("div label.s-text-subtle:contains('Effekt') + p.m-0.font-bold::text").get()

        # ================= MC DETAILS =================

        mc_details = {
            "type": "Type", #ok
            "brand": "Merke", #ok
            "model": "Modell", #ok
            "color": "Farge", #ok
            "fuel": "Drivstoff", #ok
            "battery": "Batterikapasitet", #ok
            "range": "Rekkevidde (WLTP)",
            "vin": "Chassis nr. (VIN)",
            "reg_num": "Registreringsnummer",
            "condition": "Tilstand",
            "weight": "Vekt", #ok
            "manufacturer": "Fabrikant",
            "length": "Lengde",
            "height": "Høyde",
            "width": "Bredde",
        }
        for field, label in mc_details.items():
            if response.css(f'dt:contains("{label}") + dd::text').get():
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok
            else:
                item[field] = None

        # ================= OWNERSHIP DETAILS =================
        ownership_details = {
            "prev_owners": "Eiere", #ok
            "sales_type": "Slagsform", #ok
            "transfer_fee": "Omregistrering", #iok
            "price_excl_transfer": "Pris eksl. omreg.", #ok
            "first_registration": "1. gang registrert",
            "last_eu": "Sist EU-godkjent", #ok
            "next_eu": "Neste frist for EU-kontrol", #ok
            "warranty": "Garanti", #ok
            "warranty_length": "Garantiens varighet", #ok
            "warranty_until": "Garanti inntil", #ok
            "service_history": "Servicehistorikk", #ok
            "service_book": "Servicehefte", #ok
            "condition_report": "Tilstandsrapport", #ok
        }
        for field, label in ownership_details.items():
            if response.css(f'dt:contains("{label}") + dd::text').get():
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok
            else:
                item[field] = None

        # ================= BASIC DETAILS, back-up =================
        basic_details = {
            "year": "Modellår",
            "mileage": "Kilometerstand",
            "engine_volume": "Slagvolum",
            "power": "Effekt"
        }
        for field, label in basic_details.items():
            if not label in item or item[label] is None:
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok

        # ================= ADDITIONAL DETAILS =================

        item['dealer_rating'] = None
        item['dealer_n_ratings'] = None

        yield item
