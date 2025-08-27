from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import CarItem
from datetime import datetime
import re
import json
import base64

class CarSpider(FinnBaseSpider):
    name = "car"
    _table_name = "cars"
    start_urls = [
                    'https://www.finn.no/mobility/search/car?dealer_segment=1&dealer_segment=3&make=0.744&registration_class=1', #AUDI
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.744&registration_class=1', #AUDI
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.749&registration_class=1', #BMW
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&make=0.749&registration_class=1', #BWW
                  'https://www.finn.no/mobility/search/car?dealer_segment=3&make=0.749&registration_class=1', #BMW
                  'https://www.finn.no/mobility/search/car?make=0.757&make=0.772&registration_class=1', #hyandai og citroen,
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.767&registration_class=1', #FORD,
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&dealer_segment=3&make=0.767&registration_class=1', #FORD,
                  'https://www.finn.no/mobility/search/car?make=0.777&make=0.784&registration_class=1', #KIA og MAZDA,
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.785&registration_class=1', #MERCEDES
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&dealer_segment=3&make=0.785&registration_class=1', #MERCEDES,
                  'https://www.finn.no/mobility/search/car?make=0.787&make=0.795&registration_class=1', #MITSUBISHI og OPEL,
                  'https://www.finn.no/mobility/search/car?make=0.792&make=0.804&registration_class=1', #NISSAN og RENAULT,
                  'https://www.finn.no/mobility/search/car?make=0.796&make=0.811&registration_class=1', #PEUGEOT og SUZUKI,
                  'https://www.finn.no/mobility/search/car?make=0.801&registration_class=1' ,#Prosche,
                  'https://www.finn.no/mobility/search/car?make=0.808&registration_class=1',#SKODA,
                  'https://www.finn.no/mobility/search/car?make=0.8078&registration_class=1', #TESLA,
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&dealer_segment=3&make=0.813&registration_class=1', #TOYOTA
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&make=0.813&registration_class=1', #TOYOTA,
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.817&registration_class=1', #VOLKSWAGEN
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&make=0.817&registration_class=1' ,#VOLKSWAGEN,
                  'https://www.finn.no/mobility/search/car?dealer_segment=3&make=0.817&registration_class=1', #VOLKSWAGEN,
                  'https://www.finn.no/mobility/search/car?dealer_segment=1&make=0.818&registration_class=1', #VOLVO,
                  'https://www.finn.no/mobility/search/car?dealer_segment=2&make=0.818&registration_class=1', #VOLVO,
                  'https://www.finn.no/mobility/search/car?dealer_segment=3&make=0.818&registration_class=1', #VOLVO,
                  'https://www.finn.no/mobility/search/car?make=0.8101&make=0.753&make=0.8106&make=0.764&make=0.766&make=0.8107&make=0.775&make=0.776&make=0.2000649&registration_class=1', #OTHER BRANDS
                  'https://www.finn.no/mobility/search/car?make=0.6731&make=0.781&make=0.782&make=0.7191&make=0.7153&make=0.3001&make=0.8109&make=0.8102&make=0.7170&make=0.806&make=0.810&make=0.8112&make=0.8104&registration_class=1', #OTHER BRANDS,
                    'https://www.finn.no/mobility/search/car?make=0.8101&make=0.753&make=0.764&make=0.766&make=0.767&make=0.7179&make=0.7280&make=0.2000649&make=0.781&make=0.8096&make=0.787&registration_class=2', #OTHER BRANDS, VAREBIL,
                    'https://www.finn.no/mobility/search/car?make=0.792&make=0.795&make=0.796&make=0.804&make=0.7190&make=0.810&make=0.811&make=0.200820&make=0.2252&make=0.818&registration_class=2', #OTHER BRANDS, VAREBIL
                    'https://www.finn.no/mobility/search/car?make=0.8100&make=0.757&make=0.785&make=0.796&registration_class=2',
                    "https://www.finn.no/mobility/search/car?make=0.813&registration_class=2", # TOYOTA, VAREBIL
                    "https://www.finn.no/mobility/search/car?make=0.817&registration_class=2", # VOLKSWAGEN, VAREBIL
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
        item = CarItem()

        company_data = {}
        script_tag_text = response.xpath('//script[@data-company-profile-data]/text()').get()
        try:
            company_data = json.loads(script_tag_text) if script_tag_text else {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e} in {response.url}")
        company_profile = company_data.get('companyProfile', {})

        #================= COMMON FIELDS =================
        item['item_id'] = response.css('p.s-text-subtle.mb-0:contains("FINN-kode") + p.font-bold.mb-0::text').get()  # ok
        item['title'] = response.css('h1.t1::text').get()  # ok
        item['description'] = response.css('div.whitespace-pre-wrap ::text').getall()
        if item['description'] is None:
            item['description'] = self.get_description(response)
        item['url'] = response.url #ok
        for addr in (response.css('a[role="button"]::text').getall() + response.css('p.mb-0::text').getall()):
            if re.search(r'\d{4}.*', addr):
                item['address'] = addr
                break
        item['last_updated'] = response.css('p.s-text-subtle.mb-0:contains("Sist endret") + p.font-bold.mb-0::text').get()

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
        item['email'] = company_profile.get('contacts', [{}])[0].get('email') if company_profile.get('contacts') else company_profile.get('contact', {}).get('email')
        item['web'] = company_profile.get('homepageUrl', None)

        #================= PRICE =================
        item['total_price'] = self.get_total_price(response)

        # ================= BASIC DETAILS =================
        item['subtitle'] = response.css('h1.t1 + p::text').get()
        item['model_year'] = response.css("div span.s-text-subtle:contains('Modellår') + p.m-0.font-bold::text").get()
        item['mileage'] = response.css("div span.s-text-subtle:contains('Kilometerstand') + p.m-0.font-bold::text").get()
        item['gearbox'] = response.css("div span.s-text-subtle:contains('Girkasse') + p.m-0.font-bold::text").get()
        item['fuel'] = response.css("div span.s-text-subtle:contains('Drivstoff') + p.m-0.font-bold::text").get()

        item['features'] = response.css('h2.t3.mb-0:contains("Utstyr") +  ul ::text').getall()


        # ================= CAR DETAILS =================

        car_details = {
            "brand": "Merke",
            "model": "Modell",
            "body_type": "Karosseri",
            "color": "Farge",
            "wheel_drive": "Hjuldrift",
            "seats": "Seter",
            "doors": "Dører",
            "power": "Effekt",
            "battery": "Batterikapasitet",
            "range": "Rekkevidde (WLTP)",
            "vin": "Chassis nr. (VIN)",
            "reg_num": "Registreringsnummer",
            "category": "Avgiftsklasse",
            "trailer_weight": "Maksimal tilhengervekt",
            "weight": "Vekt",
            "manufacturer": "Fabrikant",
            "model_year": "Modellår",
            "fuel_cons": "Drivstofforbruk",
            "engine_volume": "Slagvolum",
            "cargo_space": "Størrelse på bagasjerom",
            "color_description": "Fargebeskrivelse",
            "color_interior": "Interiørfarge",
            "length": "Lengde",
            "height": "Høyde",
            "width": "Bredde",
            "gearbox_type": "Girkassebetegnelse"
        }
        for field, label in car_details.items():
            if response.css(f'dt:contains("{label}") + dd::text').get():
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok
            else:
                item[field] = None

        # ================= OWNERSHIP DETAILS =================
        ownership_details = {
            "state": "Bilen står i",
            "prev_owners": "Eiere",
            "sales_type": "Slagsform",
            "transfer_fee": "Omregistrering",
            "price_excl_transfer": "Pris eksl. omreg.",
            "first_registration": "1. gang registrert",
            "last_eu": "Sist EU-godkjent",
            "next_eu": "Neste frist for EU-kontroll",
            "warranty": "Garanti",
            "warranty_length": "Garantiens varighet",
            "warranty_until": "Garanti inntil",
            "co2": "CO₂-utslipp",
            "emission_class": "Utslippsklasse",
            "emission_sticker": "Miljømerke",
            "service_history": "Servicehistorikk",
            "service_book" : "Servicehefte",
            "condition_report": "Tilstandsrapport",
            "vat_deductible": "MVA-fradrag",
        }
        for field, label in ownership_details.items():
            if response.css(f'dt:contains("{label}") + dd::text').get():
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok
            else:
                item[field] = None

        # ================= BASIC DETAILS, back-up =================
        basic_details = {
            "model_year": "Modellår",
            "mileage": "Kilometerstand",
            "gearbox": "Girkasse",
            "fuel": "Drivstoff"
        }
        for field, label in basic_details.items():
            if not label in item or item[label] is None:
                item[field] = response.css(f'dt:contains("{label}") + dd::text').get()  # ok

        # ================= ADDITIONAL DETAILS =================
        item['known_issues'] = " ".join(response.css('p:contains("Kjente feil") + p::text').getall()) #ok
        item['major_repairs'] = " ".join(response.css('p:contains("større reparasjoner") + p::text').getall()) #ok
        item['engine_tuned'] = " ".join(response.css('p:contains("motoren vært trimmet") + p::text').getall()) #ok
        item['liens'] = " ".join(response.css('p:contains("heftelser/gjeld") + p::text').getall()) #ok

        item['dealer_rating'] = None
        item['dealer_n_ratings'] = None

        yield item
