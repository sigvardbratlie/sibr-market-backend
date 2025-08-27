from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import HomeItem
from datetime import datetime
import json
from typing import Literal, Optional, List

class HomeSpider(FinnBaseSpider):
    name = "home"
    _table_name = "homes"

    _all_start_urls = {
        'Oslo' : 'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20061',
        'Akershus_1':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=1.20003.20046&location=1.20003.20047&location=1.20003.20045&location=1.20003.20058&location=1.20003.20051&location=1.20003.20042&location=1.20003.20055&location=1.20003.20060&location=1.20003.20099&location=1.20003.22105',
        'Akershus_2':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=1.20003.20100&location=1.20003.20052&location=1.20003.20059&location=1.20003.20057&location=1.20003.20043&location=1.20003.20054&location=1.20003.22104&location=1.20003.20050&location=1.20003.20056&location=1.20003.20039&location=1.20003.20041',
        'Agder':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.22042',
        'Buskerud':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20007',
        'Finnmark':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20020&location=0.20019&location=0.20009',
        'Innlandet':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.22034',
        'More_Rogaland':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20015&location=0.20012',
        'Nordland':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20018',
        'Vestland':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.22046',
        'Trønderlag':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20016',
        'Vestfold':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20008',
        'Østfold':'https://www.finn.no/realestate/homes/search.html?is_new_property=false&location=0.20002',
    }

    def __init__(self, *args, other_urls: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if other_urls:
            self.start_urls = other_urls
        else:
            self.start_urls = list(self._all_start_urls.values())

    use_playwright_listings = False
    use_playwright_items = False
    use_proxy = False

    custom_settings = {**FinnBaseSpider.custom_settings,
                       'LOG_LEVEL': 'INFO',
                       }

    @property
    def table_name(self):
        return self._table_name

    async def parse(self, response):

        #self.logger.info(f'Parsing {response.url}')
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

        item = HomeItem()

        #================= COMMON FIELDS =================
        item['item_id'] = response.css('th.pr-8:contains("FINN-kode") + td.pl-8::text' ).get()
        item['title'] = response.css('section[aria-label="Tittel"] h1::text').get()
        item['description'] = response.css('div.description-area.whitespace-pre-wrap ::text').getall()
        item['url'] = response.url
        item['address'] = response.css('span[data-testid="object-address"]::text').get()
        item['last_updated'] = response.css('th.pr-8:contains("Sist endret") + td.pl-8::text' ).get()
        item['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        item['country'] = 'NO'
        item['dealer'] = company_profile.get('orgName')
        item['contact_person'] = company_profile.get('contacts', [{}])[0].get('name') if company_profile.get('contacts') else company_profile.get('contact', {}).get('name')
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
        item['web'] = company_profile.get('homepageUrl',None)

        # ================= PRICE =================
        item['price'] = response.css('div[data-testid="pricing-incicative-price"] span.text-28.font-bold::text').get()
        item['total_price'] = response.css('div[data-testid="pricing-total-price"] dd::text').get()
        item['fees'] = response.css('div[data-testid="pricing-registration-charge"] dd::text').get()
        item['joint_debt'] = response.css('div[data-testid="pricing-joint-debt"] dd::text').get()
        item['monthly_common_cost'] = response.css('div[data-testid="pricing-common-monthly-cost"] dd::text').get()
        item['collective_assets'] = response.css('div[data-testid="pricing-collective-assets"] dd::text').get()
        item['tax_value'] = response.css('div[data-testid="pricing-tax-value"] dd::text').get()

        # ================= BASIC DETAILS =================
        item['new'] =  False
        item['sold'] = response.css('div.mb-24.py-4.px-8::text').get()
        item['district'] = response.css('div[data-testid="local-area-name"]::text').get()
        item['property_type'] = response.css('div[data-testid="info-property-type"] dd::text').get()
        item['ownership_type'] = response.css('div[data-testid="info-ownership-type"] dd::text').get()
        item['bedrooms'] = response.css('div[data-testid="info-bedrooms"] dd::text').get()
        item['rooms'] = response.css('div[data-testid="info-rooms"] dd::text').get()
        item['floor'] = response.css('div[data-testid="info-floor"] dd::text').get()
        item['balcony'] = response.css('div[data-testid="info-open-area"] dd::text').get()
        item['build_year'] = response.css('div[data-testid="info-construction-year"] dd::text').get()
        item['energy_rating'] = response.css('div[data-testid="energy-label"] dd span::text').get()
        item['primary_area'] = response.css('div[data-testid="info-primary-area"] dd::text').get()
        item['usable_area'] = response.css('div[data-testid="info-usable-area"] dd::text').get()
        item['internal_area'] = response.css('div[data-testid="info-usable-i-area"] dd::text').get()
        item['external_area'] = response.css('div[data-testid="info-usable-e-area"] dd::text').get()
        item['plot_size'] = response.css('div[data-testid="info-plot-area"] dd::text').get()
        item['plot'] = response.css('div[data-testid="info-plot"] dd::text').get()
        item['seller_insurance'] = response.css('div[data-testid="info-change-ownership-insurance"] dd::text').get()
        item['facilities'] = response.css('h2.h3[id="facilities-heading"] + div div::text').getall()

        item['municipality_num'] = response.css('div:contains("Kommunenr")::text').getall()
        item['cadastral_num'] = response.css('div:contains("Gårdsnr")::text').getall()
        item['unit_num'] = response.css('div:contains("Bruksnr")::text').getall()
        item['section_num'] = response.css('div:contains("Seksjonsnr")::text').getall()
        item['coop_unit_num'] = response.css('div:contains("Borettslag-andelsnummer")::text').getall()
        item['coop_name'] = response.css('div:contains("Borettslag-navn")::text').getall()
        item['coop_org_num'] = response.css('div:contains("Borettslag-orgnumme")::text').getall()
        item['apartment_num'] = response.css('div:contains("Leilighetsnr")::text').getall()
        item['leasehold_num'] = response.css('div:contains("Festenr")::text').getall()

        yield item

