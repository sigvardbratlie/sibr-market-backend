from finn_scraper.spiders.finn_base import FinnBaseSpider
from finn_scraper.items import NewHomeItem
from datetime import datetime
import json
import scrapy


class NewHomeSpider(FinnBaseSpider):
    name = "new_home"
    _table_name = "new_homes"
    start_urls = [
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.22042&location=0.20007', #agder og buskerud
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.20020&location=0.22034&location=0.20015&location=0.20018', #Finnmark, Møre og Romsdal, Nordland, Innlandet,
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.20061&location=0.20009&location=0.20019&location=0.20008' #oslo, telemark, troms og vestfold,
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.20012' #Rogoland,
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.20016' #Trøndelag,
                'https://www.finn.no/realestate/newbuildings/search.html?filters=&location=0.22046&location=0.20002', #vestlandet og østfold
                'https://www.finn.no/realestate/homes/search.html?filters=&is_new_property=true&location=1.20003.20041&location=1.20003.20039&location=1.20003.20056&location=1.20003.20050&location=1.20003.22104&location=1.20003.20054&location=1.20003.20043&location=1.20003.20057&location=1.20003.20059',
                'https://www.finn.no/realestate/homes/search.html?filters=&is_new_property=true&location=1.20003.20052&location=1.20003.20100&location=1.20003.20099&location=1.20003.22105&location=1.20003.20060&location=1.20003.20055&location=1.20003.20042&location=1.20003.20051&location=1.20003.20058&location=1.20003.20045&location=1.20003.20047&location=1.20003.20046',
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

        unit_selector_links = response.css(
            'div[data-testid="unit-selector"] tbody td a[tabindex="0"]::attr(href)').getall()
        # Bytt navn fra unit_selector til unit_selector_links for klarhet, valgfritt

        meta_for_new_requests = {'errback': self.errback}  # Definer self.errback i spideren din
        if self.use_playwright_items:
            meta_for_new_requests['playwright'] = True
            meta_for_new_requests['playwright_include_page'] = True
            # Vurder også playwright_context hvis du bruker spesifikke kontekster
            # meta_for_new_requests['playwright_context'] = response.meta.get('playwright_context_name', 'default')

        # self.logger.info(f"Found {len(unit_selector_links)} unit selector links on {response.url}")
        if unit_selector_links:  # Enklere sjekk for om listen ikke er tom
            for href in unit_selector_links:
                # BRUK response.urljoin(href) for å sikre absolutte URL-er
                absolute_url = response.urljoin(href)
                # self.logger.info(f"Yielding request for {absolute_url}")
                yield scrapy.Request(url=absolute_url, meta=meta_for_new_requests, callback=self.parse)

        item = NewHomeItem()

        company_data = {}
        script_tag_text = response.xpath('//script[@data-company-profile-data]/text()').get()
        try:
            company_data = json.loads(script_tag_text) if script_tag_text else {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e} in {response.url}")
        company_profile = company_data.get('companyProfile', {})

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
        item['new'] =  True
        item['district'] = response.css('section[aria-label="Tittel"] h2::text').get()
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

        #timeline
        steps = response.css(r'section[data-testid="project-phase"] ul li.group\/step')
        if steps:
            for step in steps:
                title = step.css('h4::text').get().strip()
                date = step.css('p::text').get().strip()

                if title == "Planlegging":
                    item['planning'] = date
                elif title == "Salgsstart":
                    item['sales_start'] = date
                elif title == "Byggestart":
                    item['construction_start'] = date
                elif title == "Overtakelse":
                    item['completion'] = date

        yield item

