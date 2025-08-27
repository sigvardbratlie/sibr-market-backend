import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.rental import RentalSpider
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        'https://www.finn.no/realestate/lettings/ad.html?finnkode=409112383',
                                        'https://www.finn.no/realestate/lettings/ad.html?finnkode=379488208',
                                       # 'https://www.finn.no/realestate/lettings/ad.html?finnkode=328561031',
                                       #  'https://www.finn.no/realestate/lettings/ad.html?finnkode=402268799',
                                        ])
def test_state(request):
    url = request.param
    r = requests.get(url)
    request = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=request)
    spider = RentalSpider()
    items = []

    async def parse_page():
        async for item in spider.parse(response):
            items.append(item)

    # Opprett en ny event loop hvis den gamle er lukket
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Kjør den asynkrone koden
    loop.run_until_complete(parse_page())

    return items[0]

# == COMMON INFO ==#

def test_item_id(test_state):
    assert test_state['item_id'], 'Item ID should be present'
    assert len(test_state['item_id']) > 7, 'Item ID should be longer than 7 characters'
    print(test_state['item_id'])

def test_last_updated(test_state):
    assert test_state['last_updated'], 'Last updated date should be present'
    print(test_state['last_updated'])

def test_title(test_state):
    assert test_state['title'], 'Title should be present'
    print(test_state['title'])

def test_description(test_state):
    assert test_state['description'], 'Description should be present'
    print(test_state['description'])

def test_address(test_state):
    assert test_state['address'], 'Location should be present'
    print(test_state['address'])

def test_rent(test_state):
    assert test_state['monthly_rent'], 'rent should be present'
    print(test_state['monthly_rent'])

#== PROPERTY INFO ==#

def test_area(test_state):
    assert test_state['primary_area'] or test_state['usable_area'] or test_state['internal_area'] or test_state['gross_area'], 'Area should be present'

def test_bedrooms(test_state):
    assert test_state['bedrooms'], 'Bedrooms should be present'
    print(test_state['bedrooms'])

def test_floor(test_state):
    assert test_state['floor'], 'Floor should be present'
    print(test_state['floor'])

#== DEALER INFO ==#
def test_dealer(test_state):
    assert test_state['dealer'], 'Dealer should be present'
    print(test_state['dealer'])

def test_contact_person(test_state):
    assert test_state['contact_person'], 'Contact person should be present'
    print(test_state['contact_person'])

def test_phone(test_state):
    assert test_state['phone'], 'Phone should be present'
    print(test_state['phone'])

