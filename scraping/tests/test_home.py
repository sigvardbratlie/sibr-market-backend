import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.home import HomeSpider
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        'https://www.finn.no/realestate/homes/ad.html?finnkode=409631696',
                                       'https://www.finn.no/realestate/homes/ad.html?finnkode=409687343',
                                        ])
def test_state(request):
    url = request.param
    r = requests.get(url)
    request = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=request)
    spider = HomeSpider()
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


def test_price(test_state):
    assert test_state['price'], 'Price should be present'
    print(test_state['price'])


# == PROPERTY INFO ==#

def test_area(test_state):
    assert test_state['primary_area'] or test_state['usable_area'] or test_state[
        'internal_area'], 'Area should be present'


def test_bedrooms(test_state):
    assert test_state['bedrooms'], 'Bedrooms should be present'
    print(test_state['bedrooms'])


def test_floor(test_state):
    assert test_state['floor'], 'Floor should be present'
    print(test_state['floor'])


# == DEALER INFO ==#
def test_dealer(test_state):
    assert test_state['dealer'], 'Dealer should be present'
    print(test_state['dealer'])


def test_contact_person(test_state):
    assert test_state['contact_person'], 'Contact person should be present'
    print(test_state['contact_person'])


def test_phone(test_state):
    assert test_state['phone'], 'Phone should be present'
    print(test_state['phone'])

def test_cadastral_num(test_state):
    assert test_state['cadastral_num'], 'Cadastral number should be present'
    print(test_state['cadastral_num'])


def test_unit_num(test_state):
    assert test_state['unit_num'], 'Unit number should be present'
    print(test_state['unit_num'])

def test_section_num(test_state):
    assert test_state['section_num'], 'Section number should be present'
    print(test_state['section_num'])