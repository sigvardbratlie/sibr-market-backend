import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.new_home import NewHomeSpider
from finn_scraper.items import NewHomeItem
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        'https://www.finn.no/realestate/project/ad.html?finnkode=293962961',
                                       'https://www.finn.no/realestate/project/ad.html?finnkode=402303414',
                                        'https://www.finn.no/realestate/projectunit/ad.html?finnkode=402306776',
                                        ])
def test_state(request): # 'request' her er pytest sin fixture request, ikke Scrapy Request
    url = request.param
    r = requests.get(url)
    # Gi Scrapy Request-objektet et annet navn for å unngå forveksling
    scrapy_request_obj = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=scrapy_request_obj)
    spider = NewHomeSpider()
    items_collected = [] # Endret navn for klarhet

    async def parse_page():
        async for yielded_object in spider.parse(response):
            # HER ER DEN VIKTIGE FILTRERINGEN:
            if isinstance(yielded_object, NewHomeItem): # Sjekk om det er et NewHomeItem
                items_collected.append(yielded_object)
            # Alternativt, hvis du vil ha alle objekter som IKKE er en Request:
            # if not isinstance(yielded_object, Request):
            #     items_collected.append(yielded_object)

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError: # Kan også skje hvis ingen current loop er satt i tråden
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(parse_page())

    # Legg til en sjekk for å sikre at du faktisk har samlet inn items
    assert items_collected, f"Ingen NewHomeItem-objekter ble yieldet for URL: {url}"
    return items_collected[0]


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


def test_completion(test_state):
    assert test_state['completion'], 'Completion date should be present'
    print(test_state['completion'])

def test_planning(test_state):
    assert test_state['planning'], 'Planning date should be present'
    print(test_state['planning'])