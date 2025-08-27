import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.boat import BoatSpider
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        # 'https://www.finn.no/mobility/item/360649957',
                                        # 'https://www.finn.no/mobility/item/409737913',
                                        'https://www.finn.no/mobility/item/409736642',
                                        ])
def test_state(request):
    url = request.param
    r = requests.get(url)
    request = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=request)
    spider = BoatSpider()
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

def test_item_id(test_state):
    assert test_state['item_id'], 'Item id should be present'

def test_last_updated(test_state):
    assert test_state['last_updated'], 'Last updated should be present'

def test_total_price(test_state):
    assert test_state['total_price'], 'Total price should present'

def test_year(test_state):
    assert test_state['year'], 'Year should be present'
    assert len(test_state['year']) == 4, 'Year should be a 4 digit number'

def test_fuel(test_state):
    assert test_state['fuel'], 'Fuel type should be present'

def test_engine_type(test_state):
    assert test_state['engine_type'], 'Engine_Type should be present'

def test_seats(test_state):
    assert test_state['seats'], 'Seats should be present'

def test_description(test_state):
    assert test_state['description'], 'Description should be present'


def test_title(test_state):
    print(test_state['title'])
    assert test_state['title'], 'Title should be present'

def test_brand(test_state):
    print(test_state['brand'])
    assert test_state['brand'], 'Brand should be present'

def test_model(test_state):
    assert test_state['model'], 'Model should be present'

def test_address(test_state):
    assert test_state['address'], 'Address should be present'

def test_eng_size(test_state):
    assert test_state['engine_size'], 'Engine volume should be present'

def condition(test_state):
    assert test_state['condition'], 'Condition should be present'




