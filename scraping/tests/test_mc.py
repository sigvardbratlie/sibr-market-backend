import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.mc import McSpider
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        'https://www.finn.no/mobility/item/409739269',
                                        # 'https://www.finn.no/mobility/item/346633091',
                                        'https://www.finn.no/mobility/item/409550114',
                                        ])
def test_state(request):
    url = request.param
    r = requests.get(url)
    request = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=request)
    spider = McSpider()
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

def test_total_price(test_state):
    assert test_state['total_price'], 'Total price should present'

def test_year(test_state):
    assert test_state['year'], 'Year should be present'
    assert len(test_state['year']) == 4, 'Year should be a 4 digit number'

def test_milage(test_state):
    assert test_state['mileage'], 'Mileage should be present'
    assert 'km' in test_state['mileage'], 'Milage should be in kilometers'

def test_fuel(test_state):
    assert test_state['fuel'], 'Fuel type should be present'

def test_type(test_state):
    assert test_state['type'], 'Type should be present'

def test_description(test_state):
    assert test_state['description'], 'Description should be present'

def test_features(test_state):
    assert test_state['features'], 'Features should be present'

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

def test_last_updated(test_state):
    assert test_state['last_updated'], 'Last updated should be present'

def test_power(test_state):
    assert test_state['power'], 'Power should be present'

def test_eng_vol(test_state):
    assert test_state['engine_volume'], 'Engine volume should be present'

def test_reg_num(test_state):
    print(test_state['reg_num'])
    assert test_state['reg_num'], 'Registration number should be present'

def condition(test_state):
    assert test_state['condition'], 'Condition should be present'


# def test_dealer(test_state):
#     assert test_state['dealer'], 'Dealer should be present'


