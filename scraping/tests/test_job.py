import pytest
from scrapy import Request
from scrapy.http import TextResponse
from finn_scraper.spiders.job import JobSpider
import requests
import asyncio


@pytest.fixture(scope='module',params=[
                                        'https://www.finn.no/job/fulltime/ad.html?finnkode=409638349',
                                       'https://www.finn.no/job/fulltime/ad.html?finnkode=409033362',
                                        'https://www.finn.no/job/positions/ad.html?finnkode=408937806',
                                        'https://www.finn.no/job/positions/ad.html?finnkode=408944694',
                                        ])
def test_state(request):
    url = request.param
    r = requests.get(url)
    request = Request(url, meta={"example_meta_key": "example_meta_value"})
    response = TextResponse(r.url, body=r.text, encoding='utf-8', request=request)
    spider = JobSpider()
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

def test_title(test_state):
    assert test_state['title'], 'Title should be present'
    print(test_state['title'])

def test_description(test_state):
    assert test_state['description'], 'Description should be present'
    print(test_state['description'])

def test_location(test_state):
    assert test_state['location'], 'Location should be present'
    print(test_state['location'])

def test_employer(test_state):
    assert test_state['employer'], 'Employer should be present'
    print(test_state['employer'])

# def test_contact_person(test_state):
#     assert test_state['contact_person'], 'Contact person should be present'
#     print(test_state['contact_person'])

def test_sector(test_state):
    assert test_state['sector'], 'Sector should be present'
    print(test_state['sector'])

def test_adresse(test_state):
    assert test_state['address'], 'Address should be present'
    print(test_state['address'])


def test_deadline(test_state):
    assert test_state['deadline'], 'Deadline should be present'
    print(test_state['deadline'])

def test_jobfunction(test_state):
    assert test_state['job_function'], 'Job function should be present'
    print(test_state['job_function'])

def test_itemid(test_state):
    assert test_state['item_id'], 'Item ID should be present'
    print(test_state['item_id'])

def test_last_updated(test_state):
    assert test_state['last_updated'], 'Last updated should be present'
    print(test_state['last_updated'])

