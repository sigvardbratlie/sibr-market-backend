# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item,Field

from scrapy import Item, Field

class FinnItem(Item):
    item_id = Field()
    title = Field()
    description = Field()
    url = Field()
    address = Field()
    last_updated = Field()
    scrape_date = Field()
    country = Field()
    dealer = Field()
    contact_person = Field()
    phone = Field()
    email = Field()
    web = Field()
    dealer_rating = Field()  # ok
    dealer_n_ratings = Field()  # ok

class CarItem(FinnItem):
    #PRICES
    total_price = Field()

    # Basic details
    subtitle = Field()
    model_year = Field()
    mileage = Field()
    gearbox = Field()
    fuel = Field()

    features = Field()

    # Car details
    brand = Field()
    model = Field()
    body_type = Field()
    color = Field()
    color_description = Field()
    color_interior = Field()
    wheel_drive = Field()
    seats = Field()
    doors = Field()
    cargo_space = Field()
    power = Field()
    engine_volume = Field()
    fuel_cons = Field()
    battery = Field()
    range = Field()
    reg_num = Field()
    vin = Field()
    category = Field()
    trailer_weight = Field()
    length = Field()
    height = Field()
    width = Field()
    weight = Field()
    manufacturer = Field()
    gearbox_type = Field()


    # ownership details
    state = Field()
    prev_owners = Field()
    sales_type = Field()
    transfer_fee = Field()
    price_excl_transfer = Field()
    first_registration = Field()
    last_eu = Field()
    next_eu = Field()
    warranty = Field()
    warranty_length = Field()
    warranty_until = Field()
    co2 = Field()
    emission_class = Field()
    emission_sticker = Field()
    service_history = Field()
    service_book = Field()
    condition_report = Field()
    vat_deductible = Field()

    # Additional details
    known_issues = Field()
    major_repairs = Field()
    engine_tuned = Field()
    liens = Field()


class JobItem(FinnItem):
    contact_title = Field() #ok
    subtitle = Field() #ok
    employer = Field() #ok
    about_employer = Field() #ok
    sector = Field() #ok
    industry = Field() #ok
    job_function = Field()  # ok
    deadline = Field() #ok
    employment_type = Field() #ok
    positions_available = Field() #ok
    work_language = Field() #ok
    remote_work = Field() #ok
    location = Field() #ok
    keywords = Field() #ok

class NewHomeItem(FinnItem):
    new = Field() #ok
    district = Field()
    price = Field() #ok
    total_price = Field() #ok
    fees = Field() #ok
    joint_debt = Field() #ok
    monthly_common_cost = Field() #ok
    collective_assets = Field() #ok
    tax_value = Field() #ok
    property_type = Field() #ok
    ownership_type = Field() #ok
    bedrooms = Field() #ok
    rooms = Field() #ok
    floor = Field() #ok
    balcony = Field() #ok
    build_year = Field() #Ok
    energy_rating = Field() #OK
    primary_area = Field() #ok
    usable_area = Field() #ok
    internal_area = Field() #ok
    external_area = Field() #ok
    plot_size = Field() #ok
    seller_insurance = Field() #ok
    plot = Field() #ok
    facilities = Field() #ok

    # Timeline
    planning = Field()  # ok
    sales_start = Field()  # ok
    construction_start = Field()  # ok
    completion = Field()  # ok

    municipality_num = Field()  # ok
    cadastral_num = Field()  # ok
    unit_num = Field()  # ok
    section_num = Field()  # ok
    coop_unit_num = Field()  # ok
    coop_name = Field()  # ok
    coop_org_num = Field()  # ok
    apartment_num = Field()  # ok
    leasehold_num = Field()  # ok

class HomeItem(FinnItem):
    new = Field() #ok
    sold = Field() #ok
    district = Field()
    price = Field() #ok
    total_price = Field() #ok
    fees = Field() #ok
    joint_debt = Field() #ok
    monthly_common_cost = Field() #ok
    collective_assets = Field() #ok
    tax_value = Field() #ok
    property_type = Field() #ok
    ownership_type = Field() #ok
    bedrooms = Field() #ok
    rooms = Field() #ok
    floor = Field() #ok
    balcony = Field() #ok
    build_year = Field() #Ok
    energy_rating = Field() #OK
    primary_area = Field() #ok
    usable_area = Field() #ok
    internal_area = Field() #ok
    external_area = Field() #ok
    plot_size = Field() #ok
    seller_insurance = Field() #ok
    plot = Field() #ok
    facilities = Field() #ok

    municipality_num = Field() #ok
    cadastral_num = Field() #ok
    unit_num = Field() #ok
    section_num = Field() #ok
    coop_unit_num = Field() #ok
    coop_name = Field() #ok
    coop_org_num = Field() #ok
    apartment_num = Field() #ok
    leasehold_num = Field() #ok

class BoatItem(FinnItem):
    total_price = Field() #Ok
    features = Field() #Ok
    # Basic details
    year = Field() #ok
    length = Field() #ok
    engine_type = Field() #ok
    seats = Field() #ok

    # Boat details
    condition = Field() #ok
    brand = Field() #ok
    model = Field() #ok
    type = Field() #ok
    fuel = Field() #ok
    engine_included = Field() #ok
    engine_size = Field() #ok
    engine_manufacturer = Field() #ok
    max_speed = Field() #ok
    building_material = Field() #ok
    depth = Field() #ok
    width = Field() #ok
    location = Field() #ok
    sleeping_places = Field() #ok
    color = Field() #ok
    reg_num = Field() #ok


class MCItem(FinnItem):
    total_price = Field() #Ok

    # Basic details
    year = Field() #ok
    mileage = Field() #ok
    engine_volume = Field() #ok
    power = Field()  # ok

    # MC details
    type = Field()  # ok
    brand = Field()  # ok
    model = Field()  # ok
    fuel = Field() #ok
    color = Field() #ok
    battery = Field() #ok
    range = Field() #ok
    vin = Field() #ok
    reg_num = Field() #ok
    condition = Field() #ok
    weight = Field() #ok
    manufacturer = Field() #ok
    height = Field() #ok
    length = Field() #ok
    width = Field() #ok

    # Ownership details
    prev_owners = Field() #ok
    sales_type = Field() #ok
    transfer_fee = Field() #ok
    price_excl_transfer = Field() #ok
    first_registration = Field() #ok
    last_eu = Field() #ok
    next_eu = Field() #ok
    warranty = Field() #ok
    warranty_until = Field() #ok
    warranty_length = Field() #ok
    service_history = Field() #ok
    service_book = Field() #ok
    condition_report = Field() #ok
    features = Field()


class RentalItem(FinnItem):
    property_type = Field() #ok
    primary_area = Field() #ok
    usable_area = Field() #ok
    internal_area = Field() #ok
    external_area = Field() #ok
    gross_area = Field() #ok
    balcony = Field() #ok
    floor = Field() #ok
    bedrooms = Field() #ok
    monthly_rent = Field() #ok
    deposit = Field() #ok
    build_year = Field() #ok
    energy_rating = Field() #ok
    includes = Field() #ok
    facilities = Field() #ok
    plot_size = Field() #ok

