import pytest

from tools.flights import compare_flights, prepare_booking


def search(**kwargs):
    return compare_flights.invoke({
        "origin": "London", "origin_airports": "LHR,LGW", "origin_country": "GB",
        "destination": "Lisbon", "destination_airports": "LIS", "destination_country": "PT",
        "travel_date": "2027-08-13"} | kwargs)


def test_sorted_by_price_with_cheapest_and_fastest():
    result = search()
    assert result.flights and [f.price_eur for f in result.flights] == sorted(f.price_eur for f in result.flights)
    assert result.cheapest_flight_id == result.flights[0].flight_id
    assert result.fastest_flight_id == min(result.flights, key=lambda f: f.duration_minutes).flight_id


def test_same_question_gives_the_same_flights():
    assert search() == search()
    key = lambda result: [(f.flight_id, f.price_eur, f.duration_minutes) for f in result.flights]
    assert key(search()) == key(search(origin_airports="LHR,LGW,STN,LTN,LCY,SEN"))  # same main airport listed first


def test_flights_are_sized_by_real_distance():
    plovdiv_vienna = search(origin="Plovdiv", origin_airports="PDV", origin_country="BG",
                            destination="Vienna", destination_airports="VIE", destination_country="AT")
    assert all(80 <= f.duration_minutes <= 120 for f in plovdiv_vienna.flights if f.stops == 0)  # ~830 km: about 1.5 h
    assert plovdiv_vienna.flights[0].from_airport == "Plovdiv International Airport (PDV)"


def test_budget_and_direct_filters():
    cheapest = search().flights[0].price_eur
    assert [f.price_eur for f in search(max_price=cheapest).flights] == [cheapest]
    assert all(f.stops == 0 for f in search(direct_only=True).flights)


def test_budget_in_another_currency_is_converted_in_code():
    assert search(max_price=100, currency="USD").budget_eur == 92.0  # $100 = EUR 92
    with pytest.raises(ValueError, match="Unsupported currency"):
        search(max_price=100, currency="JPY")


def test_airport_codes_are_checked_against_real_data():
    with pytest.raises(ValueError, match="not an airport"):
        search(destination_airports="XYZ")
    with pytest.raises(ValueError, match="Providence"):  # PVD is a real code, but in the US, not Bulgaria
        search(destination="Plovdiv", destination_airports="PVD", destination_country="BG")


def test_prepare_booking_returns_a_link_for_a_searched_flight_and_books_nothing():
    flight_id = search().cheapest_flight_id
    result = prepare_booking.invoke({"flight_id": flight_id.lower(), "travel_date": "2027-08-13"})
    assert result.flight.flight_id == flight_id
    assert result.booking_link.startswith("https://www.google.com/travel/flights?q=")
    assert "Nothing has been booked" in result.note


def test_prepare_booking_rejects_unknown_flight():
    with pytest.raises(ValueError, match="Unknown flight_id"):
        prepare_booking.invoke({"flight_id": "XX9999", "travel_date": "2027-08-13"})
