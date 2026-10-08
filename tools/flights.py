"""Flight tools. No third-party flight API: flights are generated for any route, deterministic per
route and date (so eval runs are reproducible) and sized by the real distance between the airports.

The agent can never book or pay: prepare_booking only returns a link where the
user books and pays on the provider's site.
"""

import random
from datetime import date
from urllib.parse import quote

from langchain_core.tools import tool
from pydantic import BaseModel

from tools.airports import Airport, check_airports, distance_km


class Flight(BaseModel):
    flight_id: str
    airline: str
    origin: str
    destination: str
    from_airport: str  # e.g. "Plovdiv International Airport (PDV)", so a wrong airport is visible
    to_airport: str
    departure_time: str
    duration_minutes: int
    stops: int
    price_eur: float


class FlightComparison(BaseModel):
    flights: list[Flight]  # sorted by price, cheapest first
    cheapest_flight_id: str | None
    fastest_flight_id: str | None
    budget_eur: float | None = None  # the user's budget converted to EUR, if one was given


class BookingPreparation(BaseModel):
    flight: Flight
    booking_link: str
    note: str = "Nothing has been booked or paid. The user books and pays on the provider's site."


# Fixed demo rates, so evals stay reproducible. Production would fetch daily rates (e.g. from the ECB).
EUR_PER_UNIT = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17}

SHORT_HAUL = [("FR", "Ryanair"), ("U2", "easyJet"), ("W6", "Wizz Air"), ("LH", "Lufthansa"),
              ("OS", "Austrian"), ("AF", "Air France"), ("KL", "KLM"), ("BA", "British Airways")]
LONG_HAUL = [("LH", "Lufthansa"), ("AF", "Air France"), ("BA", "British Airways"),
             ("TK", "Turkish Airlines"), ("EK", "Emirates"), ("QR", "Qatar Airways")]
SEARCHED_FLIGHTS: dict[str, Flight] = {}  # every generated flight, so prepare_booking can find it


def generate_flights(origin: str, destination: str, origins: list[Airport], destinations: list[Airport],
                     travel_date: str) -> list[Flight]:
    """Simulated flights. Seeded by the places and date, and sized by the distance between the first
    (main) airports listed, so small changes in the airport list don't change the flights."""
    rng = random.Random(f"{origin.strip().lower()}>{destination.strip().lower()}@{travel_date}")
    km = distance_km(origins[0], destinations[0])
    origins, destinations = sorted(origins), sorted(destinations)
    flights = []
    for i in range(rng.randint(2, 5)):
        stops = 1 if rng.random() < min(0.8, km / 5000) else 0
        code, airline = rng.choice(LONG_HAUL if km > 4000 else SHORT_HAUL)
        dep, arr = origins[i % len(origins)], destinations[i % len(destinations)]
        flight = Flight(
            flight_id=f"{code}{rng.randint(100, 9999)}", airline=airline, origin=origin, destination=destination,
            from_airport=f"{dep.name} ({dep.code})", to_airport=f"{arr.name} ({arr.code})",
            departure_time=f"{rng.randint(6, 21):02d}:{rng.choice((0, 15, 30, 45)):02d}",
            duration_minutes=round(km / 800 * 60 + 35 + stops * rng.randint(70, 180)), stops=stops,
            price_eur=round((40 + km * 0.09) * rng.uniform(0.7, 1.5) * (0.85 if stops else 1)),
        )
        SEARCHED_FLIGHTS[flight.flight_id] = flight
        flights.append(flight)
    return flights


@tool
def compare_flights(origin: str, origin_airports: str, origin_country: str,
                    destination: str, destination_airports: str, destination_country: str,
                    travel_date: str, max_price: float | None = None, currency: str = "EUR",
                    direct_only: bool = False) -> FlightComparison:
    """Compare flights between two places on a date. Returns options sorted by price.
    If a place is unclear or misspelled, ask the user to confirm it before calling this tool.

    Args:
        origin: Departure city, e.g. "London".
        origin_airports: IATA codes of the departure city's airports, comma-separated, e.g. "LHR,LGW,STN".
        origin_country: ISO country code of those airports, e.g. "GB".
        destination: Arrival city or region, e.g. "Cagliari" or "Sardinia".
        destination_airports: IATA codes of the arrival airports, e.g. "CAG", or "CAG,OLB,AHO" for Sardinia.
        destination_country: ISO country code of those airports, e.g. "IT".
        travel_date: Date of travel, YYYY-MM-DD.
        max_price: Only flights at or below this price, if the user gave a budget.
        currency: Currency of max_price as an ISO code, e.g. "USD" for "$". Converted to EUR in code.
            Only EUR, USD and GBP are supported. If the user's budget is in any other currency (e.g. denars,
            MKD, yen), don't call this tool yet: ask for the budget in EUR, USD or GBP. Never relabel a currency.
        direct_only: Only non-stop flights, if the user asked for direct flights.
    """
    date.fromisoformat(travel_date)  # raises on an invalid date
    # Checked against real airport data, so a valid-looking but wrong code (PVD for PDV) is caught.
    origins = check_airports(origin_airports, origin_country)
    destinations = check_airports(destination_airports, destination_country)
    if currency.upper() not in EUR_PER_UNIT:
        raise ValueError(f"Unsupported currency {currency!r}. Supported: {', '.join(EUR_PER_UNIT)}. "
                         "Ask the user for their budget in one of these; don't convert it yourself.")
    budget_eur = None if max_price is None else round(max_price * EUR_PER_UNIT[currency.upper()], 2)

    flights = sorted((f for f in generate_flights(origin, destination, origins, destinations, travel_date)
                      if (budget_eur is None or f.price_eur <= budget_eur) and (not direct_only or f.stops == 0)),
                     key=lambda f: f.price_eur)
    return FlightComparison(
        flights=flights,
        cheapest_flight_id=flights[0].flight_id if flights else None,
        fastest_flight_id=min(flights, key=lambda f: f.duration_minutes).flight_id if flights else None,
        budget_eur=budget_eur,
    )


@tool
def prepare_booking(flight_id: str, travel_date: str) -> BookingPreparation:
    """Prepare a flight the user chose: a summary and a link where the user books and pays.

    This does NOT book or pay for anything.

    Args:
        flight_id: A flight_id returned by compare_flights, e.g. "FR2648".
        travel_date: Date of travel, YYYY-MM-DD.
    """
    flight = SEARCHED_FLIGHTS.get(flight_id.strip().upper())
    if flight is None:
        raise ValueError(f"Unknown flight_id '{flight_id}'.")
    query = f"Flights from {flight.origin} to {flight.destination} on {travel_date} one way"
    return BookingPreparation(flight=flight, booking_link=f"https://www.google.com/travel/flights?q={quote(query)}")
