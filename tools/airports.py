"""Real airport data from OurAirports (public domain): checks the airport codes the LLM passes and
gives coordinates for generated flights. Refresh data/airports.csv with scripts/update_airports.py.
"""

import csv
import math
from pathlib import Path
from typing import NamedTuple


class Airport(NamedTuple):
    code: str
    name: str
    city: str
    country: str  # ISO code, e.g. "BG"
    lat: float
    lon: float


with (Path(__file__).resolve().parent.parent / "data" / "airports.csv").open(encoding="utf-8") as file:
    AIRPORTS = {row["iata_code"]: Airport(row["iata_code"], row["name"], row["municipality"], row["iso_country"],
                                          float(row["latitude_deg"]), float(row["longitude_deg"]))
                for row in csv.DictReader(file)}


def check_airports(codes: str, country: str) -> list[Airport]:
    """The airports for comma-separated IATA codes. Raises, with a message the LLM can act on,
    if a code isn't a scheduled-service airport or isn't in the given country."""
    airports = []
    for code in (c.strip().upper() for c in codes.split(",")):
        airport = AIRPORTS.get(code)
        if airport is None:
            raise ValueError(f"'{code}' is not an airport with scheduled flights. Check the IATA code.")
        if airport.country != country.strip().upper():
            raise ValueError(f"{code} is {airport.name}, {airport.city}, {airport.country}, not in {country}. "
                             "Check the code.")
        airports.append(airport)
    return airports


def distance_km(a: Airport, b: Airport) -> float:
    """Great-circle distance between two airports."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

