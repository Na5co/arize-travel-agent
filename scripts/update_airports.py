"""Refresh data/airports.csv from OurAirports (public domain, rebuilt nightly).

Keeps only airports with scheduled passenger service and an IATA code.

    poetry run python scripts/update_airports.py
"""

import csv
import io
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://davidmegginson.github.io/ourairports-data/airports.csv"
TARGET = Path(__file__).resolve().parent.parent / "data" / "airports.csv"
COLUMNS = ["iata_code", "name", "municipality", "iso_country", "latitude_deg", "longitude_deg"]

with urlopen(SOURCE, timeout=60) as response:
    rows = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
    airports = sorted(
        ({column: row[column] for column in COLUMNS} for row in rows
         if row["scheduled_service"] == "yes" and len(row["iata_code"]) == 3),
        key=lambda airport: airport["iata_code"],
    )

TARGET.parent.mkdir(exist_ok=True)
with TARGET.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(airports)
print(f"Wrote {len(airports)} airports to {TARGET}")
