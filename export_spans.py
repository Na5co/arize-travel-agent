"""Export every span of the Phoenix project to results/spans.csv.

    poetry run python export_spans.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from phoenix.client import Client

load_dotenv()

spans = Client().spans.get_spans_dataframe(project_identifier=os.getenv("PHOENIX_PROJECT_NAME", "default"),
                                           limit=100_000, timeout=60)
Path("results").mkdir(exist_ok=True)
spans.to_csv("results/spans.csv")
print(f"Exported {len(spans)} spans to results/spans.csv")
