# Travel Agent: LangGraph + Arize Phoenix

The starter's LangGraph agent, extended into a travel assistant that compares flights and **prepares** bookings, traced and evaluated with [Arize Phoenix](https://arize.com/docs/phoenix). It never books or pays: it hands the user a link.

> [!WARNING]
> This project is deliberately kept simple so it runs anywhere with just an OpenAI key. These parts need rework before production:
>
> | Now | Production |
> |---|---|
> | Generated flights (repeatable evals) | Real flight API (e.g. SerpApi) |
> | Booking link: a Google Flights search for the route and date; the generated flight won't be there | The provider's deep link for the chosen flight |
> | No authentication | Sessions tied to a logged-in user |
> | In-memory conversations | Persistent checkpointer (e.g. `PostgresSaver`) |
> | Test script calls the agent directly with readable session IDs (`beach-trip-d70b70`) | Server-issued IDs everywhere; scenario and run as span metadata |
> | Card details: prompt rule only | Redact card numbers before the LLM sees them |

## Quick start

Requires Python 3.12, [Poetry](https://python-poetry.org/docs/#installation) and an OpenAI API key.

```bash
poetry install
cp .env.example .env   # then set OPENAI_API_KEY

# Terminal 1: Phoenix (UI at http://localhost:6006)
PHOENIX_WORKING_DIR="$PWD/.phoenix" uvx --python 3.12 --from "arize-phoenix==20.19.0" phoenix serve

# Terminal 2: the agent API (docs at http://localhost:8000/docs)
poetry run uvicorn api:app --reload
```

Or both with Docker: `docker compose up --build`.

```bash
curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Flights from London to Lisbon on 2027-08-13?"}'
# {"response": "Here are the flights…", "session_id": "3f9c…"}  → send session_id back to continue
```

Postman: `travel-agent-demo.postman_collection.json` plays one short conversation (run its folder); `travel-agent.postman_collection.json` has every scenario. Tests: `poetry run pytest` (offline, no LLM calls).

## Agent architecture

```text
POST /chat {message, session_id?}  →  api.py  →  agent.py (LangGraph)
                                                   START → llm_call ⇄ tool_node → END
                                                   tools: duckduckgo_search (starter), compare_flights, prepare_booking
                                                   memory: MemorySaver, one thread per session_id
spans (OpenTelemetry + OpenInference)  →  Phoenix: traces, sessions, evals, datasets
```

The LLM decides (answer, or request a tool); `tool_node` runs the tool in Python and returns the result; repeat until the LLM answers.

## Design decisions

**Structured tools** (`tools/flights.py`, Pydantic output)
- `compare_flights`: flights sorted by price, plus the cheapest and fastest IDs.
- `prepare_booking`: the chosen flight and a Google Flights link.

**Code checks what the LLM fills in**
- Currency is converted in code, and airport codes are checked against [real airport data](https://ourairports.com/data/), so `PVD` instead of `PDV` is caught.

**Soft stop vs hard stop**
- Soft (prompt): never book, pay or take card details; ask back when a place is unclear.
- Hard (code): no booking tool exists, and `prepare_booking` only runs for a flight a search returned.

**Sessions**
- The API issues a `session_id`; the checkpointer replays that conversation to the LLM each turn.
- The same ID groups the turns in Phoenix's Sessions view.

## Observability

`register(auto_instrument=True)` (`phoenix.otel`) sets up OpenTelemetry and the OpenInference LangChain instrumentor, so every node, LLM call and tool call becomes a span. Each turn is wrapped in a `travel_agent` span with the plain user message and reply.

```text
travel_agent          AGENT   user message → reply
└─ LangGraph          CHAIN
   ├─ llm_call        CHAIN
   │  └─ ChatOpenAI   LLM     prompt, tool call requested, tokens
   ├─ tool_node       CHAIN
   │  └─ compare_flights TOOL arguments → structured result
   └─ llm_call → ChatOpenAI   final answer
```

## Evaluation

```bash
poetry run python run_conversations.py   # 11 conversations, 20 queries → Phoenix
poetry run python export_spans.py        # all spans → results/spans.csv
poetry run python evaluate.py            # two LLM judges → span annotations, results/*.csv, dataset
```

| Eval | Runs on | Labels |
|---|---|---|
| `user_friction` (Phoenix built-in) | each turn (`travel_agent` span), with the conversation before it | `friction` / `no_friction` |
| `tool_selection` (Phoenix built-in criteria + scope rules) | each LLM decision (`ChatOpenAI` span) | `correct` / `incorrect` |

Labels are logged to Phoenix as span annotations. Frustrated turns become the dataset `frustrated-interactions-<project>` (chat format, linked to their spans), ready to replay in the Playground or an experiment.

In Phoenix (Spans tab): `annotations['user_friction'].label == 'friction'` or `annotations['tool_selection'].label == 'incorrect'`.

**Results:** `results/*.csv`.

**Checking the judge:** its verdicts were compared with a hand review of all 63 decisions (`results/tool_selection_reviewed.csv`), and scope rules were added to the built-in prompt where it graded the wrong step.

## Debugging with PX CLI + Phoenix skills

```bash
npm install -g @arizeai/phoenix-cli
px profile create local --endpoint http://localhost:6006 --project travel-agent --activate   # your PHOENIX_PROJECT_NAME
px trace list --limit 3              # latest turns as span trees
px span list --span-kind TOOL        # every tool call
npx skills add Arize-ai/phoenix --skill phoenix-cli --skill phoenix-tracing --skill phoenix-evals
```
