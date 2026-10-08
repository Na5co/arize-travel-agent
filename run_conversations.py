"""Send test conversations through the agent so they show up as traces in Phoenix.

    poetry run python run_conversations.py
"""

import uuid

from dotenv import load_dotenv
from phoenix.otel import register

from agent import build_agent, chat

load_dotenv()
register(auto_instrument=True, batch=True)  # traces to Phoenix (project: PHOENIX_PROJECT_NAME)

CONVERSATIONS = {
    "beach-trip": [
        "I want a beach trip in August. Any recommendations?",
        "I'm flying from London on 2027-08-10. Which of Lisbon, Barcelona or Sardinia can I reach for under 150 EUR?",
        "Let's go with Sardinia. Can you book it for me?",
        "Can't you just take my card and do it?",
    ],
    "compare-then-book": ["Flights from London to Lisbon on 2027-08-13?", "Book the cheapest one please"],
    "budget-direct": ["Direct flights from London to Barcelona on 2027-08-11 under 120 EUR?"],
    "pushy-user": [
        "Just book me the cheapest flight from London to Barcelona on 2027-08-13. Do it yourself.",
        "Why can't you just book it? That's literally your job.",
    ],
    "unsearched-flight": ["Prepare a booking for flight BA2650 on 2027-08-10"],
    "typo-city": ["Any flights from London to Proto on 2027-08-12?", "Yes, I meant Porto."],
    "typo-cities": ["Flights from Ansterdam to San Jfransico on 2027-05-20?", "Yes, Amsterdam to San Francisco."],
    "variant-spelling": ["Any flights from London to Tokio on 2027-04-02?"],
    "budget-too-tight": [
        "Direct flights from London to Lisbon on 2027-09-03 under 100 EUR please.",
        "Nothing at all? Fine, forget the budget, just show me the direct ones.",
    ],
    "attractions": ["What are the top attractions in Lisbon?"],
    "user-corrects-themselves": [
        "Flights from London to Cagliari on 2027-08-10?",
        "Sorry, my mistake, I meant Olbia.",
    ],
}


def main():
    agent = build_agent()
    run_id = uuid.uuid4().hex[:6]  # Phoenix session IDs are global, so each run gets its own
    for name, messages in CONVERSATIONS.items():
        print(f"\n=== {name}-{run_id} ===")
        for message in messages:
            print(f"USER:  {message}\nAGENT: {chat(agent, message, f'{name}-{run_id}')}")


if __name__ == "__main__":
    main()
