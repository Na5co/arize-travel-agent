import json

from langchain_core.messages import AIMessage

from agent import tool_node


def tool_call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def run(*messages):
    return tool_node({"messages": list(messages)})["messages"][-1]


def test_hard_stop_blocks_a_flight_that_was_never_searched():
    result = run(tool_call("prepare_booking", flight_id="FR2648", travel_date="2027-08-10"))
    assert result.status == "error" and "Blocked" in result.content


def test_hard_stop_allows_a_searched_flight():
    search = tool_call("compare_flights", origin="London", origin_airports="LHR", origin_country="GB",
                       destination="Cagliari", destination_airports="CAG", destination_country="IT",
                       travel_date="2027-08-10")
    found = run(search)
    flight_id = json.loads(found.content)["cheapest_flight_id"]
    result = run(search, found, tool_call("prepare_booking", flight_id=flight_id, travel_date="2027-08-10"))
    assert result.status != "error" and "google.com/travel/flights" in result.content
