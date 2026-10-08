import json
import operator
from datetime import UTC, datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from typing_extensions import TypedDict

from tools.flights import compare_flights, prepare_booking

load_dotenv()

# The default description says "current events"; LLM judges read tool descriptions too, so it should say what it's for.
search_tool = DuckDuckGoSearchRun(description="Search the web for travel information: destination ideas, attractions, "
                                              "opening hours and travel tips. Input should be a search query.")
tools = [search_tool, compare_flights, prepare_booking]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

tracer = trace.get_tracer(__name__)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    system_prompt = (
        f"You are a travel assistant. Today is {datetime.now(UTC).date().isoformat()}.\n"
        "- Use web search for destination ideas and travel information.\n"
        "- Use compare_flights for flights. If the user gave a budget or wants direct flights, pass that; otherwise search without asking.\n"
        "- If a city or place is unclear, misspelled or not recognised, ask the user which one they mean. Never guess.\n"
        "- Always check with compare_flights. Never assume flights aren't available.\n"
        "- To book, show the user the options first. When they choose one, call prepare_booking and share the link.\n"
        "- You can NEVER book or pay. Never say a flight is booked. Never ask for or accept card details."
    )
    return {"messages": [model_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])]}


def searched_flight_ids(messages: list[AnyMessage]) -> set[str]:
    """Flight IDs that compare_flights returned earlier in this conversation."""
    return {
        flight["flight_id"]
        for m in messages
        if isinstance(m, ToolMessage) and m.name == "compare_flights" and m.status != "error"
        for flight in json.loads(m.content)["flights"]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message. Errors go back to the LLM instead of crashing."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        name, args = tool_call["name"], tool_call["args"]
        try:
            # Hard stop: only prepare a flight that a search actually returned.
            if name == "prepare_booking" and args["flight_id"].upper() not in searched_flight_ids(state["messages"]):
                raise PermissionError(f"Blocked: {args['flight_id']} was not returned by compare_flights. Search first.")
            observation = tools_by_name[name].invoke(args)
            content = observation if isinstance(observation, str) else observation.model_dump_json()
            result.append(ToolMessage(content=content, tool_call_id=tool_call["id"], name=name))
        except Exception as e:  # noqa: BLE001
            result.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"], name=name, status="error"))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    # The checkpointer keeps each conversation's messages under its thread_id (in memory).
    agent = graph_builder.compile(checkpointer=MemorySaver())
    return agent


def chat(agent, message: str, session_id: str) -> str:
    """Run one conversation turn, traced as one AGENT span with plain-text input and output."""
    with using_attributes(session_id=session_id), tracer.start_as_current_span("travel_agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute(SpanAttributes.SESSION_ID, session_id)
        span.set_attribute(SpanAttributes.INPUT_VALUE, message)
        result = agent.invoke({"messages": [HumanMessage(content=message)]},
                              config={"configurable": {"thread_id": session_id}})
        reply = result["messages"][-1].content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, reply)
    return reply
