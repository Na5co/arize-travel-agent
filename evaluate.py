"""Evaluate the agent with two LLM judges, attach the results to the spans in Phoenix, and save the
frustrated turns as a dataset. Each judge runs on the span level the Phoenix docs recommend for it:

- user_friction, on each turn's root `travel_agent` span: does the user push back on the agent's
  earlier replies? Phoenix's built-in UserFrictionEvaluator. Input: the conversation before the
  message, and the message itself.
- tool_selection, on each LLM span: did the model pick the right tool (or rightly none)? Phoenix's
  built-in tool-selection criteria, plus scope rules (see TOOL_SELECTION_PROMPT). Input: the
  messages the model saw, the tools it was offered, and what it chose.

If results/tool_selection_reviewed.csv exists (hand-reviewed labels), the script also prints how
often the tool-selection judge agrees with them.

    poetry run python evaluate.py                  # both judges
    poetry run python evaluate.py tool_selection   # just one
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM, ClassificationEvaluator, evaluate_dataframe
from phoenix.evals.metrics import UserFrictionEvaluator
from phoenix.evals.utils import to_annotation_dataframe

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "default")
DATASET = f"frustrated-interactions-{PROJECT}"
JUDGES = set(sys.argv[1:]) or {"user_friction", "tool_selection"}
REVIEWED = Path("results/tool_selection_reviewed.csv")
client = Client()
judge = LLM(provider="openai", model="gpt-4o")

# Phoenix's built-in tool-selection criteria, word for word, plus a scope section. Reviewing every verdict by hand
# showed the built-in prompt doesn't say which decision to judge: with the whole conversation as input, it sometimes
# graded an earlier step (the search before an answer) or assumed a user reply that never happened.
TOOL_SELECTION_PROMPT = """You are an impartial judge evaluating ONE tool-calling decision made by an LLM agent: whether it selected the most appropriate tool or tools for the task, or correctly used none.

The data has three parts:
- <input>: the conversation so far. It is context only: tool calls and tool results in it were made by earlier steps and are NOT being judged.
- <available_tools>: the tools the agent could call, with their descriptions.
- <output>: the decision being judged, made right after the conversation in <input>: either tool call(s), or a reply with no tool call.

Scope:
- Judge only the decision in <output>.
- If <output> is a reply and the tool results it needs are already in <input>, answering without a tool is correct.
- Judge the conversation exactly as given. Do not assume user messages that are not in <input>.
- Follow usage rules stated in the tool descriptions (for example, asking the user to confirm an unclear or misspelled place before searching).
- A tool is required when it can answer the request and its required arguments are known. Asking for optional details first (for compare_flights: a budget or direct-only preference) instead of searching is incorrect.
- The agent can't book or pay, but when the user asks to book and the route and date are known, it should still search (or prepare a booking for a flight it already found). Refusing without doing either is incorrect.

Examples (not from the conversations being judged):
- <input> ends with "tool compare_flights returned: {...flights Madrid to Athens...}"; <output> is "No tool call. Reply: Here are the flights from Madrid to Athens..." → correct: the results are already there, so answering without a tool is right, whatever earlier steps did.
- <input> is only "user: Flights from Rome to Pariss on 2027-06-01?"; <output> is "No tool call. Reply: Did you mean Paris, France?" → correct: the place is misspelled, so confirming first follows the tool's rule. The user has not replied yet, so don't judge what happens next.
- <input> is only "user: Flights from Oslo to Nice on 2027-07-03?"; <output> is "No tool call. Reply: What is your budget, and do you want direct flights only?" → incorrect: origin, destination and date are known and the budget is optional, so it should have searched.

Criteria
Return "correct" only when ALL of the following are true:
- The LLM chose the best available tool for the user query OR correctly avoided tools if none were needed.
- The tool name exists in the available tools list.
- The tool is allowed and safe to call.
- The LLM selected the correct number of tools for the task.

Return "incorrect" if ANY of the following are true:
- The LLM used a hallucinated or nonexistent tool.
- The LLM selected a tool when none was needed.
- The LLM did not use a tool when one was required.
- The LLM chose a suboptimal or irrelevant tool.
- The LLM selected an unsafe or not-permitted tool.
- The tool name does not appear in the available tools list.

Before providing your final judgment, explain your reasoning and consider:
- What does the input context require at this point?
- Can this be answered without tools, or is a tool necessary?
- If a tool was selected, does it exist in the available tools?
- Does the selected tool's description match the user's needs?
- Is the selection safe and appropriate?
- Is there a better tool available that should have been chosen instead?

<data>
<input>
{{input}}
</input>

<available_tools>
{{available_tools}}
</available_tools>

<output>
{{tool_selection}}
</output>
</data>

Given the above data, is the decision in <output> correct or incorrect?"""


def tool_calls(message: dict) -> list[str]:
    return [f"{call['tool_call.function.name']}({call['tool_call.function.arguments']})"
            for call in message.get("message.tool_calls", [])]


def conversation(messages: list[dict]) -> str:
    """An LLM span's input messages as readable text, including earlier tool calls and results.
    The system prompt is left out: the judge checks the choice against the user's request, not the prompt under test."""
    lines = []
    for m in messages:
        role, content = m.get("message.role"), m.get("message.content") or ""
        if role == "user":
            lines.append(f"user: {content}")
        elif role == "assistant":
            lines += [f"assistant called {call}" for call in tool_calls(m)]
            if content:
                lines.append(f"assistant: {content}")
        elif role == "tool":
            lines.append(f"tool {m.get('message.name')} returned: {content}")
    return "\n".join(lines)


def run_of(session_ids: pd.Series) -> pd.Series:
    return session_ids.str.rsplit("-", n=1).str[1].rename("run")  # run_conversations.py names sessions "<name>-<run id>"


def evaluate(rows: pd.DataFrame, evaluator) -> pd.DataFrame:
    """Run one judge, attach its labels to the spans in Phoenix, and return the rows with label and explanation."""
    results = evaluate_dataframe(dataframe=rows, evaluators=[evaluator])
    annotations = to_annotation_dataframe(dataframe=results)
    client.spans.log_span_annotations_dataframe(dataframe=annotations, sync=True)
    scores = annotations.set_index("span_id")
    rows[f"{evaluator.name}_label"], rows[f"{evaluator.name}_explanation"] = scores["label"], scores["explanation"]
    print(f"\n{evaluator.name} (rows per run):\n{pd.crosstab(run_of(rows['session_id']), rows[f'{evaluator.name}_label'])}")
    return rows


spans = client.spans.get_spans_dataframe(project_identifier=PROJECT, limit=100_000, timeout=60)
Path("results").mkdir(exist_ok=True)

# 1. user_friction: one row per turn (root span), with the conversation before it from the same session.
if "user_friction" in JUDGES:
    rows = []
    for session_id, turns in spans[spans["name"] == "travel_agent"].sort_values("start_time").groupby("attributes.session.id"):
        history, messages = [], []
        for span_id, turn in turns.iterrows():
            user_message, reply = turn["attributes.input.value"], turn["attributes.output.value"]
            rows.append({"span_id": span_id, "session_id": session_id, "conversation": "\n".join(history) or "(no earlier messages)",
                         "user_message": user_message, "reply": reply,
                         "messages": messages + [{"role": "user", "content": user_message}]})  # chat format, for the dataset
            history += [f"user: {user_message}", f"assistant: {reply}"]
            messages = messages + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
    turns = evaluate(pd.DataFrame(rows).set_index("span_id"), UserFrictionEvaluator(llm=judge))
    turns.drop(columns=["messages"]).to_csv("results/user_friction.csv")

# 2. tool_selection: one row per LLM span, i.e. per decision the model made, judged on what the model saw.
if "tool_selection" in JUDGES:
    llm_spans = spans[spans["span_kind"] == "LLM"]
    decisions = pd.DataFrame({
        "session_id": llm_spans["attributes.session.id"],
        "input": llm_spans["attributes.llm.input_messages"].map(conversation),
        "available_tools": llm_spans["attributes.llm.tools"].map(lambda tools: "\n".join(
            "{name}: {description}".format(**json.loads(t["tool.json_schema"])["function"]) for t in tools)),
        "tool_selection": llm_spans["attributes.llm.output_messages"].map(
            lambda out: "\n".join(f"called {call}" for call in tool_calls(out[0]))
            or f"No tool call. Reply: {out[0].get('message.content')}"),
    }).rename_axis("span_id")
    tool_selection = ClassificationEvaluator(name="tool_selection", llm=judge, prompt_template=TOOL_SELECTION_PROMPT,
                                             choices={"correct": 1.0, "incorrect": 0.0}, direction="maximize")
    decisions = evaluate(decisions, tool_selection)
    decisions.drop(columns=["available_tools"]).to_csv("results/tool_selection.csv")

    if REVIEWED.exists():  # how often does the judge agree with the hand-reviewed labels?
        reviewed = pd.read_csv(REVIEWED).set_index("span_id")["reviewed_label"]
        both = decisions.join(reviewed, how="inner")
        agree = both["tool_selection_label"] == both["reviewed_label"]
        print(f"\nAgreement with hand-reviewed labels: {agree.sum()}/{len(both)} ({agree.mean():.0%})")
        for span_id, row in both[~agree].iterrows():
            print(f"  {span_id} {row.session_id}: judge {row.tool_selection_label}, reviewed {row.reviewed_label}")

# 3. Dataset of the frustrated turns in Phoenix's chat format (OpenAI-style input.messages, ending with the
#    frustrated message), so they can be replayed in the Playground or an experiment. Each links to its span.
if "user_friction" in JUDGES:
    frustrated = turns[turns["user_friction_label"] == "friction"]
    try:
        client.datasets.get_dataset(dataset=DATASET)
        print(f"\nDataset {DATASET} already exists, left unchanged.")
    except ValueError:  # not found
        if not frustrated.empty:
            client.datasets.create_dataset(
                name=DATASET,
                dataset_description="Turns the user_friction judge labelled as friction, as chat messages (input.messages).",
                examples=[{"input": {"messages": turn.messages},
                           "output": {"messages": [{"role": "assistant", "content": turn.reply}]},
                           "metadata": {"run": turn.session_id.rsplit("-", 1)[-1], "session_id": turn.session_id,
                                        "judge_explanation": turn.user_friction_explanation},
                           "span_id": span_id}
                          for span_id, turn in frustrated.iterrows()],
            )
            print(f"\nCreated dataset {DATASET} with {len(frustrated)} turns")
