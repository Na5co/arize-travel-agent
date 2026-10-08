import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from phoenix.otel import register
from pydantic import BaseModel

from agent import build_agent
from agent import chat as run_chat

load_dotenv()
register(auto_instrument=True, batch=True)  # traces to Phoenix (project: PHOENIX_PROJECT_NAME)

agent = build_agent()

app = FastAPI(
    title="Travel Agent API",
    description="A LangGraph travel agent that compares flights and prepares bookings",
    version="0.2.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent. Send the returned session_id back to continue the conversation."""
    if request.session_id is None:
        session_id = uuid.uuid4().hex  # issued by the server, never chosen by the client
    elif agent.get_state({"configurable": {"thread_id": request.session_id}}).values:
        session_id = request.session_id
    else:
        raise HTTPException(status_code=404, detail="Unknown session_id. Omit it to start a new conversation.")
    return ChatResponse(response=run_chat(agent, request.message, session_id), session_id=session_id)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
