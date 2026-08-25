"""HTTP front door for the banking ADK agent.

Wraps the existing `banking_agent` package without changing it: same agents,
same tools, same SQLite database. The console page posts a question here and
gets back the answer plus the trace of what the agents did to produce it.

Run locally:
    uvicorn api.main:app --reload --port 8000

Environment:
    GOOGLE_API_KEY        required — same key the local agent uses
    ALLOWED_ORIGINS       comma-separated list, or * (default)
    DAILY_QUESTION_LIMIT  model calls allowed per day (default 5)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The agent reads its key from banking_agent/.env locally; on Render the
# same names come from the dashboard's environment variables.
load_dotenv(Path(__file__).resolve().parents[1] / "banking_agent" / ".env")

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from banking_agent.agent import root_agent  # noqa: E402

APP_NAME = "banking_conversational_ai"
ROOT_AGENT_NAME = "banking_agent"
SUB_AGENT_NAME = "customer_agent"

DAILY_QUESTION_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", "5"))
MAX_MESSAGE_CHARS = 500

app = FastAPI(title="Banking Agent API", version="1.0.0")

_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else
                  [o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# ---------------------------------------------------------------------------
# Daily budget — the whole point is that a public page can't run up a bill
# ---------------------------------------------------------------------------

_usage: dict[str, Any] = {"date": None, "used": 0}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def questions_remaining() -> int:
    if _usage["date"] != _today():
        _usage.update(date=_today(), used=0)
    return max(0, DAILY_QUESTION_LIMIT - _usage["used"])


def record_question() -> None:
    if _usage["date"] != _today():
        _usage.update(date=_today(), used=0)
    _usage["used"] += 1


# ---------------------------------------------------------------------------
# Event parsing — identical shape to the Streamlit console's trace
# ---------------------------------------------------------------------------

def _shorten(value: Any, limit: int = 400) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)[:limit]


def _looks_masked(payload: Any) -> bool:
    blob = str(payload).upper()
    return "XXX-XX-" in blob or "***" in blob


async def run_agent(message: str, session_id: str) -> dict[str, Any]:
    try:
        existing = await session_service.get_session(
            app_name=APP_NAME, user_id="web", session_id=session_id
        )
    except Exception:
        existing = None

    if existing is None:
        await session_service.create_session(
            app_name=APP_NAME, user_id="web", session_id=session_id
        )

    content = types.Content(role="user", parts=[types.Part(text=message)])

    trace: list[dict[str, Any]] = []
    streamed: list[str] = []
    final_text: str | None = None
    started = time.perf_counter()

    async for event in runner.run_async(
        user_id="web", session_id=session_id, new_message=content
    ):
        author = getattr(event, "author", None) or ROOT_AGENT_NAME
        parts = list(getattr(getattr(event, "content", None), "parts", None) or [])

        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None:
                name = getattr(call, "name", "unknown_tool")
                args = dict(getattr(call, "args", None) or {})
                if name == "transfer_to_agent":
                    trace.append({
                        "kind": "transfer",
                        "from": author,
                        "to": args.get("agent_name", SUB_AGENT_NAME),
                    })
                else:
                    trace.append({"kind": "call", "agent": author, "tool": name, "args": args})

            response = getattr(part, "function_response", None)
            if response is not None:
                name = getattr(response, "name", "unknown_tool")
                if name != "transfer_to_agent":
                    payload = getattr(response, "response", None)
                    trace.append({
                        "kind": "result",
                        "agent": author,
                        "tool": name,
                        "masked": _looks_masked(payload),
                        "payload": _shorten(payload),
                    })

        text = "".join(p.text for p in parts if getattr(p, "text", None))
        is_final = bool(
            callable(getattr(event, "is_final_response", None)) and event.is_final_response()
        )
        if text:
            if getattr(event, "partial", False):
                streamed.append(text)
            elif is_final:
                final_text = text
                trace.append({"kind": "answer", "agent": author})

    answer = final_text or "".join(streamed)
    if not answer:
        answer = "The request completed, but the agent returned no text response."

    agent = ROOT_AGENT_NAME
    for step in trace:
        if step["kind"] == "transfer":
            agent = step.get("to") or SUB_AGENT_NAME

    return {
        "answer": answer,
        "trace": trace,
        "agent": agent,
        "elapsed": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class Ask(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": root_agent.name,
        "questions_remaining": questions_remaining(),
        "daily_limit": DAILY_QUESTION_LIMIT,
    }


@app.post("/ask")
async def ask(body: Ask) -> dict[str, Any]:
    if questions_remaining() <= 0:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit reached. This demo answers {DAILY_QUESTION_LIMIT} "
                "questions a day to keep model costs predictable. Resets at midnight UTC."
            ),
        )

    session_id = body.session_id or str(uuid.uuid4())

    # Counted before the call: a run that fails partway can still cost a turn.
    record_question()

    try:
        result = await run_agent(body.message.strip(), session_id)
    except Exception as error:  # noqa: BLE001 — surfaced as a clean 502
        raise HTTPException(
            status_code=502,
            detail=f"The agent failed to answer ({type(error).__name__}).",
        ) from error

    result["session_id"] = session_id
    result["questions_remaining"] = questions_remaining()
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
