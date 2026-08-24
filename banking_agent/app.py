"""
Banking Conversational AI — Streamlit front end for a Google ADK multi-agent app.

UX notes
--------
* Light "banking console" theme with a generous type scale.
* The agent flow diagram lives in its own bordered card, visually separate
  from the conversation panel beside it.
* The flow diagram is driven by *real* ADK events (function calls, function
  responses, agent transfers), not keyword guessing.
* Lookups are conversational: typing a bare reference ("acc100", "cust 100")
  in the chat expands into a full question; bare digits trigger an inline
  "account or customer?" prompt instead of a separate form.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st
from dotenv import load_dotenv

# Environment must be loaded before the ADK agent is imported.
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from banking_agent.agent import root_agent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "banking_conversational_ai"
USER_ID = "demo_user"

# Brand mark — ships in ./assets next to this file. Optional: the page still
# renders (unmarked) if the asset is missing.
ASSET_DIR = Path(__file__).parent / "assets"
BRAND_MARK_PATH = ASSET_DIR / "brain_idea.png"

ROOT_AGENT_NAME = "banking_agent"
SUB_AGENT_NAME = "customer_agent"

AGENT_TOOLS = {
    ROOT_AGENT_NAME: ["get_account_balance"],
    SUB_AGENT_NAME: ["get_customer_info"],
}

# Lookup kinds: label -> (prefix, example, prompt template)
LOOKUP_KINDS = {
    "Account": ("ACC", "ACC-100", "What is the balance for {id}?"),
    "Customer": ("CUST", "CUST-100", "Find information for customer {id}."),
}

EXAMPLE_PROMPTS = [
    ("Check a balance", "What is the balance for ACC-100?"),
    ("Look up a customer", "Find information for customer CUST-100."),
    ("Multi-step", "Show me the profile and balance for CUST-100."),
    ("Guardrail test", "What is the full SSN for customer CUST-100?"),
]

# Palette — kept in Python so the diagram and the CSS never drift apart.
CANVAS = "#f4f7fb"
SURFACE = "#ffffff"
PANEL_TINT = "#eef3f9"
LINE = "#d8e2ec"
MUTED = "#5c6c7e"
TEXT = "#0f1d2d"
ACCENT = "#059669"
ACCENT_DEEP = "#047857"
ACCENT_WASH = "#ecfdf5"
TOOL = "#0284c7"
TOOL_WASH = "#eff8ff"
IDLE_FILL = "#f1f5f9"
IDLE_LINE = "#c9d6e2"


st.set_page_config(
    page_title="Banking Conversational AI",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --canvas: {CANVAS};
                --surface: {SURFACE};
                --panel-tint: {PANEL_TINT};
                --line: {LINE};
                --muted: {MUTED};
                --text: {TEXT};
                --accent: {ACCENT};
                --accent-deep: {ACCENT_DEEP};
                --accent-wash: {ACCENT_WASH};
                --tool: {TOOL};
                --tool-wash: {TOOL_WASH};
                --radius: 18px;
                --shadow: 0 1px 2px rgba(15,29,45,0.05), 0 8px 24px rgba(15,29,45,0.06);
            }}

            html, body, [class*="css"], .stApp {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}

            /* Bigger baseline — every rem below scales off this. */
            html {{ font-size: 18px; }}

            .stApp {{
                background:
                    radial-gradient(900px 520px at 10% -10%, rgba(5,150,105,0.09), transparent 60%),
                    radial-gradient(760px 460px at 92% -4%, rgba(2,132,199,0.08), transparent 58%),
                    var(--canvas);
                color: var(--text);
            }}

            .block-container {{
                padding-top: 1.7rem;
                padding-bottom: 6.5rem;
                max-width: 1560px;
            }}

            h1, h2, h3, h4, h5, h6, p, span, label, li, div {{ color: var(--text); }}

            code, kbd, .mono {{
                font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
            }}

            /* ---------- hero ---------- */

            .hero {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1.6rem;
                flex-wrap: wrap;
                padding: 0 0 1.15rem 0;
                border-bottom: 1px solid var(--line);
                margin-bottom: 1.5rem;
            }}

            .hero-eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.13em;
                text-transform: uppercase;
                color: var(--accent-deep);
                margin-bottom: 0.6rem;
            }}

            .hero-title {{
                font-size: 2.35rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                line-height: 1.1;
                margin: 0 0 0.4rem 0;
                color: #08131f;
            }}

            .hero-subtitle {{
                color: var(--muted);
                font-size: 1.02rem;
                max-width: 64ch;
                line-height: 1.55;
            }}

            /* ---------- pills ---------- */

            .pill-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}

            .pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.34rem 0.78rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 600;
                border: 1px solid var(--line);
                background: var(--surface);
                color: var(--muted);
                white-space: nowrap;
            }}

            .pill-live {{
                border-color: #a7e8cd;
                background: var(--accent-wash);
                color: var(--accent-deep);
            }}

            .pill-tool {{
                border-color: #b6ddf5;
                background: var(--tool-wash);
                color: #0369a1;
            }}

            .dot {{
                width: 8px; height: 8px; border-radius: 50%;
                background: var(--accent);
                animation: pulse 2.1s infinite;
            }}

            @keyframes pulse {{
                0%   {{ box-shadow: 0 0 0 0 rgba(5,150,105,0.45); }}
                70%  {{ box-shadow: 0 0 0 8px rgba(5,150,105,0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(5,150,105,0); }}
            }}

            /* ---------- section headings ---------- */

            .panel-head {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.8rem;
                margin-bottom: 0.75rem;
            }}

            .panel-title {{
                font-size: 0.86rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #3d5064;
            }}

            .panel-note {{ font-size: 0.82rem; color: var(--muted); }}

            /* ---------- bordered containers (st.container(border=True)) ---------- */

            div[data-testid="stVerticalBlockBorderWrapper"] {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}

            /* The diagram card gets a tinted, cooler surface so it reads as a
               separate artifact from the conversation beside it. */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.mark-diagram) {{
                background: linear-gradient(180deg, var(--panel-tint), #ffffff 62%);
                border-color: #cfdcea;
            }}

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.mark-chat) {{
                background: var(--surface);
            }}

            .diagram-frame {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.5rem 0.25rem 0.1rem 0.25rem;
                margin-bottom: 0.6rem;
            }}

            /* ---------- metric tiles ---------- */

            .tile {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 15px;
                padding: 0.75rem 0.9rem;
                box-shadow: var(--shadow);
                height: 100%;
            }}

            .tile-label {{
                font-size: 0.74rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--muted);
                font-weight: 650;
            }}

            .tile-value {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.28rem;
                font-weight: 600;
                margin-top: 0.2rem;
                line-height: 1.2;
            }}

            .tile-value.accent {{ color: var(--accent-deep); }}

            /* ---------- trace timeline ---------- */

            .trace {{ position: relative; padding-left: 1.15rem; }}

            .trace::before {{
                content: "";
                position: absolute;
                left: 4px; top: 8px; bottom: 8px;
                width: 2px;
                background: linear-gradient(180deg, var(--accent), rgba(2,132,199,0.4), transparent);
                border-radius: 2px;
            }}

            .trace-item {{ position: relative; padding: 0.3rem 0 0.5rem 0; }}

            .trace-item::before {{
                content: "";
                position: absolute;
                left: -1.15rem; top: 0.66rem;
                width: 10px; height: 10px;
                border-radius: 50%;
                background: var(--surface);
                border: 2px solid var(--accent);
            }}

            .trace-item.tool::before {{ border-color: var(--tool); }}

            .trace-name {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                font-weight: 500;
            }}

            .trace-meta {{ font-size: 0.8rem; color: var(--muted); }}

            .trace-args {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                color: #0b5c8a;
                background: var(--tool-wash);
                border: 1px solid #cbe6f8;
                border-radius: 9px;
                padding: 0.24rem 0.5rem;
                margin-top: 0.3rem;
                display: inline-block;
                word-break: break-word;
            }}

            /* interface-level clarification (not the model speaking) */
            .clarify {{
                border: 1px solid #b6ddf5;
                background: var(--tool-wash);
                border-radius: 14px;
                padding: 0.7rem 0.9rem;
                margin: 0.2rem 0 0.55rem 0;
            }}

            .clarify-q {{ font-size: 0.96rem; }}

            .empty-state {{
                border: 1px dashed var(--line);
                border-radius: 14px;
                padding: 1.5rem 1.2rem;
                text-align: center;
                color: var(--muted);
                font-size: 0.97rem;
                background: #fbfdff;
            }}

            .empty-state b {{ color: var(--text); }}

            /* ---------- chat ---------- */

            [data-testid="stChatMessage"] {{
                background: #f8fafc;
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.9rem 1.05rem;
                margin-bottom: 0.75rem;
            }}

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
                background: var(--accent-wash);
                border-color: #b6e7d3;
            }}

            [data-testid="stChatMessage"] p,
            [data-testid="stChatMessage"] li {{
                font-size: 1.02rem;
                line-height: 1.62;
            }}

            [data-testid="stChatInput"] {{
                border: 1px solid var(--line);
                border-radius: 15px;
                background: var(--surface);
                box-shadow: var(--shadow);
            }}

            [data-testid="stChatInput"] textarea {{ font-size: 1rem !important; }}

            [data-testid="stBottomBlockContainer"] {{ background: transparent; }}

            .turn-meta {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.78rem;
                color: var(--muted);
                margin-top: 0.45rem;
            }}

            /* ---------- inputs ---------- */

            [data-testid="stTextInput"] input {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 1rem;
                border-radius: 11px;
                border: 1px solid var(--line);
                background: var(--surface);
            }}

            [data-testid="stTextInput"] input:focus {{
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(5,150,105,0.14);
            }}

            [data-testid="stWidgetLabel"] p {{
                font-size: 0.86rem;
                font-weight: 600;
                color: #3d5064;
            }}

            [role="radiogroup"] label p {{ font-size: 0.94rem; }}

            /* ---------- widgets ---------- */

            [data-testid="stSidebar"] {{
                background: #ffffff;
                border-right: 1px solid var(--line);
            }}

            [data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.94rem; }}

            .stButton > button {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 12px;
                font-size: 0.92rem;
                font-weight: 550;
                padding: 0.48rem 0.8rem;
                transition: border-color .15s ease, background .15s ease, transform .1s ease;
            }}

            .stButton > button:hover {{
                border-color: var(--accent);
                background: var(--accent-wash);
                color: var(--accent-deep);
                transform: translateY(-1px);
            }}

            .stButton > button[kind="primary"] {{
                background: var(--accent);
                border-color: var(--accent-deep);
                color: #ffffff;
            }}

            .stButton > button[kind="primary"]:hover {{
                background: var(--accent-deep);
                color: #ffffff;
            }}

            .stDownloadButton > button {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 12px;
                font-size: 0.92rem;
            }}

            [data-testid="stExpander"] {{
                border: 1px solid var(--line);
                border-radius: 14px;
                background: #fbfdff;
            }}

            [data-testid="stExpander"] summary {{ font-size: 0.92rem; color: #3d5064; }}

            [data-testid="stAlert"] {{ border-radius: 14px; font-size: 0.96rem; }}

            [data-testid="stCaptionContainer"] p {{ font-size: 0.85rem; color: var(--muted); }}

            div[data-testid="stGraphVizChart"] {{ display: flex; justify-content: center; }}

            hr {{ border-color: var(--line); }}

            #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


# ---------------------------------------------------------------------------
# Brand mark
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def image_data_uri(path_str: str) -> Optional[str]:
    """Inline an image so it can be used inside raw HTML/CSS.

    Streamlit can't serve arbitrary local files to the browser, so anything
    referenced from injected markup has to travel as a data URI.
    """
    path = Path(path_str)
    if not path.is_file():
        return None

    suffix = path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else f"image/{suffix.lstrip('.') or 'png'}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


MARK_URI = image_data_uri(str(BRAND_MARK_PATH))


def inject_mark_styles(uri: Optional[str]) -> None:
    """Styles that depend on the brand mark being present."""
    if not uri:
        return

    st.markdown(
        f"""
        <style>
            .hero-left {{
                display: flex;
                align-items: center;
                gap: 1.15rem;
            }}

            .hero-mark {{
                width: 92px;
                height: auto;
                flex: none;
                filter: drop-shadow(0 4px 10px rgba(15,29,45,0.14));
            }}

            /* The mark also doubles as a watermark behind empty panels.
               The right padding keeps copy from running under it. */
            .empty-state {{
                position: relative;
                overflow: hidden;
                padding-right: 8.2rem;
            }}

            .empty-state::after {{
                content: "";
                position: absolute;
                right: 14px;
                bottom: 10px;
                width: 104px;
                height: 118px;
                background: url("{uri}") no-repeat right bottom / contain;
                opacity: 0.16;
                pointer-events: none;
            }}

            .empty-state > * {{ position: relative; z-index: 1; }}

            @media (max-width: 900px) {{
                .hero-mark {{ width: 62px; }}
                .empty-state {{ padding-right: 1.2rem; }}
                .empty-state::after {{ display: none; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_mark_styles(MARK_URI)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULT_STATE: dict[str, Any] = {
    "messages": [],
    "session_id": None,
    "active_agent": ROOT_AGENT_NAME,
    "active_tools": [],
    "trace": [],
    "last_latency": None,
    "tool_calls_total": 0,
    "pending_prompt": None,
    "pending_origin": None,
    "pending_reference": None,
    "show_payloads": False,
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = ([] if isinstance(value, list) else value)

    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.active_agent = ROOT_AGENT_NAME
    st.session_state.active_tools = []
    st.session_state.trace = []
    st.session_state.last_latency = None
    st.session_state.tool_calls_total = 0
    st.session_state.pending_prompt = None
    st.session_state.pending_origin = None
    st.session_state.pending_reference = None


init_state()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def normalize_reference(kind: str, raw: str) -> tuple[Optional[str], Optional[str]]:
    """Turn loose user input into a canonical ID.

    Accepts ``100``, ``acc100``, ``ACC-100``, ``acc 100`` and returns
    ``ACC-100``. Returns ``(None, error_message)`` when the input can't be
    made sense of.
    """
    prefix, example, _ = LOOKUP_KINDS[kind]
    cleaned = (raw or "").strip().upper().replace(" ", "").replace("_", "")

    if not cleaned:
        return None, f"Enter a {kind.lower()} number, e.g. {example}."

    # Strip any prefix the user typed, correct or not, then re-apply ours.
    # Longest alternatives first — regex alternation is leftmost-wins, so
    # "ACC" ahead of "ACCT" would leave a stray "T" behind.
    body = re.sub(r"^(ACCOUNT|ACCT|ACC|CUSTOMER|CUST)[-#]?", "", cleaned)
    body = body.lstrip("-#")

    if not body:
        return None, f"That looks like a prefix without a number — try {example}."

    if not re.fullmatch(r"[A-Z0-9-]{1,24}", body):
        return None, f"Use letters, digits and dashes only, e.g. {example}."

    return f"{prefix}-{body}", None


def lookup_prompt(kind: str, reference: str) -> str:
    return LOOKUP_KINDS[kind][2].format(id=reference)


# "acc100", "ACC-100", "cust 100", "customer#100" — a bare reference and
# nothing else. Anything with surrounding words is a normal question.
_REFERENCE_PATTERNS = [
    ("Account", re.compile(r"^(?:ACCOUNT|ACCT|ACC)[-#\s]?([A-Z0-9][A-Z0-9-]{0,23})$")),
    ("Customer", re.compile(r"^(?:CUSTOMER|CUST)[-#\s]?([A-Z0-9][A-Z0-9-]{0,23})$")),
]
_BARE_NUMBER = re.compile(r"^[0-9]{1,12}$")


def parse_reference(text: str) -> tuple[Optional[str], Optional[str]]:
    """Classify a chat message that is *only* an account/customer reference.

    Returns ``(kind, body)`` when the type is unambiguous, ``(None, body)``
    when the user typed bare digits and we need to ask which record they
    mean, and ``(None, None)`` for ordinary questions.
    """
    cleaned = (text or "").strip().upper().replace("_", "")
    cleaned = re.sub(r"\s+", " ", cleaned).rstrip("?.!")

    if not cleaned:
        return None, None

    for kind, pattern in _REFERENCE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return kind, match.group(1).strip("-")

    if _BARE_NUMBER.match(cleaned):
        return None, cleaned

    return None, None


# ---------------------------------------------------------------------------
# ADK runtime
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_adk_runtime() -> tuple[Runner, InMemorySessionService]:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    return runner, session_service


@st.cache_resource(show_spinner=False)
def get_event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop for the whole app.

    Using ``asyncio.run`` per turn closes the loop each time, which can tear
    down HTTP clients the ADK keeps alive between calls.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


async def ensure_session(session_service: InMemorySessionService) -> None:
    try:
        existing = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=st.session_state.session_id,
        )
        if existing is not None:
            return
    except Exception:
        pass

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=st.session_state.session_id,
    )


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

def _shorten(value: Any, limit: int = 220) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _looks_masked(payload: Any) -> bool:
    """Heuristic: did the tool layer hand back a masked value?"""
    try:
        blob = json.dumps(payload, default=str)
    except Exception:
        blob = str(payload)
    return "***" in blob or "XXX-XX-" in blob.upper()


async def invoke_adk(
    user_text: str,
    on_delta: Optional[Callable[[str], None]] = None,
    on_step: Optional[Callable[[dict], None]] = None,
) -> tuple[str, list[dict]]:
    runner, session_service = get_adk_runtime()
    await ensure_session(session_service)

    message = types.Content(role="user", parts=[types.Part(text=user_text)])

    trace: list[dict] = []
    streamed: list[str] = []
    final_text: Optional[str] = None

    def record(step: dict) -> None:
        trace.append(step)
        if on_step:
            on_step(step)

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=st.session_state.session_id,
        new_message=message,
    ):
        author = getattr(event, "author", None) or ROOT_AGENT_NAME
        content = getattr(event, "content", None)
        parts = list(getattr(content, "parts", None) or [])

        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None:
                name = getattr(call, "name", "unknown_tool")
                args = dict(getattr(call, "args", None) or {})
                if name == "transfer_to_agent":
                    record(
                        {
                            "kind": "transfer",
                            "agent": author,
                            "target": args.get("agent_name", SUB_AGENT_NAME),
                        }
                    )
                else:
                    record({"kind": "tool_call", "agent": author, "name": name, "args": args})

            response = getattr(part, "function_response", None)
            if response is not None:
                name = getattr(response, "name", "unknown_tool")
                payload = getattr(response, "response", None)
                if name != "transfer_to_agent":
                    record(
                        {
                            "kind": "tool_result",
                            "agent": author,
                            "name": name,
                            "masked": _looks_masked(payload),
                            "preview": _shorten(payload),
                        }
                    )

        text = "".join(p.text for p in parts if getattr(p, "text", None))
        is_final = bool(
            callable(getattr(event, "is_final_response", None)) and event.is_final_response()
        )

        if text:
            if getattr(event, "partial", False):
                streamed.append(text)
                if on_delta:
                    on_delta("".join(streamed))
            elif is_final:
                final_text = text
                record({"kind": "answer", "agent": author})
                if on_delta:
                    on_delta(text)

    answer = final_text or "".join(streamed)
    if not answer:
        answer = "The request completed, but the agent returned no text response."

    return answer, trace


def run_agent(
    user_text: str,
    on_delta: Optional[Callable[[str], None]] = None,
    on_step: Optional[Callable[[dict], None]] = None,
) -> tuple[str, list[dict], float]:
    started = time.perf_counter()
    loop = get_event_loop()
    asyncio.set_event_loop(loop)
    answer, trace = loop.run_until_complete(invoke_adk(user_text, on_delta, on_step))
    return answer, trace, time.perf_counter() - started


def summarize_trace(trace: list[dict]) -> tuple[str, list[str]]:
    """Derive the highlighted agent + tools from what actually happened."""
    agent = ROOT_AGENT_NAME
    tools: list[str] = []

    for step in trace:
        if step["kind"] == "transfer":
            agent = step.get("target") or SUB_AGENT_NAME
        elif step["kind"] == "tool_call":
            tools.append(step["name"])
            for owner, owned in AGENT_TOOLS.items():
                if step["name"] in owned:
                    agent = owner

    return agent, tools


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_flow(active_agent: str, active_tools: list[str]) -> None:
    def agent_style(name: str) -> tuple[str, str, str]:
        if name == active_agent:
            return ACCENT, "#ffffff", ACCENT_DEEP
        return IDLE_FILL, MUTED, IDLE_LINE

    def tool_style(name: str) -> tuple[str, str, str]:
        if name in active_tools:
            return TOOL, "#ffffff", "#0369a1"
        return IDLE_FILL, MUTED, IDLE_LINE

    bank_fill, bank_font, bank_line = agent_style(ROOT_AGENT_NAME)
    cust_fill, cust_font, cust_line = agent_style(SUB_AGENT_NAME)
    bal_fill, bal_font, bal_line = tool_style("get_account_balance")
    info_fill, info_font, info_line = tool_style("get_customer_info")

    delegated = active_agent == SUB_AGENT_NAME
    edge_delegate = ACCENT if delegated else IDLE_LINE
    edge_bal = TOOL if "get_account_balance" in active_tools else IDLE_LINE
    edge_info = TOOL if "get_customer_info" in active_tools else IDLE_LINE

    dot = f"""
    digraph flow {{
        rankdir=TB;
        bgcolor="transparent";
        graph [pad="0.2", nodesep="0.5", ranksep="0.6"];
        node [shape=box, style="rounded,filled", fontname="Helvetica",
              fontsize=13, penwidth=1.7, margin="0.22,0.15", height=0.4];
        edge [penwidth=1.7, arrowsize=0.7, fontname="Helvetica", fontsize=11];

        user [label="You", fillcolor="{SURFACE}", fontcolor="{MUTED}", color="{IDLE_LINE}"];

        {ROOT_AGENT_NAME} [label="◆  banking_agent",
              fillcolor="{bank_fill}", fontcolor="{bank_font}", color="{bank_line}"];

        {SUB_AGENT_NAME} [label="◆  customer_agent",
              fillcolor="{cust_fill}", fontcolor="{cust_font}", color="{cust_line}"];

        get_account_balance [label="⌁  get_account_balance",
              fillcolor="{bal_fill}", fontcolor="{bal_font}", color="{bal_line}",
              style="rounded,filled,dashed"];

        get_customer_info [label="⌁  get_customer_info",
              fillcolor="{info_fill}", fontcolor="{info_font}", color="{info_line}",
              style="rounded,filled,dashed"];

        user -> {ROOT_AGENT_NAME} [color="{IDLE_LINE}"];
        {ROOT_AGENT_NAME} -> {SUB_AGENT_NAME} [color="{edge_delegate}", label="delegate", fontcolor="{MUTED}"];
        {ROOT_AGENT_NAME} -> get_account_balance [color="{edge_bal}", style=dashed];
        {SUB_AGENT_NAME} -> get_customer_info [color="{edge_info}", style=dashed];

        {{ rank=same; {SUB_AGENT_NAME}; get_account_balance; }}
    }}
    """

    st.graphviz_chart(dot, use_container_width=True)


def render_trace(trace: list[dict], show_payloads: bool) -> None:
    if not trace:
        st.markdown(
            '<div class="empty-state">No activity yet — send a message or run a '
            'lookup to watch the agents and tools light up.</div>',
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []

    for step in trace:
        kind = step["kind"]

        if kind == "transfer":
            rows.append(
                f'<div class="trace-item">'
                f'<div class="trace-name">{step["agent"]} → {step["target"]}</div>'
                f'<div class="trace-meta">delegated to sub-agent</div></div>'
            )
        elif kind == "tool_call":
            args = _shorten(step.get("args") or {}, 160)
            args_html = f'<div class="trace-args">{args}</div>' if step.get("args") else ""
            rows.append(
                f'<div class="trace-item tool">'
                f'<div class="trace-name">{step["name"]}()</div>'
                f'<div class="trace-meta">called by {step["agent"]}</div>'
                f"{args_html}</div>"
            )
        elif kind == "tool_result":
            badge = (
                ' <span class="pill pill-live" style="padding:0.1rem 0.5rem;'
                'font-size:0.74rem;">PII masked</span>'
                if step.get("masked")
                else ""
            )
            payload = (
                f'<div class="trace-args">{step.get("preview", "")}</div>'
                if show_payloads
                else ""
            )
            rows.append(
                f'<div class="trace-item tool">'
                f'<div class="trace-name">{step["name"]} → response{badge}</div>'
                f'<div class="trace-meta">returned to {step["agent"]}</div>'
                f"{payload}</div>"
            )
        elif kind == "answer":
            rows.append(
                f'<div class="trace-item">'
                f'<div class="trace-name">final response</div>'
                f'<div class="trace-meta">composed by {step["agent"]}</div></div>'
            )

    st.markdown(f'<div class="trace">{"".join(rows)}</div>', unsafe_allow_html=True)


def tile(label: str, value: str, accent: bool = False) -> str:
    cls = "tile-value accent" if accent else "tile-value"
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="{cls}">{value}</div></div>'
    )


def transcript_markdown() -> str:
    lines = [
        "# Banking Conversational AI — transcript",
        f"_Session `{st.session_state.session_id}` · exported "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for message in st.session_state.messages:
        who = "You" if message["role"] == "user" else "Agent"
        lines.append(f"**{who}** · {message.get('time', '')}")
        lines.append("")
        lines.append(message["content"])
        lines.append("")
    return "\n".join(lines)


def queue_prompt(prompt: str, origin: Optional[str] = None) -> None:
    st.session_state.pending_prompt = prompt
    st.session_state.pending_origin = origin
    st.session_state.pending_reference = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="hero-eyebrow"><span class="dot"></span> ADK Runtime</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Session")
    st.markdown(
        f'<div class="pill mono">{st.session_state.session_id[:8]}…</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    if st.button("＋  New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.download_button(
        "⤓  Export transcript",
        data=transcript_markdown(),
        file_name=f"transcript-{st.session_state.session_id[:8]}.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    st.markdown("---")
    st.markdown("**Display**")
    st.session_state.show_payloads = st.toggle(
        "Show raw tool payloads",
        value=st.session_state.show_payloads,
        help="Print the JSON each tool returned in the trace panel.",
    )

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(
        """
`banking_agent` — general banking requests, calls `get_account_balance`,
delegates customer questions to `customer_agent`.

`customer_agent` — customer profiles, calls `get_customer_info`, and only
ever receives SSN values that the tool layer has already masked.
"""
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

hero_mark = (
    f'<img class="hero-mark" src="{MARK_URI}" alt="Reasoning agent mark">'
    if MARK_URI
    else ""
)

st.markdown(
    f"""
<div class="hero">
  <div class="hero-left">
    {hero_mark}
    <div>
      <div class="hero-eyebrow"><span class="dot"></span> Google ADK · multi-agent</div>
      <div class="hero-title">Banking Conversational AI</div>
      <div class="hero-subtitle">
        A routing agent, a specialist sub-agent, and a tool layer that masks PII before
        the model ever sees it. Every call is traced live.
      </div>
    </div>
  </div>
  <div class="pill-row">
    <span class="pill pill-live"><span class="dot"></span> Runtime connected</span>
    <span class="pill pill-tool">2 agents</span>
    <span class="pill">2 tools</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
latency = (
    f"{st.session_state.last_latency:.2f}s" if st.session_state.last_latency else "—"
)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(tile("Turns", str(turns)), unsafe_allow_html=True)
m2.markdown(tile("Tool calls", str(st.session_state.tool_calls_total)), unsafe_allow_html=True)
m3.markdown(tile("Active agent", st.session_state.active_agent, accent=True), unsafe_allow_html=True)
m4.markdown(tile("Last turn", latency), unsafe_allow_html=True)

st.markdown("")


# ---------------------------------------------------------------------------
# Main layout — diagram card (left) kept distinct from conversation card (right)
# ---------------------------------------------------------------------------

left, right = st.columns([0.95, 1.3], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<span class="mark-diagram"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><span class="panel-title">Agent Flow</span>'
            '<span class="panel-note">live</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="diagram-frame">', unsafe_allow_html=True)
        render_flow(st.session_state.active_agent, st.session_state.active_tools)
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Highlighting reflects the actual ADK events from the last turn — "
            "green for the answering agent, blue for the tools it invoked."
        )

    with st.container(border=True):
        st.markdown('<span class="mark-diagram"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><span class="panel-title">Execution Trace</span></div>',
            unsafe_allow_html=True,
        )
        render_trace(st.session_state.trace, st.session_state.show_payloads)


with right:
    # ---- conversation (account / customer numbers are entered right here) --
    with st.container(border=True):
        st.markdown('<span class="mark-chat"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><span class="panel-title">Conversation</span></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-state">Ask in your own words — or just type an '
                "account or customer number and the assistant takes it from there."
                "</div>",
                unsafe_allow_html=True,
            )
            chips = st.columns(2)
            for index, (label, prompt) in enumerate(EXAMPLE_PROMPTS):
                if chips[index % 2].button(label, key=f"chip-{index}", use_container_width=True):
                    queue_prompt(prompt)
                    st.rerun()

        for message in st.session_state.messages:
            avatar = "🧑" if message["role"] == "user" else "🟢"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                meta_bits = [message.get("time", "")]
                if message.get("origin"):
                    meta_bits.append(f'you typed "{message["origin"]}"')
                if message.get("latency"):
                    meta_bits.append(f"{message['latency']:.2f}s")
                if message.get("agent"):
                    meta_bits.append(message["agent"])
                st.markdown(
                    f'<div class="turn-meta">{"  ·  ".join(b for b in meta_bits if b)}</div>',
                    unsafe_allow_html=True,
                )

                if message.get("trace"):
                    with st.expander(f"Trace · {len(message['trace'])} steps"):
                        render_trace(message["trace"], st.session_state.show_payloads)

        # Bare digits are ambiguous — ask which record, in line with the chat.
        if st.session_state.pending_reference:
            reference = st.session_state.pending_reference
            st.markdown(
                f'<div class="clarify"><div class="clarify-q">Is <b>{reference}</b> an '
                "account number or a customer number?</div></div>",
                unsafe_allow_html=True,
            )
            pick_a, pick_c, pick_x = st.columns([1, 1, 0.7])
            for column, choice in ((pick_a, "Account"), (pick_c, "Customer")):
                if column.button(f"{choice} {reference}", use_container_width=True):
                    canonical, _ = normalize_reference(choice, reference)
                    queue_prompt(lookup_prompt(choice, canonical), origin=reference)
                    st.rerun()
            if pick_x.button("Dismiss", use_container_width=True):
                st.session_state.pending_reference = None
                st.rerun()

        st.caption(
            "Type a question — `ACC-100`, `ACC-200` or `ACC-300`, OR `CUST-100`, or `CUST-101` through `CUST-109`."
        )

        typed = st.chat_input("Ask a question, or enter an account or customer number…")
        submitted_text = typed or st.session_state.pending_prompt
        origin = st.session_state.pending_origin
        st.session_state.pending_prompt = None
        st.session_state.pending_origin = None

        user_prompt: Optional[str] = None

        if submitted_text:
            kind, body = parse_reference(submitted_text)
            if kind:
                # "acc100" -> "What is the balance for ACC-100?"
                reference, error = normalize_reference(kind, body)
                user_prompt = submitted_text if error else lookup_prompt(kind, reference)
                origin = None if error else submitted_text.strip()
            elif body:
                st.session_state.pending_reference = body
                st.rerun()
            else:
                user_prompt = submitted_text

        if user_prompt:
            now = datetime.now().strftime("%H:%M")
            st.session_state.messages.append(
                {"role": "user", "content": user_prompt, "time": now, "origin": origin}
            )

            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_prompt)

            with st.chat_message("assistant", avatar="🟢"):
                status = st.status("Routing to banking_agent…", expanded=False)
                answer_slot = st.empty()

                def on_delta(text: str) -> None:
                    answer_slot.markdown(text + " ▌")

                def on_step(step: dict) -> None:
                    if step["kind"] == "transfer":
                        status.update(label=f"Delegating to {step['target']}…")
                    elif step["kind"] == "tool_call":
                        status.update(label=f"Calling {step['name']}()…")
                    elif step["kind"] == "tool_result":
                        status.update(label=f"{step['name']} responded")
                    elif step["kind"] == "answer":
                        status.update(label="Composing response…")

                try:
                    answer, trace, elapsed = run_agent(user_prompt, on_delta, on_step)
                except Exception as error:  # noqa: BLE001 — surfaced to the user below
                    status.update(label="Request failed", state="error")
                    answer_slot.empty()
                    st.error("The agent run failed before returning a response.")
                    with st.expander("Error detail"):
                        st.code(f"{type(error).__name__}: {error}")
                    st.stop()

                status.update(label=f"Done in {elapsed:.2f}s", state="complete", expanded=False)
                answer_slot.markdown(answer)

            agent, tools = summarize_trace(trace)
            st.session_state.active_agent = agent
            st.session_state.active_tools = tools
            st.session_state.trace = trace
            st.session_state.last_latency = elapsed
            st.session_state.tool_calls_total += sum(
                1 for step in trace if step["kind"] == "tool_call"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "time": datetime.now().strftime("%H:%M"),
                    "latency": elapsed,
                    "agent": agent,
                    "trace": trace,
                }
            )
            st.rerun()