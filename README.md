# Banking Conversational AI Agent 

A multi-agent conversational banking assistant built on the **Google Agent Development Kit (ADK)**.
A routing agent answers balance questions itself and hands customer questions to a specialist
sub-agent. The tool that reads customer records masks the SSN before returning it, so the model
never receives the full number under any phrasing of the question.

Live demo: **[ideaiskey.com/bankconsole.html](https://www.ideaiskey.com/bankconsole.html)**

![The Banking Agent Console answering a customer lookup: the flow diagram shows banking_agent
delegating to customer_agent, which calls get_customer_info, reads banking.db and passes the row
through mask_ssn(); the answer reports only that the SSN ends in 1010.](docs/console-v2.png)

The diagram is not decoration, it is drawn from the actual ADK events for the turn you just ran.
Green is the agent that answered, blue the tool it called, amber the masking step, and the grey
`get_account_balance` box shows the branch that was *not* taken.

---

## How a question flows

**"What is the balance for ACC-100?"**

```
browser  →  POST /ask  →  FastAPI (Render)
                       →  banking_agent            ← Gemini
                       →  get_account_balance()      in-process lookup
                       →  answer                   ← Gemini
```

**"Find information for customer CUST-104"**

```
browser  →  POST /ask  →  FastAPI (Render)
                       →  banking_agent            ← Gemini
                       →  transfer_to_agent
                       →  customer_agent           ← Gemini
                       →  get_customer_info()
                       →  sqlite3 → banking.db → mask_ssn()
                       →  answer
```

Each step marked *Gemini* is a paid model call. A balance question costs about two; a customer
question costs about four, because delegating gives the sub-agent its own turn. That is why the
API caps questions per day rather than per token.

---

## Layout

```
banking_agent/
├── agent.py                       root agent: routes, owns get_account_balance
├── get_account_balance.py         balance tool
├── customer_agent/
│   ├── agent.py                   specialist: customer profiles
│   └── get_customer_info.py       customer tool + mask_ssn()
├── banking.db                     SQLite, 10 seeded customers
├── setup_database.py              recreates banking.db from scratch
├── app.py                         Streamlit console (local UI)
└── .env                           GOOGLE_API_KEY — never committed

api/
├── main.py                        FastAPI wrapper: POST /ask, GET /health
└── requirements.txt

render.yaml                        Render blueprint for the API
bankconsole.html                   browser console (also deployed to ideaiskey.com)
index.html                         portfolio page listing the prototype
```

---

## The data

Ten customers, `CUST-100` … `CUST-109`, in the `customers` table:

| column | example |
|---|---|
| customer_id | `CUST-104` |
| name, age | David Wilson, 47 |
| city, state | Chicago, IL |
| segment | Premier / Standard |
| risk_level | LOW / MEDIUM / HIGH |
| ssn | `111-22-1005` — a sequential placeholder, not a real number |

Balances live in a dict in `get_account_balance.py`, not in the database:
`ACC-100` $5,200 · `ACC-200` $18,500 · `ACC-300` $750.

> **Known gap.** The web console's help bubble lists ten accounts because its offline simulation
> has more data than the live agent. Against the real API only those three resolve; anything else
> returns `NOT_FOUND`. Fix by expanding the dict, or by moving balances into `banking.db`.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-adk streamlit python-dotenv
echo "GOOGLE_API_KEY=your-key" > banking_agent/.env

streamlit run banking_agent/app.py
```

The Streamlit console shows the agent flow diagram, a live execution trace of every delegation
and tool call, and a daily question budget. To rebuild the database: `python banking_agent/setup_database.py`.

---

## The API

`api/main.py` wraps the agent unchanged and exposes two routes.

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
```

**`GET /health`** — liveness plus the remaining budget. Free; makes no model call.

```json
{"status":"ok","agent":"banking_agent","questions_remaining":5,"daily_limit":5}
```

**`POST /ask`** — `{"message": "...", "session_id": "optional"}` returns the answer and the trace
of what the agents actually did:

```json
{
  "answer": "David Wilson (CUST-104) is 47, based in Chicago, IL…",
  "trace": [
    {"kind":"transfer","from":"banking_agent","to":"customer_agent"},
    {"kind":"call","agent":"customer_agent","tool":"get_customer_info","args":{"customer_id":"CUST-104"}},
    {"kind":"result","agent":"customer_agent","tool":"get_customer_info","masked":true,"payload":{"ssn":"XXX-XX-1005"}},
    {"kind":"answer","agent":"customer_agent"}
  ],
  "agent": "customer_agent",
  "elapsed": 2.4,
  "questions_remaining": 4
}
```

Over the cap it returns **429** with a plain-language message. A failed agent run returns **502**.

### Environment

| variable | purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini key. Required. Never commit it. |
| `GOOGLE_GENAI_USE_VERTEXAI` | `FALSE` for AI Studio keys |
| `DAILY_QUESTION_LIMIT` | model calls allowed per day, default `5` |
| `ALLOWED_ORIGINS` | comma-separated origins, or `*` |

---

## Deploying

Render reads `render.yaml`: **New → Blueprint → this repo**, then set `GOOGLE_API_KEY` in the
dashboard. It is marked `sync: false` precisely so it never lives in the repo.

`banking.db` **is committed on purpose** — the deployed container has no other copy, and the data
is fake. Add it back to `.gitignore` the day it holds anything real.

Two things to expect on the free tier: the service sleeps after ~15 minutes idle and the next
request takes 30–60s to wake it, and the daily counter lives in memory, so a restart resets it.
Move the counter to a database if the cap needs to be dependable.

---

## The web console

`bankconsole.html` is a single self-contained file with no build step. It works two ways:

- **Live** — posts to the API in `DEFAULT_API_BASE` and renders the real trace
- **Simulated** — with no API configured, or when one can't be reached, it plays the same
  sequence from built-in data and labels itself so no one mistakes it for a live run

Override the endpoint per visit with `?api=https://…`, or force the simulation with `?api=`.

If the page is served from a site with a Content-Security-Policy, the API's origin must be listed
in `connect-src` or the browser blocks the call — it fails instantly and looks like the API is
down. On ideaiskey.com that lives in `netlify.toml`.

---

## Notes on the masking

`mask_ssn()` in `customer_agent/get_customer_info.py` reduces the SSN to `XXX-XX-1005` before the
tool returns. The guarantee is structural rather than instructional: a prompt saying "never reveal
the SSN" can be talked around, but a value that was never sent to the model cannot be recalled.
Ask the console for a full SSN and it declines while quoting the masked value it actually holds.

The stronger version of this puts masking inside the database — a view or a `SECURITY DEFINER`
function — so the full value never crosses the network at all. That matters when the data is real.
Here it is an in-process function over placeholder numbers.

**No real customer data is in this repository.** Every name, address, and SSN is invented, and the
SSNs are sequential placeholders (`111-22-1001` upward) that are obviously not genuine.
