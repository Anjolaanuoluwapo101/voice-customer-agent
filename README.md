<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Voice Customer Agent

A customer-facing voice agent for businesses, built with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/). Callers talk to it over the phone (SIP). It answers questions from a knowledge base, manages appointments, checks ticket status and takes messages.

The agent core is business-agnostic. Everything specific to one business lives in a profile under `src/profiles/`: the facts the agent answers from, extra rules and guardrails, speech-recognition keyterms, time zone and currency. Choose one with `BUSINESS_PROFILE`. The default is `generic`, and `aj_hospital` is the first business the agent was tuned for.

To adapt it to another business, copy `src/profiles/generic.py` to `src/profiles/<name>.py`, fill in the fields, and set `BUSINESS_PROFILE=<name>`.

## What it does

- Answers questions about the business, such as services, hours, insurance and location.
- Books, reschedules and cancels appointments.
- Checks the status of an order or ticket.
- Takes messages for staff.
- Recognizes returning callers by their phone number and greets them by first name.

Built-in guardrails: it never shares confidential figures, only discusses the caller's own records, ignores instructions from callers that try to change its rules, and cannot transfer calls. Domain rules, such as medical safety in `aj_hospital`, come from the profile.

## How it works

The agent is a thin voice layer. Business logic lives in [n8n](https://n8n.io/) webhooks, and the agent calls them over HTTP with an `X-API-Key` header.

- **Greeting:** when a call starts, the agent looks up the caller by SIP phone number and speaks a scripted greeting. A known caller is welcomed back by first name.
- **Retries:** read-only webhooks (`lookup-caller`, `ticket-status`, `check-availability`, `ask-knowledge`) are retried once. Writes are never retried, so a slow response can't create a duplicate booking.
- **Warm-up:** at the start of each call the agent pings the n8n `/healthz` endpoint, because free-tier hosting can sleep.
- **Fillers:** if a tool takes longer than 0.8 seconds, the agent speaks a short filler line.
- **Limits:** calls are capped at `MAX_CALL_SECONDS`. If the caller goes silent, the agent asks once and ends the call on the second silence.
- **Call log:** when the call ends, the agent sends the transcript, an LLM-written summary, the outcome, the duration and a follow-up flag to the n8n `log-call` webhook.

### Tools

| Tool | What it does |
|---|---|
| `ask_knowledge` | Searches the business knowledge base |
| `check_availability` | Checks open appointment slots |
| `book_appointment` | Books an appointment |
| `send_details_link` | Sends the caller a details link |
| `manage_booking` | Lists, cancels or reschedules bookings |
| `check_ticket_status` | Looks up an order or ticket |
| `save_message` | Saves a message or lead for staff |
| `end_call` | Ends the call after speech finishes playing |

### Knowledge search

`ask_knowledge` embeds the question with Gemini (`gemini-embedding-001`) and queries a [Qdrant](https://qdrant.tech/) collection. Chunks from files listed in `KNOWLEDGE_BLOCKED_FILES` are filtered out in code. If Gemini or Qdrant isn't configured, it falls back to the n8n `ask-knowledge` webhook.

### Voice pipeline

Models run through [LiveKit Inference](https://docs.livekit.io/agents/models/inference/):

- **STT:** AssemblyAI realtime, with keyterms for staff names and medical terms.
- **LLM:** `google/gemma-4-31b-it` by default. Override it with `LLM_MODEL`.
- **TTS:** Fish Audio S2.1 Pro, in [expressive mode](https://docs.livekit.io/agents/models/tts/expressive/).
- **Turn detection:** [LiveKit Turn Detector](https://docs.livekit.io/agents/logic/turns/turn-detector/), with preemptive generation on.
- **Noise cancellation:** [ai-coustics](https://docs.livekit.io/transport/media/noise-cancellation/) audio enhancement.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It manages Python (3.10 to 3.14) and the dependencies.
- The [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) and a LiveKit Cloud account.
- An n8n instance that exposes the webhooks listed above.
- Optional: a Gemini API key and a Qdrant instance, for direct knowledge search.
- A SIP trunk, to take real phone calls. See the [telephony docs](https://docs.livekit.io/telephony/).

## Configuration

For local runs, put these in `.env.local`, which is gitignored. For a deployed agent, set them as LiveKit secrets.

| Variable | Default | Purpose |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | none | LiveKit Cloud credentials |
| `N8N_BASE_URL` | none | Base URL of the n8n instance. If empty, n8n calls are skipped |
| `BUSINESS_PROFILE` | `generic` | Which profile in `src/profiles/` to load (for example `aj_hospital`) |
| `N8N_API_KEY` | none | Sent as `X-API-Key`. If empty, n8n calls are skipped |
| `BUSINESS_NAME` | the profile's name | Name the agent uses for the business |
| `AGENT_NAME` | `the virtual assistant` | Name the agent uses for itself |
| `GEMINI_API_KEY` | none | Embeddings for direct knowledge search |
| `QDRANT_URL`, `QDRANT_API_KEY` | none | Qdrant connection |
| `QDRANT_COLLECTION` | `business_knowledge` | Qdrant collection to search |
| `KNOWLEDGE_BLOCKED_FILES` | `Internal,Confidential` | Comma-separated source files never returned to callers |
| `TTS_VOICE` | a Fish Audio voice ID | Voice used for speech |
| `LLM_MODEL` | `google/gemma-4-31b-it` | LLM served through LiveKit Inference |
| `MAX_CALL_SECONDS` | `600` | Maximum call length |
| `USER_AWAY_SECONDS` | `20` | Silence before the agent checks the caller is still there |

## Run the agent

Install dependencies:

```console
uv sync
```

Talk to it in your terminal:

```console
lk agent console
```

Connect it to LiveKit Cloud so a frontend or phone call can reach it:

```console
lk agent dev
```

Test one text turn at a time with the [agent debugger](https://docs.livekit.io/testing/debugger/). Each turn shows the reply and the tool calls behind it:

```console
lk agent debugger start
lk agent debugger say "Hi, what are your opening hours?"
lk agent debugger stop
```

In production, run the agent directly:

```console
uv run src/agent.py start
```

The agent registers under the name `my-agent`. That means it is dispatched explicitly, so for phone calls you need a SIP [dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule/) that targets this agent name.

## Deploy

The `Dockerfile` builds a container that runs as a non-root user and downloads model files at build time. To deploy to LiveKit Cloud, see the [deployment guide](https://docs.livekit.io/deploy/agents/). To stream logs from the deployed agent:

```console
lk agent logs
```

## Testing

Simulations run full conversations between a simulated user and the agent, then judge the transcripts:

```console
lk agent simulate text --scenarios scenarios.yaml --yes
```

The `Simulations` workflow in `.github/workflows/simulations.yml` runs the same file on every merge to `main`. It needs `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` as repository secrets.

**Current gaps:** `scenarios.yaml` still contains the three generic starter scenarios, and none of them test booking, tickets, messages or confidentiality. `tests/test_agent.py` is fully commented out, so no pytest tests run. Add scenarios for your business flows before relying on CI to catch regressions.

## Project layout

```
src/agent.py            Agent, tools, and session setup
src/profiles/           Per-business facts, rules and locale
scenarios.yaml          Simulation scenarios
tests/test_agent.py     Pytest tests (currently commented out)
Dockerfile              Container for LiveKit Cloud
livekit.toml            LiveKit Cloud project and agent link
.github/workflows/      Lint and simulation CI
```

## Development

Keep formatting and lint clean:

```console
uv run ruff format
uv run ruff check
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
