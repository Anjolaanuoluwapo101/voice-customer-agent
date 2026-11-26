"""Business profiles: everything in the agent that is specific to one business.

The agent core in `agent.py` is business-agnostic. A profile supplies the facts the agent
answers from, extra rules, speech-recognition keyterms and locale details.

Select a profile with the `BUSINESS_PROFILE` environment variable (default: `generic`).
To add one, create `src/profiles/<name>.py` that defines `PROFILE = BusinessProfile(...)`.
"""

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusinessProfile:
    # Name the agent uses for the business. `BUSINESS_NAME` overrides it.
    name: str = "our business"
    # What kind of business this is, used in the opening line of the prompt.
    description: str = "a business"
    # Facts the agent answers from directly, written as a bulleted list.
    facts: str = ""
    # Extra capability rules, appended to "What you can do" (one bullet per line).
    rules: str = ""
    # Extra guardrails specific to the domain (for example medical or legal safety).
    guardrails: str = ""
    # Words speech-to-text often gets wrong: staff names, product names, jargon, ID prefixes.
    keyterms: tuple[str, ...] = field(default_factory=tuple)
    # Local time, as a fixed UTC offset in hours, and how to refer to it aloud.
    utc_offset_hours: float = 0
    timezone_label: str = "local time"
    # Currency word the agent says after a fee, for example "naira" or "dollars".
    currency: str = ""
    # Said after a successful booking, for example "ask them to arrive fifteen minutes early".
    arrival_note: str = ""


def load_profile(name: str) -> BusinessProfile:
    """Load `profiles/<name>.py`. Falls back to the generic profile if it can't be found."""
    safe = (
        "".join(c for c in name.strip().lower() if c.isalnum() or c == "_") or "generic"
    )
    try:
        return importlib.import_module(f"profiles.{safe}").PROFILE
    except (ImportError, AttributeError):
        from profiles.generic import PROFILE

        return PROFILE
