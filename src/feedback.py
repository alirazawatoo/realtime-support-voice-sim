import re
from typing import Dict

from .state_manager import ConversationState


def evaluate(state: ConversationState) -> str:
    user_utts = [t["text"] for t in state.turns if t["role"] == "user"]
    assistant_utts = [t["text"] for t in state.turns if t["role"] == "assistant"]
    greeting = any(re.search(r"\bhello|hi|good\s+(morning|afternoon|evening)\b", u, re.I) for u in user_utts[:2])
    verification = any(re.search(r"name|verify|security|dob|date of birth|account", a, re.I) for a in assistant_utts)
    empathy = any(re.search(r"sorry|understand|i can imagine|that sounds", a, re.I) for a in assistant_utts)
    resolution = any(re.search(r"let's|we can|i will|steps|block|reissue|unlock|transfer", a, re.I) for a in assistant_utts)

    pos = []
    neg = []
    if greeting: pos.append("Greeting present")
    else: neg.append("Missing greeting")
    if verification: pos.append("Verification asked")
    else: neg.append("Verification missing")
    if empathy: pos.append("Empathy detected")
    else: neg.append("Empathy missing")
    if resolution: pos.append("Resolution provided")
    else: neg.append("Resolution unclear")

    lines = ["Post-run evaluation:"]
    for p in pos: lines.append(f"✅ {p}")
    for n in neg: lines.append(f"⚠️ {n}")
    return "\n".join(lines)
