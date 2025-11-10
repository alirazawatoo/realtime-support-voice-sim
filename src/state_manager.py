from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConversationState:
    """
    Track a single conversational session with the assistant.

    The state stores the minimal information needed to reconstruct
    chat-style messages for the LLM, and enforces a rolling window
    of turns via `max_turns`.
    """

    session_id: str
    persona_name: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    max_turns: int = 8

    def add_turn(self, role: str, text: str) -> None:
        """
        Append a new conversational turn and enforce the `max_turns` limit.
        """
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """
        Convert the stored turns into an LLM-ready messages list.

        The first message is always a `system` message constructed from
        the given `system_prompt`, followed by the historical turns in
        the order they occurred.
        """
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for t in self.turns:
            msgs.append({"role": t["role"], "content": t["text"]})
        return msgs
