"""
CLI Entry Adapter.

Responsibility:
- Receive user input from terminal
- Normalize to EntryRequest contract
- Forward to pipeline, render response
- NO intent parsing, NO domain logic, NO skill access
"""

import uuid

from shared.models import EntryRequest


class CLIAdapter:
    """Command-line entry adapter."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]

    def read_input(self, raw_input: str) -> EntryRequest:
        """Normalize raw CLI input to EntryRequest."""
        return EntryRequest(
            session_id=self.session_id,
            input_text=raw_input.strip(),
            metadata={"source": "cli"},
        )
