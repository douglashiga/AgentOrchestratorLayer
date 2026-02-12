"""
Observability Layer — Structured Logging & metrics.

Responsibility:
- Log events in a structured JSON format (or human-readable for CLI)
- Track metrics (latency, tokens, specific error types)
- Contextual logging (session_id, trace_id)

This replaces standard logging for domain events.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

# Configure standard logger
logger = logging.getLogger("observability")


class Observability:
    """Structured logger for agent events."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.trace_id = str(uuid.uuid4())

    def log_event(self, event_type: str, payload: dict[str, Any], level: str = "INFO") -> None:
        """Log a structured event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "event": event_type,
            "level": level,
            **payload,
        }
        
        # For CLI output, we might want to keep it clean, but for file/system logs:
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(json.dumps(entry))

    @contextmanager
    def measure(self, operation: str, metadata: dict[str, Any] | None = None):
        """Context manager to measure execution time of an operation."""
        start_time = time.perf_counter()
        meta = metadata or {}
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_event(
                "execution_metric",
                {
                    "operation": operation,
                    "duration_ms": round(duration_ms, 2),
                    "success": success,
                    "error": error,
                    **meta,
                },
            )

    def span(self, trace_id: str) -> "Observability":
        """Create a new logger instance sharing the trace_id (for deep calls)."""
        obs = Observability(self.session_id)
        obs.trace_id = trace_id
        return obs
