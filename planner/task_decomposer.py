"""
Task Decomposer — converts an Intent into multi-step plans.

The decomposer is deterministic-first:
- Builds a default one-step plan for any intent.
- Detects multi-domain notify flows (e.g., finance analysis + Telegram notification).
"""

from __future__ import annotations

from typing import Iterable

from shared.models import ExecutionPlan, ExecutionStep, IntentOutput


class TaskDecomposer:
    """Deterministic task decomposition with optional multi-domain composition."""

    TELEGRAM_KEYWORDS = (
        "telegram",
        "notifica",
        "notifique",
        "enviar",
        "envia",
        "manda",
        "send",
        "notify",
    )

    def __init__(self, known_capabilities: Iterable[str] | None = None):
        self.known_capabilities = set(known_capabilities or [])

    def update_capabilities(self, capabilities: Iterable[str]) -> None:
        self.known_capabilities = set(capabilities)

    def decompose(self, intent: IntentOutput) -> ExecutionPlan:
        """Decompose intent into 1..N execution steps."""
        if self._should_create_finance_notify_flow(intent):
            communication_action = self._pick_communication_action()
            if communication_action:
                return self._build_finance_notify_plan(intent, communication_action)

        return ExecutionPlan(
            execution_mode="sequential",
            combine_mode="last",
            steps=[
                ExecutionStep(
                    id=1,
                    domain=intent.domain,
                    capability=intent.capability,
                    params=dict(intent.parameters),
                    depends_on=[],
                    required=True,
                    output_key="primary",
                )
            ],
        )

    def _should_create_finance_notify_flow(self, intent: IntentOutput) -> bool:
        if intent.domain != "finance":
            return False
        if intent.capability in ("chat", "list_capabilities"):
            return False
        query = (intent.original_query or "").lower()
        return any(token in query for token in self.TELEGRAM_KEYWORDS)

    def _pick_communication_action(self) -> str | None:
        if "send_telegram_message" in self.known_capabilities:
            return "send_telegram_message"
        if "send_telegram_group_message" in self.known_capabilities:
            return "send_telegram_group_message"
        return None

    def _build_finance_notify_plan(self, intent: IntentOutput, communication_action: str) -> ExecutionPlan:
        analysis_params = dict(intent.parameters)

        # Keep communication parameters out of finance execution payload.
        analysis_params.pop("chat_id", None)
        analysis_params.pop("group_id", None)
        analysis_params.pop("message", None)
        analysis_params.pop("notify", None)

        chat_id = (
            intent.parameters.get("chat_id")
            or intent.parameters.get("group_id")
            or "${ENV:TELEGRAM_DEFAULT_CHAT_ID}"
        )
        message = intent.parameters.get("message") or "${1.explanation}"

        return ExecutionPlan(
            execution_mode="dag",
            combine_mode="report",
            max_concurrency=4,
            steps=[
                ExecutionStep(
                    id=1,
                    domain="finance",
                    capability=intent.capability,
                    params=analysis_params,
                    depends_on=[],
                    required=True,
                    output_key="analysis",
                ),
                ExecutionStep(
                    id=2,
                    domain="communication",
                    capability=communication_action,
                    params={
                        "chat_id": chat_id,
                        "message": message,
                    },
                    depends_on=[1],
                    required=False,
                    output_key="notification",
                ),
            ],
        )
