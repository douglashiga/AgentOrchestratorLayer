from __future__ import annotations

import re
from typing import Any

from shared.models import DomainOutput


def _round_value(value: Any, decimals: int = 2) -> Any:
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {k: _round_value(v, decimals=decimals) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_value(item, decimals=decimals) for item in value]
    return value


def _round_numbers_in_text(text: str) -> str:
    if not text:
        return ""

    def repl(match: re.Match[str]) -> str:
        try:
            return f"{float(match.group(0)):.2f}"
        except Exception:
            return match.group(0)

    return re.sub(r"-?\d+\.\d{3,}", repl, text)


def _format_price_line(payload: dict[str, Any]) -> str | None:
    symbol = str(payload.get("symbol", "")).strip()
    price = payload.get("price")
    currency = str(payload.get("currency", "")).strip()
    date = str(payload.get("date", "")).strip()
    if symbol and isinstance(price, (int, float)):
        parts = [f"{symbol}: {float(price):.2f}"]
        if currency:
            parts.append(currency)
        if date:
            parts.append(f"({date})")
        return " ".join(parts)
    return None


def _format_dag_result(result: dict[str, Any]) -> str | None:
    steps = result.get("steps")
    if not isinstance(steps, dict):
        return None

    def _ordered_steps(step_map: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        indexed: list[tuple[int, str, dict[str, Any]]] = []
        unindexed: list[tuple[str, dict[str, Any]]] = []
        for key, payload in step_map.items():
            if not isinstance(payload, dict):
                continue
            step_id = payload.get("step_id")
            if isinstance(step_id, int):
                indexed.append((step_id, key, payload))
            else:
                unindexed.append((key, payload))
        indexed.sort(key=lambda item: (item[0], item[1]))
        unindexed.sort(key=lambda item: item[0])
        return [(key, payload) for _, key, payload in indexed] + unindexed

    ordered_steps = _ordered_steps(steps)
    lines: list[str] = []
    for _step_key, step_payload in ordered_steps:
        step_result = step_payload.get("result", {})
        if isinstance(step_result, dict):
            price_line = _format_price_line(step_result)
            if price_line and price_line not in lines:
                lines.append(price_line)

    if not lines:
        primary = steps.get("primary")
        if isinstance(primary, dict):
            primary_explanation = str(primary.get("explanation", "")).strip()
            if primary_explanation:
                lines.append(_round_numbers_in_text(primary_explanation))
        if not lines:
            for _step_key, step_payload in ordered_steps:
                step_explanation = str(step_payload.get("explanation", "")).strip()
                if step_explanation:
                    lines.append(_round_numbers_in_text(step_explanation))
                    break

    telegram_status: str | None = None
    for step_key, step_payload in ordered_steps:
        capability = str(step_payload.get("capability", "")).strip().lower()
        key_l = str(step_key).strip().lower()
        status = str(step_payload.get("status", "")).strip().lower()
        if key_l in {"notification", "notifier"} and status in {"success", "failure"}:
            telegram_status = status
            break
        if capability not in {"send_telegram_message", "send_telegram_group_message"}:
            continue
        if status in {"success", "failure"}:
            telegram_status = status
            break

    if telegram_status == "success":
        lines.append("Mensagem enviada no Telegram.")
    elif telegram_status == "failure":
        lines.append("Falha ao enviar no Telegram.")

    return "\n".join(line for line in lines if line).strip() or None


def format_domain_output(output: DomainOutput, channel: str = "frontend") -> str:
    """
    Normalize domain output for UX channels (frontend/telegram):
    - no raw JSON
    - objective wording
    - rounded numeric values (2 decimals)
    """
    rounded_result = _round_value(output.result, decimals=2)
    explanation = _round_numbers_in_text(str(output.explanation or "").strip())

    if output.status == "success":
        if isinstance(rounded_result, dict):
            dag_line = _format_dag_result(rounded_result)
            if dag_line:
                return dag_line

            price_line = _format_price_line(rounded_result)
            if price_line:
                return price_line

        if explanation:
            if explanation.lower().startswith("executed "):
                return "Concluído com sucesso."
            return explanation
        return "Concluído com sucesso."

    if output.status == "clarification":
        if explanation:
            return explanation
        return "Preciso de mais detalhes para continuar."

    # failure
    err = ""
    if isinstance(output.metadata, dict):
        err = str(output.metadata.get("error", "")).strip()
    err = _round_numbers_in_text(err)
    if err:
        return f"Não consegui concluir: {err}"
    if explanation:
        return f"Não consegui concluir: {explanation}"
    return "Não consegui concluir sua solicitação."
