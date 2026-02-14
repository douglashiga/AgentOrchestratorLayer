from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.models import DomainOutput
from shared.response_formatter import format_domain_output


@dataclass
class DeliveryPayload:
    kind: str
    content: str
    data: dict[str, Any]


def _extract_primary_payload(result: dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return None
    steps = result.get("steps")
    if isinstance(steps, dict):
        primary = steps.get("primary")
        if isinstance(primary, dict):
            primary_result = primary.get("result")
            if isinstance(primary_result, dict):
                return primary_result
    return result


def _looks_like_url(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _url_kind(url: str) -> str:
    u = url.lower()
    if any(domain in u for domain in ("youtube.com", "youtu.be", "vimeo.com")):
        return "video"
    if any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
        return "image"
    return "link"


def _to_markdown_table(rows: list[dict[str, Any]], max_rows: int = 12) -> str:
    if not rows:
        return ""
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(str(key))
    if not cols:
        return ""
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body: list[str] = []
    for row in rows[:max_rows]:
        values = [str(row.get(col, "")) for col in cols]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def build_delivery_payload(output: DomainOutput) -> DeliveryPayload:
    result = output.result if isinstance(output.result, dict) else {}
    payload = _extract_primary_payload(result)

    if isinstance(payload, dict):
        for key in ("video_url", "youtube_url", "vimeo_url"):
            value = payload.get(key)
            if isinstance(value, str) and _looks_like_url(value):
                return DeliveryPayload(kind="video", content=f"[Abrir vídeo]({value})", data={"url": value})
        for key in ("image_url", "image"):
            value = payload.get(key)
            if isinstance(value, str) and _looks_like_url(value):
                return DeliveryPayload(kind="image", content=f"![imagem]({value})", data={"url": value})
        for key in ("url", "link"):
            value = payload.get(key)
            if isinstance(value, str) and _looks_like_url(value):
                detected = _url_kind(value)
                label = "Abrir vídeo" if detected == "video" else "Abrir link"
                return DeliveryPayload(kind=detected, content=f"[{label}]({value})", data={"url": value})

        for key in ("code", "snippet", "source_code"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return DeliveryPayload(
                    kind="code",
                    content=f"```text\n{value.strip()}\n```",
                    data={"language": "text"},
                )

        for key in ("items", "rows", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                table = _to_markdown_table(value)
                if table:
                    return DeliveryPayload(kind="table", content=table, data={"rows": len(value)})

    formatted_text = format_domain_output(output, channel="frontend")
    return DeliveryPayload(kind="text", content=formatted_text, data={})
