from api.openai_server import (
    ChatCompletionRequest,
    _build_pending_clarification_entry,
    _request_messages_to_turn_history,
    _resolve_pending_clarification_intent,
    _resolve_session_id,
    _should_store_pending_clarification,
)
from shared.models import DomainOutput, IntentOutput


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def test_resolve_session_id_prefers_explicit_chat_id_with_user() -> None:
    request = ChatCompletionRequest(
        user="alice",
        chat_id="chat-123",
        messages=[_message("user", "qual o preco da petro?")],
    )

    assert _resolve_session_id(request) == "alice:chat-123"


def test_resolve_session_id_uses_stable_user_scope_without_chat_id() -> None:
    request_turn_1 = ChatCompletionRequest(
        user="alice",
        messages=[_message("user", "qual o preco da petro?")],
    )
    request_turn_2 = ChatCompletionRequest(
        user="alice",
        messages=[_message("user", "ITUB4.SA")],
    )

    assert _resolve_session_id(request_turn_1) == "user:alice"
    assert _resolve_session_id(request_turn_2) == "user:alice"


def test_resolve_session_id_uses_stable_fingerprint_without_user() -> None:
    request_turn_1 = ChatCompletionRequest(
        messages=[_message("user", "qual o preco da petro?")],
    )
    request_turn_2 = ChatCompletionRequest(
        messages=[
            _message("user", "qual o preco da petro?"),
            _message("assistant", "Encontrei múltiplos resultados para Itaú. Qual você quer?"),
            _message("user", "ITUB4.SA"),
        ],
    )

    assert _resolve_session_id(request_turn_1) == _resolve_session_id(request_turn_2)


def test_resolve_session_id_accepts_nested_metadata_id() -> None:
    request = ChatCompletionRequest(
        metadata={"conversation_id": "conv-42"},
        messages=[_message("user", "oi")],
    )

    assert _resolve_session_id(request) == "chat:conv-42"


def test_pending_ambiguous_symbol_clarification_resumes_previous_intent() -> None:
    original_intent = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=0.99,
        parameters={"symbols": ["PETR4.SA", "VALE3.SA", "ITAU"]},
        original_query="qual o preco da petro vale e itau?",
    )
    clarification = DomainOutput(
        status="clarification",
        result={"candidates": [{"symbol": "ITUB3.SA"}, {"symbol": "ITUB4.SA"}]},
        explanation="Encontrei mais de um ticker para 'ITAU': ITUB3.SA, ITUB4.SA. Qual deles você quer consultar?",
        metadata={"resolution": "ambiguous_symbol"},
    )

    assert _should_store_pending_clarification(clarification) is True

    pending_entry = _build_pending_clarification_entry(original_intent, clarification)
    resumed_intent = _resolve_pending_clarification_intent(pending_entry, "ITUB4.SA")

    assert resumed_intent is not None
    assert resumed_intent.capability == "get_stock_price"
    assert resumed_intent.parameters.get("symbols") == ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]


def test_request_messages_to_turn_history_excludes_current_user_message() -> None:
    request = ChatCompletionRequest(
        messages=[
            _message("user", "qual o preco do itau?"),
            _message("assistant", "Encontrei mais de um ticker para ITAU. Qual deles você quer consultar?"),
            _message("user", "ITUB4.SA"),
        ],
    )

    turns = _request_messages_to_turn_history(request.messages)
    assert turns == [
        {"role": "user", "content": "qual o preco do itau?"},
        {"role": "assistant", "content": "Encontrei mais de um ticker para ITAU. Qual deles você quer consultar?"},
    ]


def test_pending_clarification_is_extracted_from_combined_plan_steps() -> None:
    original_intent = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=0.99,
        parameters={"symbols": ["PETR4.SA", "VALE3.SA", "ITAU"]},
        original_query="qual o valor da petro vale e itau?",
    )
    combined_output = DomainOutput(
        status="clarification",
        explanation="Encontrei mais de um ticker para 'ITAU'.",
        metadata={"execution_mode": "sequential", "combine_mode": "last"},
        result={
            "combined": {},
            "steps": {
                "step_1": {
                    "step_id": 1,
                    "required": True,
                    "status": "success",
                    "explanation": "PETR4.SA at 36.8 BRL",
                    "result": {"symbol": "PETR4.SA", "price": 36.8},
                    "metadata": {},
                },
                "step_2": {
                    "step_id": 2,
                    "required": True,
                    "status": "clarification",
                    "explanation": "Encontrei mais de um ticker para 'ITAU': ITUB3.SA, ITUB4.SA. Qual deles você quer consultar?",
                    "result": {"candidates": [{"symbol": "ITUB3.SA"}, {"symbol": "ITUB4.SA"}]},
                    "metadata": {"resolution": "ambiguous_symbol"},
                },
            },
        },
    )

    assert _should_store_pending_clarification(combined_output) is True

    pending_entry = _build_pending_clarification_entry(original_intent, combined_output)
    resumed_intent = _resolve_pending_clarification_intent(pending_entry, "ITUB4.SA")

    assert resumed_intent is not None
    assert resumed_intent.capability == "get_stock_price"
    assert resumed_intent.parameters.get("symbols") == ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
