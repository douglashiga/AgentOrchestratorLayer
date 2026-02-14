from shared.safe_eval import safe_eval_bool, safe_eval_expression


def test_safe_eval_bool_supports_attribute_and_comparisons() -> None:
    ctx = {
        "resolve": {
            "confidence": 0.8,
            "candidates_count": 2,
        }
    }
    assert safe_eval_bool("resolve.confidence < 0.85 or resolve.candidates_count != 1", ctx) is True


def test_safe_eval_rejects_unsafe_calls() -> None:
    expr = "__import__('os').system('echo hacked')"
    assert safe_eval_bool(expr, {}) is False


def test_safe_eval_expression_supports_len() -> None:
    ctx = {"candidates": ["A", "B", "C"]}
    value = safe_eval_expression("len(candidates)", ctx)
    assert value == 3
