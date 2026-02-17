"""
Simple end-to-end check for multi-domain decomposition:
- finance step (stock price)
- communication step (telegram notification)
"""

from __future__ import annotations

import asyncio
import os

from main import build_pipeline
from shared.models import ExecutionIntent


async def run_flow() -> int:
    symbol = os.getenv("TEST_SYMBOL", "AAPL").strip().upper()

    (
        _cli,
        conversation,
        _intent_adapter,
        planner,
        engine,
        model_selector,
        mcp_adapter,
    ) = build_pipeline()

    try:
        intent = ExecutionIntent(
            domain="finance",
            capability="get_stock_price",
            confidence=1.0,
            parameters={"symbol": symbol, "notify": True},
            original_query=f"pegue o preco da {symbol} e envie no telegram",
        )

        plan = planner.generate_plan(intent)
        print("Execution mode:", plan.execution_mode)
        print("Combine mode:", plan.combine_mode)
        print("Steps:", [(s.id, s.domain, s.capability, s.depends_on) for s in plan.steps])

        output = await engine.execute_plan(plan, original_intent=intent)
        print("Status:", output.status)
        print("Explanation:", output.explanation)
        print("Result keys:", list(output.result.keys()))
        return 0 if output.status == "success" else 1
    finally:
        planner.close()
        conversation.close()
        model_selector.close()
        mcp_adapter.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_flow()))
