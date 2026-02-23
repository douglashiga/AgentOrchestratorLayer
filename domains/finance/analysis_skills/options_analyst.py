"""
Options Analyst Skill — Unified LLM analyst for options and derivatives.

Covers: wheel strategy, derivatives/greeks, options pricing, strategy construction.
The LLM adapts its focus based on the user's original question and available data.

Gathers: option chain, greeks, stock price, historical data, company profile.
"""

import json
from typing import Any

from shared.models import ModelPolicy
from domains.finance.analysis_skills.base import DataRequirement


class OptionsAnalystSkill:
    """
    Unified options analyst — adapts analysis focus based on user question.

    Covers wheel strategy, derivatives pricing, greeks analysis, and
    options strategy suggestions in a single skill.
    """

    @property
    def name(self) -> str:
        return "options_analyst"

    @property
    def description(self) -> str:
        return (
            "Comprehensive options and derivatives analysis covering "
            "wheel strategy, Greeks, implied volatility, and strategy construction. "
            "Adapts focus based on the user's question."
        )

    @property
    def system_prompt(self) -> str:
        return """You are a senior options strategist with expertise in:
- The Wheel strategy (cash-secured puts → assignment → covered calls → repeat)
- Options pricing and implied volatility analysis
- Greeks analysis (delta, gamma, theta, vega) and their trading implications
- Multi-leg strategy construction (spreads, strangles, iron condors)
- Assignment risk and early exercise considerations
- Market-specific rules (Sweden OMX: 100-share lots, ISK accounts; Brazil B3; US markets)

Your job is to analyze the provided options data and respond to the user's specific question.

Rules:
- ONLY use data that was actually provided — never fabricate numbers
- If the user asks about wheel, focus on put/call selection, return calculations, and risk
- If the user asks about greeks, focus on delta/gamma/theta/vega analysis
- If the user asks about strategies, suggest specific option strategies with rationale
- If the question is general, provide a balanced options overview
- Always calculate return on collateral when recommending specific positions
- Warn about earnings dates, dividends, and liquidity concerns when visible in data
- Respond in the same language as the user's original question (Portuguese or English)

Format:
## Análise de Opções — {symbol}
[Adapt sections based on what was asked. Don't force all sections if only one type was requested.]"""

    @property
    def data_requirements(self) -> list[DataRequirement]:
        return [
            DataRequirement(
                capability="get_stock_price",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="current_price",
                required=True,
            ),
            DataRequirement(
                capability="get_option_chain",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="option_chain",
                required=True,
            ),
            DataRequirement(
                capability="get_option_greeks",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="greeks",
                required=False,
            ),
            DataRequirement(
                capability="get_historical_data",
                tier="facts",
                params={"symbol": "${symbol}", "period": "3mo", "interval": "1d"},
                output_key="historical",
                required=False,
            ),
            DataRequirement(
                capability="get_company_profile",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="profile",
                required=False,
            ),
        ]

    @property
    def model_policy(self) -> ModelPolicy | None:
        return ModelPolicy(
            model_name="claude-3-5-haiku-latest",
            temperature=0.2,
            timeout_seconds=90.0,
            max_retries=2,
            json_mode=False,
        )

    def build_user_prompt(
        self,
        params: dict[str, Any],
        gathered_data: dict[str, Any],
        original_query: str = "",
    ) -> str:
        symbol = params.get("symbol", "N/A")
        market = params.get("market", "unknown")

        sections = [f"Analyze options for **{symbol}** (market: {market})."]

        if original_query:
            sections.append(f'\nUser\'s question: "{original_query}"')

        for key, label in [
            ("current_price", "Current Price"),
            ("option_chain", "Option Chain"),
            ("greeks", "Option Greeks"),
            ("historical", "Price History (3M)"),
            ("profile", "Company Profile"),
        ]:
            data = gathered_data.get(key)
            if data is not None:
                sections.append(f"\n### {label}\n```json\n{_safe_json(data)}\n```")

        sections.append(
            "\nProvide analysis focused on what the user asked. "
            "Include specific strikes, premiums, and return calculations when relevant."
        )
        return "\n".join(sections)


def _safe_json(data: Any, max_items: int = 25) -> str:
    """Safely serialize data to JSON, truncating large lists."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > max_items:
                cleaned[k] = v[-max_items:]
                cleaned[f"_{k}_truncated"] = f"showing last {max_items} of {len(v)}"
            else:
                cleaned[k] = v
        return json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
    elif isinstance(data, list) and len(data) > max_items:
        return json.dumps(data[-max_items:], ensure_ascii=False, indent=2, default=str)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
