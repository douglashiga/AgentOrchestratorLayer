"""
Stock Analyst Skill — Unified LLM analyst for stock analysis.

Covers: technical analysis, fundamental analysis, risk assessment, asset comparison.
The LLM adapts its focus based on the user's original question and available data.

Gathers: historical data, technical indicators, company profile, financials,
         analyst recommendations, current price.
"""

import json
from typing import Any

from shared.models import ModelPolicy
from domains.finance.analysis_skills.base import DataRequirement


class StockAnalystSkill:
    """
    Unified stock analyst — adapts analysis focus based on user question.

    Instead of separate technical/fundamental/risk skills, this single skill
    gathers comprehensive data and lets the LLM decide the analysis focus.
    """

    @property
    def name(self) -> str:
        return "stock_analyst"

    @property
    def description(self) -> str:
        return (
            "Comprehensive stock analysis covering technical indicators, "
            "fundamentals, risk assessment, and asset comparison. "
            "Adapts focus based on the user's question."
        )

    @property
    def system_prompt(self) -> str:
        return """You are a senior financial analyst with expertise in:
- Technical analysis (RSI, MACD, moving averages, support/resistance, chart patterns)
- Fundamental analysis (P/E, P/B, margins, ROE, growth, valuation)
- Risk assessment (volatility, drawdown, sector risk, liquidity)
- Comparative analysis (relative valuation, performance comparison)

Your job is to analyze the provided financial data and respond to the user's specific question.

Rules:
- ONLY use data that was actually provided — never fabricate numbers or prices
- Focus your analysis on what the user actually asked for
- If the user asks about technicals, focus on technical indicators
- If the user asks about fundamentals, focus on valuation and financials
- If the user asks about risk, focus on volatility and downside scenarios
- If the user asks to compare assets, do a comparative analysis
- If the question is general ("tell me about X"), provide a balanced overview
- Use bullet points for clarity
- End with a clear, actionable conclusion
- Respond in the same language as the user's original question (Portuguese or English)

Format:
## Análise — {symbol}
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
                capability="get_historical_data",
                tier="facts",
                params={"symbol": "${symbol}", "period": "6mo", "interval": "1d"},
                output_key="historical",
                required=True,
            ),
            DataRequirement(
                capability="get_technical_analysis",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="technical",
                required=False,
            ),
            DataRequirement(
                capability="get_company_profile",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="profile",
                required=False,
            ),
            DataRequirement(
                capability="get_financial_statements",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="financials",
                required=False,
            ),
            DataRequirement(
                capability="get_analyst_recommendations",
                tier="facts",
                params={"symbol": "${symbol}"},
                output_key="analyst_recs",
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
        symbols = params.get("symbols", [])

        if symbols and len(symbols) > 1:
            target = ", ".join(symbols)
        else:
            target = symbol

        sections = [f"Analyze **{target}** based on the data below."]

        if original_query:
            sections.append(f'\nUser\'s question: "{original_query}"')

        # Add all available data
        for key, label in [
            ("current_price", "Current Price"),
            ("historical", "Historical Price Data (6M daily)"),
            ("technical", "Technical Indicators"),
            ("profile", "Company Profile"),
            ("financials", "Financial Statements"),
            ("analyst_recs", "Analyst Recommendations"),
        ]:
            data = gathered_data.get(key)
            if data is not None:
                sections.append(f"\n### {label}\n```json\n{_safe_json(data)}\n```")

        sections.append(
            "\nProvide analysis focused on what the user asked. "
            "Be specific with numbers from the data."
        )
        return "\n".join(sections)


def _safe_json(data: Any, max_items: int = 30) -> str:
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
