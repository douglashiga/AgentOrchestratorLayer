"""
Finance Calculators — Local deterministic math functions.

Each calculator is an async function: (params, DataFetcher) → dict
Calculators can fetch data from MCP via DataFetcher when needed.
"""

from domains.finance.calculators.options import register_options_calculators
from domains.finance.calculators.finance import register_finance_calculators
from domains.finance.calculators.portfolio import register_portfolio_calculators

__all__ = [
    "register_options_calculators",
    "register_finance_calculators",
    "register_portfolio_calculators",
]
