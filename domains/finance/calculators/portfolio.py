"""
Portfolio Calculators — Capital allocation and portfolio math.

Placeholder for future portfolio-level calculations:
- Position sizing
- Risk-weighted allocation
- Rebalancing math
"""

import logging
from typing import Any

from domains.finance.tiers.calculators import CalculatorRegistry, DataFetcher

logger = logging.getLogger(__name__)


def register_portfolio_calculators(registry: CalculatorRegistry) -> None:
    """Register portfolio calculators. Currently empty — placeholder for expansion."""
    # Future calculators:
    # registry.register("calc_position_size", calc_position_size)
    # registry.register("calc_portfolio_allocation", calc_portfolio_allocation)
    logger.info("Portfolio calculators registered (placeholder)")
