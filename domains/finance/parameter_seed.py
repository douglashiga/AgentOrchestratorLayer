"""
Parameter Seed — Load known parameter mappings into ParameterResolverDB.

Reads alias dicts from server.py and config.py and seeds the database.
Runs on startup. Idempotent (uses UPSERT via bulk_seed).
"""

import logging
from typing import Any

from domains.finance.parameter_resolver_db import ParameterResolverDB

logger = logging.getLogger(__name__)


def _collect_aliases_from_metadata_overrides() -> dict[str, dict[str, str]]:
    """
    Import alias dicts from server.py metadata definitions.
    Returns {parameter_name: {input_value: resolved_value}}.
    """
    from domains.finance.server import (
        MARKET_ALIASES,
        RANKING_PERIOD_ALIASES,
        HISTORICAL_PERIOD_ALIASES,
        INTERVAL_ALIASES,
        SIGNAL_TYPE_ALIASES,
        SYMBOL_ALIASES,
        METADATA_OVERRIDES,
    )

    aliases: dict[str, dict[str, str]] = {}

    # Market aliases (combined from config + server)
    aliases["market"] = dict(MARKET_ALIASES)

    # Period aliases for ranking capabilities (get_top_gainers, get_top_losers, etc.)
    aliases["period"] = dict(RANKING_PERIOD_ALIASES)

    # Period aliases for historical data (different valid values)
    aliases["period:historical"] = dict(HISTORICAL_PERIOD_ALIASES)

    # Interval aliases
    aliases["interval"] = dict(INTERVAL_ALIASES)

    # Signal type aliases
    aliases["signal_type"] = dict(SIGNAL_TYPE_ALIASES)

    # Symbol aliases (also seeded for deterministic DB lookup)
    aliases["symbol"] = dict(SYMBOL_ALIASES)

    # Extract additional aliases from parameter_specs in METADATA_OVERRIDES
    for _capability, overrides in METADATA_OVERRIDES.items():
        parameter_specs = overrides.get("parameter_specs")
        if not isinstance(parameter_specs, dict):
            continue
        for param_name, spec in parameter_specs.items():
            if not isinstance(spec, dict):
                continue
            spec_aliases = spec.get("aliases")
            if not isinstance(spec_aliases, dict):
                continue
            # Skip params we already handle directly
            if param_name in ("symbol", "symbols"):
                continue
            # Merge into existing or create new
            key = param_name
            if key not in aliases:
                aliases[key] = {}
            for alias_input, alias_resolved in spec_aliases.items():
                if isinstance(alias_input, str) and isinstance(alias_resolved, str):
                    aliases[key][alias_input] = alias_resolved

    return aliases


def seed_parameter_database(db: ParameterResolverDB) -> dict[str, int]:
    """
    Seed the parameter resolver database with all known mappings.

    Returns dict of parameter_name -> count of seeded mappings.
    Idempotent — safe to call on every startup.
    """
    all_aliases = _collect_aliases_from_metadata_overrides()

    results: dict[str, int] = {}
    total = 0

    for parameter_name, mappings in all_aliases.items():
        count = db.bulk_seed(parameter_name, mappings, source="seed")
        results[parameter_name] = count
        total += count

    logger.info(
        "Parameter seed complete: %d total mappings across %d parameters",
        total,
        len(results),
    )
    return results
