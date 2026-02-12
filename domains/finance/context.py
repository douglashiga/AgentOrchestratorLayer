"""
Finance Domain — Context Resolver.

Responsibility:
- Generate deterministic DomainContext from ticker/symbol
- Detect market from symbol suffix
- Provide complete market profiles (exchange, tax, trading rules)
- Normalize symbol format for correct MCP/IB queries

Prohibitions:
- No skill access
- No LLM
- No strategy calculations

Each market profile is 100% deterministic and immutable.
"""

import logging

from shared.models import DomainContext

logger = logging.getLogger(__name__)

# ─── Market Profiles ───────────────────────────────────────────
# All data is fixed and deterministic per market.

MARKET_US: dict = {
    "market": "US",
    "country": "United States",
    "currency": "USD",
    "currency_symbol": "$",
    "exchange": "NASDAQ",
    "exchange_suffix": "",
    "exchange_timezone": "America/New_York",
    "lot_size": 100,
    "tick_size": 0.01,
    "trading_hours": "09:30-16:00",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": True,
    "tax_model": "standard",
    "tax_rate_gains": 0.15,
    "tax_notes": "Long-term gains (>1yr) taxed at 15-20%. Short-term at ordinary income rate.",
}

MARKET_BR: dict = {
    "market": "BR",
    "country": "Brasil",
    "currency": "BRL",
    "currency_symbol": "R$",
    "exchange": "B3",
    "exchange_suffix": ".SA",
    "exchange_timezone": "America/Sao_Paulo",
    "lot_size": 100,
    "tick_size": 0.01,
    "trading_hours": "10:00-17:00",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": True,
    "tax_model": "BR_standard",
    "tax_rate_gains": 0.15,
    "tax_notes": (
        "Vendas até R$20k/mês em ações isentas de IR. "
        "Day trade: 20%. Swing trade: 15%. "
        "FIIs: 20% sem isenção. Prejuízo compensável no mesmo tipo."
    ),
}

MARKET_SE: dict = {
    "market": "SE",
    "country": "Sverige",
    "currency": "SEK",
    "currency_symbol": "kr",
    "exchange": "OMX",
    "exchange_suffix": ".ST",
    "exchange_timezone": "Europe/Stockholm",
    "lot_size": 1,
    "tick_size": 0.01,
    "trading_hours": "09:00-17:30",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": False,
    "tax_model": "ISK",
    "tax_rate_gains": 0.0,
    "tax_notes": (
        "ISK (Investeringssparkonto): schablonskatt ~1.09% av kapitalet/år, "
        "inga vinstskatter. AF (Aktie- & Fondkonto): 30% skatt på vinst."
    ),
}

MARKET_DE: dict = {
    "market": "DE",
    "country": "Deutschland",
    "currency": "EUR",
    "currency_symbol": "€",
    "exchange": "IBIS",
    "exchange_suffix": ".DE",
    "exchange_timezone": "Europe/Berlin",
    "lot_size": 1,
    "tick_size": 0.01,
    "trading_hours": "09:00-17:30",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": False,
    "tax_model": "DE_standard",
    "tax_rate_gains": 0.26375,
    "tax_notes": "Abgeltungsteuer 25% + Soli 5.5% = 26.375%. Freibetrag €1000/year.",
}

MARKET_UK: dict = {
    "market": "UK",
    "country": "United Kingdom",
    "currency": "GBP",
    "currency_symbol": "£",
    "exchange": "LSE",
    "exchange_suffix": ".L",
    "exchange_timezone": "Europe/London",
    "lot_size": 1,
    "tick_size": 0.01,
    "trading_hours": "08:00-16:30",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": False,
    "tax_model": "ISA",
    "tax_rate_gains": 0.20,
    "tax_notes": "ISA: tax-free allowance £20k/year. Outside ISA: CGT 10-20%.",
}

MARKET_CA: dict = {
    "market": "CA",
    "country": "Canada",
    "currency": "CAD",
    "currency_symbol": "C$",
    "exchange": "TSE",
    "exchange_suffix": ".TO",
    "exchange_timezone": "America/Toronto",
    "lot_size": 100,
    "tick_size": 0.01,
    "trading_hours": "09:30-16:00",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": False,
    "tax_model": "TFSA",
    "tax_rate_gains": 0.25,
    "tax_notes": "TFSA: tax-free. RRSP: tax-deferred. Outside: 50% of gains taxable.",
}

MARKET_HK: dict = {
    "market": "HK",
    "country": "Hong Kong",
    "currency": "HKD",
    "currency_symbol": "HK$",
    "exchange": "SEHK",
    "exchange_suffix": ".HK",
    "exchange_timezone": "Asia/Hong_Kong",
    "lot_size": 100,
    "tick_size": 0.01,
    "trading_hours": "09:30-16:00",
    "settlement_days": 2,
    "has_options": True,
    "has_fractional": False,
    "tax_model": "standard",
    "tax_rate_gains": 0.0,
    "tax_notes": "No capital gains tax in Hong Kong. Stamp duty 0.13% per trade.",
}

# ─── Suffix → Market Profile mapping ──────────────────────────

_SUFFIX_MAP: dict[str, dict] = {
    ".ST": MARKET_SE,
    ".SA": MARKET_BR,
    ".DE": MARKET_DE,
    ".L": MARKET_UK,
    ".TO": MARKET_CA,
    ".HK": MARKET_HK,
}


class ContextResolver:
    """Resolves deterministic DomainContext from ticker analysis.

    Detection priority:
    1. Symbol suffix (.SA, .ST, .DE, etc.)
    2. Falls back to US market (no suffix)
    """

    def resolve(self, symbol: str | None = None) -> DomainContext:
        """
        Build DomainContext from symbol suffix detection.
        Returns immutable context with full market profile.
        """
        if not symbol:
            ctx = DomainContext(**MARKET_US)
            logger.info("Context resolved → %s (default, no symbol)", ctx.market)
            return ctx

        symbol_upper = symbol.upper()
        for suffix, profile in _SUFFIX_MAP.items():
            if symbol_upper.endswith(suffix.upper()):
                ctx = DomainContext(**profile)
                logger.info("Context resolved → %s (%s, suffix=%s)", ctx.market, ctx.exchange, suffix)
                return ctx

        ctx = DomainContext(**MARKET_US)
        logger.info("Context resolved → %s (default for '%s')", ctx.market, symbol)
        return ctx

    def normalize_symbol(self, symbol: str) -> str:
        """
        Ensure symbol has the correct suffix for its detected market.
        e.g., 'PETR4' with no suffix stays as-is (US assumed),
        but the context will tell the caller it's US.
        """
        return symbol.strip().upper()

    def get_market_profile(self, market_code: str) -> DomainContext | None:
        """Get a full market profile by market code (e.g., 'BR', 'SE', 'US')."""
        profiles = {
            "US": MARKET_US,
            "BR": MARKET_BR,
            "SE": MARKET_SE,
            "DE": MARKET_DE,
            "UK": MARKET_UK,
            "CA": MARKET_CA,
            "HK": MARKET_HK,
        }
        profile = profiles.get(market_code.upper())
        if profile:
            return DomainContext(**profile)
        return None
