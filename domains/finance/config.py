"""
Finance Domain Configuration - Market and Exchange Management.

Defines supported exchanges, markets, and system-wide constraints for symbol resolution.
"""

# Supported exchanges and their metadata
SUPPORTED_EXCHANGES = {
    "SA": {
        "name": "B3 (Brazil Stock Exchange)",
        "country": "Brazil",
        "country_code": "BR",
        "suffix_format": ".SA",
    },
    "ST": {
        "name": "Nasdaq Stockholm",
        "country": "Sweden",
        "country_code": "SE",
        "suffix_format": ".ST",
    },
    "US": {
        "name": "US Stock Exchange (NYSE, NASDAQ, etc)",
        "country": "United States",
        "country_code": "US",
        "suffix_format": "",  # US stocks don't have suffix
    },
}

# Exchanges to focus on (others will be deprioritized or filtered)
FOCUS_EXCHANGES = ["SA", "ST", "US"]
FOCUS_COUNTRIES = ["BR", "SE", "US"]

# Market aliases (user input → country/exchange code)
MARKET_ALIASES = {
    # Brazil
    "BOVESPA": "BR",
    "IBOVESPA": "BR",
    "IBOV": "BR",
    "B3": "BR",
    "BRASIL": "BR",
    "BRAZIL": "BR",

    # USA
    "NYSE": "US",
    "NASDAQ": "US",
    "EUA": "US",
    "USA": "US",
    "US": "US",

    # Sweden
    "SUECIA": "SE",
    "SUÉCIA": "SE",
    "SWEDEN": "SE",
    "STOCKHOLM": "SE",
}


def get_supported_exchanges() -> dict:
    """Get all supported exchanges."""
    return SUPPORTED_EXCHANGES.copy()


def is_exchange_supported(exchange: str) -> bool:
    """Check if an exchange is supported."""
    return exchange.upper() in SUPPORTED_EXCHANGES


def is_exchange_in_focus(exchange: str) -> bool:
    """Check if an exchange is in the focus list."""
    return exchange.upper() in FOCUS_EXCHANGES


def get_focus_exchanges() -> list[str]:
    """Get list of focus exchanges."""
    return FOCUS_EXCHANGES.copy()


def get_focus_countries() -> list[str]:
    """Get list of focus countries."""
    return FOCUS_COUNTRIES.copy()


def filter_by_focus_exchange(symbols: list[dict]) -> list[dict]:
    """
    Filter symbol list to prioritize focus exchanges.

    Moves symbols from focus exchanges to the front of the list.

    Args:
        symbols: List of symbol dicts with 'exchange' key

    Returns:
        Reordered list with focus exchange symbols first
    """
    focus_symbols = []
    other_symbols = []

    for symbol in symbols:
        exchange = (symbol.get("exchange") or "").upper()
        if is_exchange_in_focus(exchange):
            focus_symbols.append(symbol)
        else:
            other_symbols.append(symbol)

    return focus_symbols + other_symbols


def get_llm_exchange_guidance() -> str:
    """
    Get LLM prompt guidance on supported exchanges.

    Returns a string for inclusion in LLM prompts about which exchanges to prioritize.
    """
    focus_list = ", ".join(FOCUS_EXCHANGES)
    countries_list = ", ".join(FOCUS_COUNTRIES)

    return f"""IMPORTANT: Only resolve symbols to these exchanges: {focus_list}
Corresponding countries: {countries_list}
If a symbol exists on other exchanges, prefer the focus exchanges.
If only non-focus exchanges are available, return the best match but indicate low confidence."""
