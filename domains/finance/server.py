"""
Finance Domain Server — Standalone Service.

Responsibility:
- Expose Finance Logic via HTTP (Standard Domain Protocol)
- Manage Skill Gateway & MCP Connection
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow absolute imports when run directly
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from shared.models import ExecutionIntent, IntentOutput, DomainOutput
from domains.finance.handler import FinanceDomainHandler
from domains.finance.symbol_resolver import SymbolResolver
from domains.finance.config import MARKET_ALIASES as CONFIG_MARKET_ALIASES
from skills.gateway import SkillGateway
from skills.registry import SkillRegistry
from skills.implementations.mcp_adapter import MCPAdapter

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_server")

# Environment
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/sse")

# Static Metadata Overrides (Templates & Rules)
# This enriches the raw MCP tools with Orchestrator-specific UI/Logic hints.
# Base market aliases from config, plus any non-focus exchanges for reference
MARKET_ALIASES = {
    **CONFIG_MARKET_ALIASES,
    # Non-focus exchanges (for reference, but not prioritized)
    "HONG KONG": "HK",
    "HONGKONG": "HK",
    "HKEX": "HK",
    # Extended Portuguese / English forms
    "ESTADOS UNIDOS": "US",
    "AMERICA": "US",
    "AMERICANO": "US",
    "MERCADO AMERICANO": "US",
    "BRASILEIRO": "BR",
    "MERCADO BRASILEIRO": "BR",
    "SUECO": "SE",
    "MERCADO SUECO": "SE",
    "OMX": "SE",
}

SYMBOL_ALIASES = {
    # Brazilian Stocks (B3/SA Exchange)
    "PETRO": "PETR4.SA",
    "PETRO4": "PETR4.SA",
    "PETROBRAS": "PETR4.SA",
    "PETR": "PETR4.SA",
    "VALE": "VALE3.SA",
    "VALE3": "VALE3.SA",
    "ITAU": "ITUB4.SA",
    "ITAU UNIBANCO": "ITUB4.SA",
    "ITUB": "ITUB4.SA",
    "BBAS": "BBAS3.SA",
    "BANCO DO BRASIL": "BBAS3.SA",
    "BRASIL": "BBAS3.SA",
    "BB": "BBAS3.SA",
    "MGLU": "MGLU3.SA",
    "MAGAZINE LUIZA": "MGLU3.SA",
    "BBDC": "BBDC4.SA",
    "BRADESCO": "BBDC4.SA",
    "CSAN": "CSAN3.SA",
    "COSAN": "CSAN3.SA",
    "JBSS": "JBSS3.SA",
    "JBS": "JBSS3.SA",
    "ABEV": "ABEV3.SA",
    "AMBEV": "ABEV3.SA",
    "EMBRAER": "EMBE3.SA",
    "EMBE": "EMBE3.SA",
    "NATURA": "NATU3.SA",
    "NATURA COSMETICOS": "NATU3.SA",
    "NATU": "NATU3.SA",

    # USA Stocks (US Exchange)
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "TSLA": "TSLA",
    "TESLA": "TSLA",
    "MSFT": "MSFT",
    "MICROSOFT": "MSFT",
    "NVDA": "NVDA",
    "NVIDIA": "NVDA",
    "GOOGL": "GOOGL",
    "GOOGLE": "GOOGL",
    "GOOG": "GOOGL",
    "AMZN": "AMZN",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "FB": "META",
    "NFLX": "NFLX",
    "NETFLIX": "NFLX",

    # Swedish Stocks (ST Exchange - Nasdaq Stockholm)
    "NORDEA": "NDA-SE.ST",
    "NDA": "NDA-SE.ST",
    "TELIA": "TELIA.ST",
    "TELIA COMPANY": "TELIA.ST",
    "VOLVO": "VOLV-B.ST",
    "VOLVO-B": "VOLV-B.ST",
    "HM": "HM-B.ST",
    "H&M": "HM-B.ST",
    "HENNES": "HM-B.ST",
    "SKF": "SKF-B.ST",
    "ELECTROLUX": "ELUX-B.ST",
    "ELUX": "ELUX-B.ST",
    "ERICSSON": "ERIC-B.ST",
    "ERIC": "ERIC-B.ST",
    "SEB": "SEB-A.ST",
    "SEB BANK": "SEB-A.ST",
    "SWEDBANK": "SWED-A.ST",
    "SWED": "SWED-A.ST",
    "ASEA": "ASEA-B.ST",
}

# Global State
handler: FinanceDomainHandler | None = None
mcp_adapter: MCPAdapter | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle of the Finance Server."""
    global handler, mcp_adapter

    logger.info("Starting Finance Server...")

    # Initialize Skills
    mcp_adapter = MCPAdapter(mcp_url=MCP_URL)
    skill_registry = SkillRegistry()
    skill_registry.register("mcp_finance", mcp_adapter)
    skill_gateway = SkillGateway(skill_registry)

    # Initialize Symbol Resolver with production SYMBOL_ALIASES
    symbol_resolver = SymbolResolver(
        aliases=SYMBOL_ALIASES,
        skill_gateway=skill_gateway,
        enable_llm=False,  # LLM fallback disabled in server (only in tests)
    )

    # Initialize Parameter Resolver (DB-backed deterministic + LLM fallback)
    from domains.finance.parameter_resolver_db import ParameterResolverDB
    from domains.finance.parameter_resolver import (
        ParameterResolver,
        ParameterResolutionConfig,
        SymbolSubResolver,
        SymbolListSubResolver,
    )
    from domains.finance.parameter_seed import seed_parameter_database

    pr_db = ParameterResolverDB()
    seed_parameter_database(pr_db)
    parameter_resolver = ParameterResolver(
        db=pr_db,
        model_selector=None,  # LLM fallback disabled in standalone server mode
        config=ParameterResolutionConfig(enable_llm=False),
    )
    parameter_resolver.register_resolver("symbol", SymbolSubResolver(symbol_resolver))
    parameter_resolver.register_resolver("symbols", SymbolListSubResolver(symbol_resolver))

    # Initialize Domain Handler with local capabilities for metadata-driven logic
    from registry.domain_registry import HandlerRegistry
    mock_registry = HandlerRegistry()
    manifest = get_manifest()
    for cap in manifest["capabilities"]:
        # Register capability to a dummy handler so we can use get_metadata
        cap_metadata = dict(cap.get("metadata", {}))
        cap_metadata.setdefault("schema", cap.get("schema", {}))
        cap_metadata.setdefault("description", cap.get("description", ""))
        mock_registry.register_capability(cap["name"], None, metadata=cap_metadata)

    # ─── Initialize 3-Tier Architecture ─────────────────────────────

    # Tier 1: Facts (MCP passthrough — always available)
    from domains.finance.context import ContextResolver
    from domains.finance.core import StrategyCore
    from domains.finance.tiers.facts import FactsTier

    context_resolver = ContextResolver()
    strategy_core = StrategyCore()
    facts_tier = FactsTier(
        skill_gateway=skill_gateway,
        context_resolver=context_resolver,
        strategy_core=strategy_core,
        registry=mock_registry,
    )

    # Tier 2: Calculator (local math + MCP data fetching)
    from domains.finance.tiers.calculators import CalculatorTier, CalculatorRegistry
    from domains.finance.calculators.options import register_options_calculators
    from domains.finance.calculators.finance import register_finance_calculators
    from domains.finance.calculators.portfolio import register_portfolio_calculators

    calc_registry = CalculatorRegistry()
    register_options_calculators(calc_registry)
    register_finance_calculators(calc_registry)
    register_portfolio_calculators(calc_registry)
    calculator_tier = CalculatorTier(
        calculator_registry=calc_registry,
        skill_gateway=skill_gateway,
    )

    # Tier 3: Analysis (LLM + skills — optional, requires model_selector)
    analysis_tier = None
    model_selector = None
    try:
        model_selector_env = os.getenv("MODEL_BASE_URL", "").strip()
        if model_selector_env:
            from models.selector import ModelSelector
            model_selector = ModelSelector()

            from domains.finance.tiers.agents import AnalysisTier
            from domains.finance.analysis_skills.registry import FinanceSkillRegistry
            from domains.finance.analysis_skills.stock_analyst import StockAnalystSkill
            from domains.finance.analysis_skills.options_analyst import OptionsAnalystSkill

            finance_skill_registry = FinanceSkillRegistry()
            finance_skill_registry.register(StockAnalystSkill(), ["analyze_stock"])
            finance_skill_registry.register(OptionsAnalystSkill(), ["analyze_options"])

            analysis_tier = AnalysisTier(
                skill_registry=finance_skill_registry,
                facts_tier=facts_tier,
                calculator_tier=calculator_tier,
                model_selector=model_selector,
            )
            logger.info("Analysis tier initialized with %d skills", len(finance_skill_registry.list_skill_names()))
        else:
            logger.info("Analysis tier disabled (MODEL_BASE_URL not set)")
    except Exception as e:
        logger.warning("Analysis tier initialization failed: %s", e)

    # ─── Initialize Domain Handler with Tiers ─────────────────────

    handler = FinanceDomainHandler(
        skill_gateway=skill_gateway,
        registry=mock_registry,
        symbol_resolver=symbol_resolver,
        parameter_resolver=parameter_resolver,
        model_selector=model_selector,
        facts_tier=facts_tier,
        calculator_tier=calculator_tier,
        analysis_tier=analysis_tier,
    )

    logger.info(
        "Finance Server ready — Facts: OK | Calculator: %d functions | Analysis: %s",
        len(calc_registry.list_capabilities()),
        "OK" if analysis_tier else "disabled",
    )

    yield

    logger.info("Shutting down Finance Server...")
    if mcp_adapter:
        mcp_adapter.close()

app = FastAPI(title="Finance Domain Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

RANKING_PERIOD_ALIASES = {
    "HOJE": "1d",
    "AGORA": "1d",
    "TODAY": "1d",
    "ULTIMO DIA": "1d",
    "DIA": "1d",
    "DAY": "1d",
    "SEMANA": "5d",
    "SEMANAL": "5d",
    "ULTIMOS DIAS": "5d",
    "WEEK": "5d",
    "LAST WEEK": "5d",
    "MÊS": "1mo",
    "MES": "1mo",
    "MENSAL": "1mo",
    "ULTIMO MES": "1mo",
    "ULTIMOS MES": "1mo",
    "ULTIMO MÊS": "1mo",
    "ULTIMOS MÊS": "1mo",
    "MONTH": "1mo",
    "LAST MONTH": "1mo",
    "TRIMESTRE": "3mo",
    "ULTIMO TRIMESTRE": "3mo",
    "QUARTER": "3mo",
    "LAST QUARTER": "3mo",
    "ANO": "1y",
    "ANUAL": "1y",
    "ULTIMO ANO": "1y",
    "YEAR": "1y",
    "LAST YEAR": "1y",
}

HISTORICAL_PERIOD_ALIASES = {
    "HOJE": "1d",
    "TODAY": "1d",
    "1 DIA": "1d",
    "1 DAY": "1d",
    "5 DIAS": "5d",
    "5 DAYS": "5d",
    "SEMANA": "5d",
    "WEEK": "5d",
    "1 WEEK": "5d",
    "1 MÊS": "1mo",
    "1 MES": "1mo",
    "MÊS": "1mo",
    "MES": "1mo",
    "1 MONTH": "1mo",
    "MONTH": "1mo",
    "3 MESES": "3mo",
    "3 MONTHS": "3mo",
    "6 MESES": "6mo",
    "6 MONTHS": "6mo",
    "SEMESTRE": "6mo",
    "1 ANO": "1y",
    "ANO": "1y",
    "1 YEAR": "1y",
    "YEAR": "1y",
    "2 ANOS": "2y",
    "2 YEARS": "2y",
    "5 ANOS": "5y",
    "5 YEARS": "5y",
    "YTD": "ytd",
    "MAX": "max",
}

INTERVAL_ALIASES = {
    "DIARIO": "1d",
    "DIÁRIO": "1d",
    "DIA": "1d",
    "DAILY": "1d",
    "DAY": "1d",
    "SEMANAL": "1wk",
    "SEMANA": "1wk",
    "WEEKLY": "1wk",
    "WEEK": "1wk",
    "MENSAL": "1mo",
    "MES": "1mo",
    "MÊS": "1mo",
    "MONTHLY": "1mo",
    "MONTH": "1mo",
}

SIGNAL_TYPE_ALIASES = {
    "RSI OVERSOLD": "rsi_oversold",
    "RSI SOBREVENDIDO": "rsi_oversold",
    "SOBREVENDIDO": "rsi_oversold",
    "OVERSOLD": "rsi_oversold",
    "RSI OVERBOUGHT": "rsi_overbought",
    "RSI SOBRECOMPRADO": "rsi_overbought",
    "SOBRECOMPRADO": "rsi_overbought",
    "OVERBOUGHT": "rsi_overbought",
    "MACD CROSS": "macd_cross",
    "CRUZAMENTO MACD": "macd_cross",
    "MACD": "macd_cross",
    "CRUZAMENTO": "macd_cross",
}

MARKET_PARAM_SPEC = {
    "type": "string",
    "required": True,
    "examples": ["BR", "US", "SE"],
    "normalization": {"case": "upper"},
    "aliases": MARKET_ALIASES,
}

SYMBOL_PARAM_SPEC = {
    "type": "string",
    "required": True,
    "examples": ["PETR4.SA", "VALE3.SA", "AAPL"],
    "normalization": {"case": "upper"},
    "aliases": SYMBOL_ALIASES,
}

SYMBOL_SEARCH_FLOW = {
    "pre": [
        {
            "type": "resolve_symbol",
            "param": "symbol",
            "required": True,
            "search_capability": "yahoo_search",
            "search_fallback_capabilities": ["search_by_name"],
        }
    ]
}

SYMBOL_LIST_SEARCH_FLOW = {
    "pre": [
        {
            "type": "resolve_symbol_list",
            "param": "symbols",
            "required": True,
            "search_capability": "yahoo_search",
            "search_fallback_capabilities": ["search_by_name"],
        }
    ]
}

METADATA_OVERRIDES = {
    "get_stock_price": {
        "intent_description": (
            "Get current stock price/quote for one or more symbols. "
            "Use for questions about current value, quote, and ticker/company mentions."
        ),
        "intent_hints": {
            "keywords": [
                "qual o preco",
                "qual o preço",
                "qual o valor",
                "cotacao",
                "cotação",
                "quanto esta",
                "quanto está",
                "ticker",
                "stock price",
            ],
            "examples": [
                "qual o valor da petr4?",
                "quanto esta a vale3 hoje?",
                "price of aapl",
            ],
        },
        "explanation_template": "{symbol} is trading at {result[price]} {currency}.",
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "symbols": {
                "type": "array",
                "items": {
                    "type": "string",
                    "normalization": {"case": "upper"},
                    "aliases": SYMBOL_ALIASES,
                },
                "examples": [["VALE3.SA", "PETR4.SA"]],
            },
            "currency": {
                "type": "string",
                "infer_from_symbol_suffix": {
                    ".SA": "BRL",
                    ".ST": "SEK",
                },
                "default": "USD",
                "aliases": {
                    "REAL": "BRL",
                    "REAIS": "BRL",
                    "DOLAR": "USD",
                    "DÓLAR": "USD",
                    "DOLARES": "USD",
                    "DÓLARES": "USD",
                },
                "value_labels": {
                    "USD": "dólar",
                    "BRL": "real",
                    "SEK": "coroa sueca",
                },
            },
            "exchange": {
                "type": "string",
                "infer_from_symbol_suffix": {
                    ".SA": "BOVESPA",
                    ".ST": "SFB",
                },
                "default": "SMART",
                "aliases": {
                    "B3": "BOVESPA",
                    "IBOVESPA": "BOVESPA",
                    "NASDAQ": "SMART",
                    "NYSE": "SMART",
                },
                "value_labels": {
                    "SMART": "mercado dos EUA",
                    "BOVESPA": "Bovespa",
                    "SFB": "bolsa da Suécia",
                },
            },
        },
        "decomposition": {
            "array_params": [
                {
                    "param_name": "symbols_text",
                    "single_param_name": "symbol_text",
                    "max_concurrency": 4,
                },
                {
                    "param_name": "symbols",
                    "single_param_name": "symbol",
                    "max_concurrency": 4,
                },
            ]
        },
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "get_historical_data": {
        "intent_description": (
            "Get historical OHLC price data for a symbol and period. "
            "Use for history, chart, and time-series requests."
        ),
        "intent_hints": {
            "keywords": [
                "historico",
                "histórico",
                "grafico",
                "gráfico",
                "serie historica",
                "dados historicos",
                "preco nos ultimos",
            ],
            "examples": [
                "historico da petr4 no ultimo mes",
                "grafico de vale3 em 1 ano",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "1mo",
                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                "aliases": HISTORICAL_PERIOD_ALIASES,
                "examples": ["1mo", "1y", "ytd"],
            },
            "interval": {
                "type": "string",
                "default": "1d",
                "enum": ["1d", "1wk", "1mo"],
                "aliases": INTERVAL_ALIASES,
                "examples": ["1d", "1wk", "1mo"],
            },
        },
        "explanation_template": "Historical data for {symbol} ({params[period]}).",
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "get_stock_screener": {
        "intent_description": (
            "Screen stocks by market/filters and ranking criteria. "
            "Use for discovery questions like 'melhores ações por setor'."
        ),
        "intent_hints": {
            "keywords": [
                "screener",
                "filtrar acoes",
                "filtrar ações",
                "ranking de acoes",
                "acoes por setor",
            ],
            "examples": [
                "screener de acoes no brasil",
                "melhores acoes de tecnologia nos eua",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "valid_values": {"market": ["US", "BR", "SE", "HK"]},
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "sector": {
                "type": "string",
                "examples": ["technology", "financials", "energy", "healthcare", "consumer_defensive"],
                "enum": [
                    "technology", "financials", "energy", "healthcare",
                    "consumer_defensive", "consumer_cyclical", "industrials",
                    "basic_materials", "communication_services", "utilities",
                    "real_estate",
                ],
                "aliases": {
                    "TECNOLOGIA": "technology",
                    "TECH": "technology",
                    "FINANCEIRO": "financials",
                    "FINANCEIRAS": "financials",
                    "BANCOS": "financials",
                    "ENERGIA": "energy",
                    "SAUDE": "healthcare",
                    "SAÚDE": "healthcare",
                    "HEALTH": "healthcare",
                    "CONSUMO DEFENSIVO": "consumer_defensive",
                    "CONSUMO BASICO": "consumer_defensive",
                    "CONSUMO BÁSICO": "consumer_defensive",
                    "CONSUMO": "consumer_cyclical",
                    "CONSUMO CICLICO": "consumer_cyclical",
                    "CONSUMO CÍCLICO": "consumer_cyclical",
                    "INDUSTRIA": "industrials",
                    "INDÚSTRIA": "industrials",
                    "INDUSTRIAL": "industrials",
                    "MATERIAIS": "basic_materials",
                    "MATERIAIS BASICOS": "basic_materials",
                    "MATERIAIS BÁSICOS": "basic_materials",
                    "COMUNICACAO": "communication_services",
                    "COMUNICAÇÃO": "communication_services",
                    "TELECOM": "communication_services",
                    "UTILIDADES": "utilities",
                    "UTILITIES": "utilities",
                    "IMOBILIARIO": "real_estate",
                    "IMOBILIÁRIO": "real_estate",
                    "REAL ESTATE": "real_estate",
                },
            },
            "sort_by": {
                "type": "string",
                "default": "market_cap",
                "enum": ["market_cap", "pe_ratio", "dividend_yield", "price", "change_pct", "volume"],
                "aliases": {
                    "VALOR DE MERCADO": "market_cap",
                    "MARKET CAP": "market_cap",
                    "CAPITALIZACAO": "market_cap",
                    "CAPITALIZAÇÃO": "market_cap",
                    "P/L": "pe_ratio",
                    "P/E": "pe_ratio",
                    "PE": "pe_ratio",
                    "PE RATIO": "pe_ratio",
                    "DY": "dividend_yield",
                    "DIVIDENDO": "dividend_yield",
                    "DIVIDEND YIELD": "dividend_yield",
                    "DIVIDENDOS": "dividend_yield",
                    "PRECO": "price",
                    "PREÇO": "price",
                    "PRICE": "price",
                    "VARIACAO": "change_pct",
                    "VARIAÇÃO": "change_pct",
                    "CHANGE": "change_pct",
                    "VOLUME": "volume",
                },
                "examples": ["market_cap", "pe_ratio", "dividend_yield"],
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Stock screener results for {result[market]} market.",
    },
    "get_top_gainers": {
        "intent_description": (
            "List top gainers for a market/period. "
            "Use for momentum/ranking questions like 'maiores altas'."
        ),
        "intent_hints": {
            "keywords": [
                "maiores altas",
                "maiores ganhos",
                "ganhos",
                "top gainers",
                "acoes que mais subiram",
                "ações que mais subiram",
                "mais altas hoje",
                "ranking de altas",
            ],
            "examples": [
                "quais as maiores altas de hoje do bovespa?",
                "top gainers no brasil hoje",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "default_period": "1d",
        "valid_values": {"market": ["US", "BR", "SE"], "period": ["1d", "5d", "1mo", "3mo", "1y"]},
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "1d",
                "enum": ["1d", "5d", "1mo", "3mo", "1y"],
                "aliases": RANKING_PERIOD_ALIASES,
                "examples": ["1d", "5d", "1mo"],
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Top gainers in {result[market]} for {result[period]}.",
    },
    "get_top_losers": {
        "intent_description": (
            "List top losers for a market/period. "
            "Use for momentum/ranking questions like 'maiores baixas'."
        ),
        "intent_hints": {
            "keywords": [
                "maiores baixas",
                "maiores perdas",
                "perdas",
                "top losers",
                "acoes que mais cairam",
                "ações que mais caíram",
                "mais baixas hoje",
                "ranking de baixas",
            ],
            "examples": [
                "quais as maiores baixas de hoje do bovespa?",
                "top losers no brasil hoje",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "default_period": "1d",
        "valid_values": {"market": ["US", "BR", "SE"], "period": ["1d", "5d", "1mo", "3mo", "1y"]},
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "1d",
                "enum": ["1d", "5d", "1mo", "3mo", "1y"],
                "aliases": RANKING_PERIOD_ALIASES,
                "examples": ["1d", "5d", "1mo"],
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Top losers in {result[market]} for {result[period]}.",
    },
    "get_top_dividend_payers": {
        "intent_description": (
            "List high dividend yield stocks for a market. "
            "Use for dividend ranking and income-focused requests."
        ),
        "intent_hints": {
            "keywords": [
                "dividendos",
                "dividend yield",
                "maiores dividendos",
                "acoes pagadoras",
                "ações pagadoras",
            ],
            "examples": [
                "quais acoes pagam mais dividendos no brasil?",
                "top dividendos nos eua",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "valid_values": {"market": ["US", "BR", "SE"]},
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "High yield stocks in {result[market]}.",
    },
    "get_technical_signals": {
        "intent_description": (
            "Find stocks with technical signals (RSI/MACD) by market. "
            "Use for technical analysis screeners."
        ),
        "intent_hints": {
            "keywords": [
                "sinal tecnico",
                "sinal técnico",
                "rsi",
                "macd",
                "sobrevendido",
                "sobrecomprado",
            ],
            "examples": [
                "acoes com rsi sobrevendido no brasil",
                "sinais macd no mercado americano",
            ],
        },
        "market_required": True,
        "signal_required": True,
        "default_market": "SE",
        "default_signal_type": "rsi_oversold",
        "valid_values": {"market": ["US", "BR", "SE"], "signal_type": ["rsi_oversold", "rsi_overbought", "macd_cross"]},
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "signal_type": {
                "type": "string",
                "required": True,
                "default": "rsi_oversold",
                "enum": ["rsi_oversold", "rsi_overbought", "macd_cross"],
                "aliases": SIGNAL_TYPE_ALIASES,
                "examples": ["rsi_oversold", "rsi_overbought", "macd_cross"],
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Stocks showing {params[signal_type]} signals in {result[market]}.",
    },
    "list_jobs": {
        "intent_description": "List available data pipeline jobs.",
        "intent_hints": {
            "keywords": ["listar jobs", "jobs disponiveis", "jobs disponíveis", "pipeline jobs"],
            "examples": ["quais jobs de dados estao disponiveis?"],
        },
        "explanation_template": "Data pipeline job list.",
    },
    "get_job_status": {
        "intent_description": "Get status/health of a pipeline job.",
        "intent_hints": {
            "keywords": ["status do job", "job status", "pipeline status", "saude do pipeline"],
            "examples": ["qual o status do job earnings_sync?"],
        },
        "parameter_specs": {
            "job_name": {
                "type": "string",
                "required": True,
                "examples": ["earnings_sync", "prices_daily"],
            }
        },
        "explanation_template": "Pipeline health status overview.",
    },
    "get_fundamentals": {
        "intent_description": "Get fundamentals for a specific stock symbol.",
        "intent_hints": {
            "keywords": ["fundamentos", "fundamental", "valuation", "balanco da empresa"],
            "examples": ["fundamentos da vale3", "valuation da petr4"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Fundamentals data for {symbol} ({market} market).",
    },
    "get_dividend_history": {
        "intent_description": "Get dividend history for a specific stock symbol.",
        "intent_hints": {
            "keywords": ["dividendos da", "historico de dividendos", "dividend history"],
            "examples": ["dividendos da petr4", "historico de dividendos da vale3"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "2y",
                "enum": ["1y", "2y", "5y"],
                "aliases": {
                    "1 ANO": "1y",
                    "ANO": "1y",
                    "ULTIMO ANO": "1y",
                    "1 YEAR": "1y",
                    "YEAR": "1y",
                    "2 ANOS": "2y",
                    "DOIS ANOS": "2y",
                    "2 YEARS": "2y",
                    "5 ANOS": "5y",
                    "CINCO ANOS": "5y",
                    "5 YEARS": "5y",
                },
                "examples": ["2y", "5y"],
            },
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Dividend history for {symbol} ({currency}).",
    },
    "get_company_profile": {
        "intent_description": "Get company profile and business information for a symbol.",
        "intent_hints": {
            "keywords": ["sobre a empresa", "company info", "setor da empresa", "perfil da empresa", "company profile"],
            "examples": ["me fale sobre a empresa vale3", "company info da aapl"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Company profile for {result[name]}.",
    },
    "get_option_chain": {
        "intent_description": "Get option chain for a stock symbol.",
        "intent_hints": {
            "keywords": ["option chain", "cadeia de opcoes", "opcoes da acao", "opções da ação"],
            "examples": ["option chain da aapl", "cadeia de opcoes da petr4"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "expiry": {
                "type": "string",
                "format": "date",
                "description": "Option expiry date (YYYY-MM-DD). Accepts relative expressions like 'essa semana', 'next month'.",
                "examples": ["2026-03-20", "2026-06-19"],
            },
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Option chain for {symbol}.",
    },
    "get_option_greeks": {
        "intent_description": "Get options Greeks for a stock symbol.",
        "intent_hints": {
            "keywords": ["greeks", "delta gamma", "gregas das opcoes", "gregas das opções"],
            "examples": ["gregas das opcoes de aapl", "delta gamma da petr4"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "expiry": {
                "type": "string",
                "format": "date",
                "description": "Option expiry date (YYYY-MM-DD). Accepts relative expressions.",
                "examples": ["2026-03-20", "2026-06-19"],
            },
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Option Greeks for {symbol}.",
    },
    "get_financial_statements": {
        "intent_description": "Get financial statements for a stock symbol.",
        "intent_hints": {
            "keywords": ["demonstracoes financeiras", "financial statements", "dre", "balanco patrimonial"],
            "examples": ["demonstracoes financeiras da vale3", "financial statements de msft"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Financial statements for {symbol}.",
    },
    "search_by_name": {
        "intent_description": "Search ticker symbols by company name.",
        "intent_hints": {
            "keywords": ["qual o ticker", "procurar ticker", "buscar simbolo", "buscar símbolo"],
            "examples": ["qual o ticker da petrobras?", "buscar simbolo da nordea"],
        },
        "parameter_specs": {
            "name": {
                "type": "string",
                "required": True,
                "examples": ["petrobras", "nordea"],
            }
        },
        "explanation_template": "Search results for '{params[name]}'.",
    },
    "yahoo_search": {
        "intent_description": "Search symbols using Yahoo lookup.",
        "intent_hints": {
            "keywords": ["buscar no yahoo", "yahoo ticker search", "lookup de ticker"],
            "examples": ["yahoo search para vale3"],
        },
        "parameter_specs": {
            "query": {
                "type": "string",
                "required": True,
                "examples": ["vale3", "aapl"],
            }
        },
        "explanation_template": "Yahoo search results for '{params[query]}'.",
    },
    # ─── Expanded Screeners ────────────────────────────────────────────
    "get_most_active_stocks": {
        "intent_description": "Find most actively traded stocks by volume for a market.",
        "intent_hints": {
            "keywords": [
                "mais negociadas", "mais ativas", "volume alto",
                "most active", "mais liquidez",
            ],
            "examples": [
                "quais as acoes mais negociadas hoje no brasil?",
                "most active stocks in sweden",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "1D",
                "enum": ["1D", "5D"],
                "aliases": {
                    "HOJE": "1D",
                    "TODAY": "1D",
                    "DIA": "1D",
                    "DIARIO": "1D",
                    "DIÁRIO": "1D",
                    "DAILY": "1D",
                    "1D": "1D",
                    "SEMANA": "5D",
                    "SEMANAL": "5D",
                    "WEEK": "5D",
                    "WEEKLY": "5D",
                    "5 DIAS": "5D",
                    "5 DAYS": "5D",
                    "5D": "5D",
                },
                "examples": ["1D", "5D"],
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Most active stocks in {result[market]}.",
    },
    "get_oversold_stocks": {
        "intent_description": "Find oversold stocks (low RSI) for a market.",
        "intent_hints": {
            "keywords": [
                "sobrevendido", "oversold", "rsi baixo",
                "oportunidade de compra", "acoes baratas rsi",
            ],
            "examples": [
                "acoes sobrevendidas no brasil",
                "oversold stocks in sweden",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Oversold stocks in {result[market]}.",
    },
    "get_overbought_stocks": {
        "intent_description": "Find overbought stocks (high RSI) for a market.",
        "intent_hints": {
            "keywords": [
                "sobrecomprado", "overbought", "rsi alto",
                "excessivamente comprado",
            ],
            "examples": [
                "acoes sobrecompradas no brasil",
                "overbought stocks in usa",
            ],
        },
        "market_required": True,
        "default_market": "SE",
        "parameter_specs": {
            "market": dict(MARKET_PARAM_SPEC),
            "limit": {
                "type": "integer",
                "default": 10,
                "examples": [10, 20],
            },
        },
        "explanation_template": "Overbought stocks in {result[market]}.",
    },
    # ─── Fundamentals Intelligence ─────────────────────────────────────
    "get_analyst_recommendations": {
        "intent_description": "Get analyst consensus recommendations (buy/hold/sell) for a stock.",
        "intent_hints": {
            "keywords": [
                "recomendacao", "recomendação", "analistas",
                "consenso", "buy sell hold", "analyst recommendation",
            ],
            "examples": [
                "qual a recomendacao dos analistas para petr4?",
                "analyst recommendations for aapl",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Analyst recommendations for {symbol}.",
    },
    "get_technical_analysis": {
        "intent_description": "Get comprehensive technical analysis indicators for a stock.",
        "intent_hints": {
            "keywords": [
                "analise tecnica", "análise técnica", "indicadores tecnicos",
                "technical analysis", "analise completa tecnica",
            ],
            "examples": [
                "analise tecnica da vale3",
                "technical analysis of aapl",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "period": {
                "type": "string",
                "default": "1y",
                "enum": ["6mo", "1y", "2y"],
                "aliases": {
                    "6 MESES": "6mo",
                    "SEMESTRE": "6mo",
                    "6 MONTHS": "6mo",
                    "1 ANO": "1y",
                    "ANO": "1y",
                    "1 YEAR": "1y",
                    "YEAR": "1y",
                    "2 ANOS": "2y",
                    "DOIS ANOS": "2y",
                    "2 YEARS": "2y",
                },
                "examples": ["1y", "6mo"],
            },
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Technical analysis for {symbol}.",
    },
    "get_news_sentiment": {
        "intent_description": "Get news sentiment analysis for a stock.",
        "intent_hints": {
            "keywords": [
                "sentimento", "noticias", "notícias", "news sentiment",
                "humor do mercado", "sentimento de mercado",
            ],
            "examples": [
                "qual o sentimento das noticias sobre petr4?",
                "news sentiment for tsla",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "News sentiment for {symbol}.",
    },
    "get_comprehensive_stock_info": {
        "intent_description": "Get comprehensive information about a stock (price, fundamentals, analyst views, technicals).",
        "intent_hints": {
            "keywords": [
                "tudo sobre", "analise completa", "análise completa",
                "deep dive", "informacoes completas", "visao geral",
                "comprehensive", "resumo completo",
            ],
            "examples": [
                "me fale tudo sobre a petr4",
                "deep dive in aapl",
                "analise completa da vale3",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Comprehensive analysis for {symbol}.",
    },
    # ─── Wheel Strategy (Facts Tier — MCP passthrough) ─────────────────
    "get_wheel_put_candidates": {
        "intent_description": "Find put option candidates for the wheel strategy on a given stock.",
        "intent_hints": {
            "keywords": [
                "wheel puts", "puts para wheel", "candidatos wheel",
                "vender puts", "sell puts", "put candidates",
            ],
            "examples": [
                "quais os melhores puts para wheel na nordea?",
                "wheel put candidates for aapl",
            ],
        },
        "default_market": "sweden",
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "market": {
                "type": "string",
                "default": "sweden",
                "examples": ["sweden", "brazil", "usa"],
            },
            "delta_min": {"type": "number", "default": 0.25},
            "delta_max": {"type": "number", "default": 0.35},
            "dte_min": {"type": "integer", "default": 4},
            "dte_max": {"type": "integer", "default": 10},
            "limit": {"type": "integer", "default": 5},
            "require_liquidity": {"type": "boolean", "default": True},
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Wheel put candidates for {symbol}.",
    },
    "calc_put_return": {
        "tier": "calculator",
        "intent_description": "Calculate return metrics for a put position (ROC, breakeven, cushion).",
        "intent_hints": {
            "keywords": [
                "retorno do put", "rendimento put", "quanto vou ganhar",
                "put return", "retorno opcao", "option return",
            ],
            "examples": [
                "qual o retorno do put de nordea strike 120 com premio 2.50?",
                "put return for aapl strike 180 premium 3.00",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "strike": {"type": "number", "required": True},
            "premium": {"type": "number", "required": True},
        },
        "schema": {"required": ["symbol", "strike", "premium"]},
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "get_wheel_covered_call_candidates": {
        "intent_description": "Find covered call candidates for the wheel strategy after assignment.",
        "intent_hints": {
            "keywords": [
                "covered calls", "calls cobertas", "wheel calls",
                "vender calls", "call candidates",
            ],
            "examples": [
                "covered calls para nordea com custo medio de 120?",
                "wheel call candidates for aapl cost basis 180",
            ],
        },
        "default_market": "sweden",
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "average_cost": {"type": "number", "required": True, "description": "Average cost basis per share"},
            "market": {"type": "string", "default": "sweden"},
            "delta_min": {"type": "number", "default": 0.25},
            "delta_max": {"type": "number", "default": 0.35},
            "dte_min": {"type": "integer", "default": 4},
            "dte_max": {"type": "integer", "default": 21},
            "min_upside_pct": {"type": "number", "default": 1.0},
            "limit": {"type": "integer", "default": 5},
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Covered call candidates for {symbol}.",
    },
    "calc_contract_capacity": {
        "tier": "calculator",
        "intent_description": "Calculate how many option contracts you can sell given capital and allocation percentage.",
        "intent_hints": {
            "keywords": [
                "quantos contratos", "capacidade capital", "quanto posso vender",
                "contract capacity", "capital allocation", "quanto do capital",
            ],
            "examples": [
                "quantos contratos posso vender de nordea com 100000 sek?",
                "quantos contratos com 20% dos meus 500000?",
                "contract capacity for aapl with 50000",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "capital": {"type": "number", "required": True, "description": "Available capital"},
            "allocation_pct": {"type": "number", "default": 1.0, "description": "Fraction of capital (0.2 = 20%)"},
            "strike": {"type": "number"},
            "margin_requirement_pct": {"type": "number", "default": 1.0},
        },
        "schema": {"required": ["symbol", "capital"]},
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "build_wheel_multi_stock_plan": {
        "intent_description": "Build a diversified wheel strategy plan across multiple stocks.",
        "intent_hints": {
            "keywords": [
                "plano wheel multiplos", "carteira wheel", "distribuir capital",
                "multi stock plan", "diversificar wheel",
            ],
            "examples": [
                "monte um plano wheel com 500000 sek",
                "build wheel plan with 100000 sek across multiple stocks",
            ],
        },
        "default_market": "sweden",
        "parameter_specs": {
            "capital_sek": {"type": "number", "required": True, "description": "Total capital to allocate in SEK"},
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of symbols to consider (optional, auto-selects if empty)",
            },
            "market": {"type": "string", "default": "sweden"},
            "delta_min": {"type": "number", "default": 0.25},
            "delta_max": {"type": "number", "default": 0.35},
            "dte_min": {"type": "integer", "default": 4},
            "dte_max": {"type": "integer", "default": 10},
            "margin_requirement_pct": {"type": "number", "default": 1.0},
            "cash_buffer_pct": {"type": "number", "default": 0.10},
        },
        "explanation_template": "Wheel multi-stock plan with {params[capital_sek]} SEK.",
    },
    "calc_put_risk": {
        "tier": "calculator",
        "intent_description": "Analyze downside risk scenarios for a put position.",
        "intent_hints": {
            "keywords": [
                "risco do put", "risco opcao", "analisar risco",
                "put risk", "risk analysis", "downside risk", "cenarios",
            ],
            "examples": [
                "qual o risco do put na nordea?",
                "analyze put risk for aapl",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "pct_below_spot": {"type": "number", "default": 5.0, "description": "Percentage below spot for strike"},
        },
        "schema": {"required": ["symbol"]},
        "flow": SYMBOL_SEARCH_FLOW,
    },
    # ─── New Options Calculators (Calculator Tier) ───────────────────────
    "calc_required_premium": {
        "tier": "calculator",
        "intent_description": "Calculate required option premium for a target return percentage.",
        "intent_hints": {
            "keywords": [
                "premio necessario", "quanto de premio", "required premium",
                "target return", "premio minimo", "premium for return",
            ],
            "examples": [
                "quanto de premio preciso para 1.5% no strike 120?",
                "required premium for 2% return on strike 180",
            ],
        },
        "parameter_specs": {
            "strike": {"type": "number", "required": True},
            "target_return_pct": {"type": "number", "required": True},
            "days_to_expiry": {"type": "integer"},
        },
        "schema": {"required": ["strike", "target_return_pct"]},
    },
    "calc_income_target": {
        "tier": "calculator",
        "intent_description": "Calculate weekly/monthly income targets for a given capital and return goal.",
        "intent_hints": {
            "keywords": [
                "meta mensal", "meta semanal", "quanto ganhar", "income target",
                "por mes", "por semana", "renda mensal", "weekly income",
            ],
            "examples": [
                "quanto preciso ganhar por semana para ter 2% ao mes com 500000?",
                "income target for 24% annual with 100000 capital",
                "quero 1.5% ao mes com 300000, quanto por semana?",
            ],
        },
        "parameter_specs": {
            "capital": {"type": "number", "required": True},
            "target_monthly_pct": {"type": "number"},
            "target_annual_pct": {"type": "number"},
            "num_contracts": {"type": "integer"},
        },
        "schema": {"required": ["capital"]},
    },
    "calc_annualized_return": {
        "tier": "calculator",
        "intent_description": "Annualize a return from a specific period (e.g., weekly return to annual).",
        "intent_hints": {
            "keywords": [
                "anualizado", "anualizar", "annualized", "por ano",
                "retorno anual", "yearly return",
            ],
            "examples": [
                "1.2% em 7 dias anualizado da quanto?",
                "annualize 0.8% weekly return",
                "quanto da 2% por mes anualizado?",
            ],
        },
        "parameter_specs": {
            "return_pct": {"type": "number", "required": True},
            "period_days": {"type": "integer", "required": True},
            "num_periods": {"type": "integer"},
        },
        "schema": {"required": ["return_pct", "period_days"]},
    },
    "calc_margin_collateral": {
        "tier": "calculator",
        "intent_description": "Calculate total margin or collateral required for option positions.",
        "intent_hints": {
            "keywords": [
                "margem", "colateral", "margin", "collateral",
                "quanto de margem", "quanto preciso de colateral",
            ],
            "examples": [
                "quanto de colateral para 5 contratos no strike 120?",
                "margin required for 3 contracts at strike 180",
            ],
        },
        "parameter_specs": {
            "strike": {"type": "number", "required": True},
            "num_contracts": {"type": "integer", "default": 1},
            "margin_pct": {"type": "number", "default": 1.0},
        },
        "schema": {"required": ["strike"]},
    },
    # ─── Basic Finance Math (Calculator Tier) ────────────────────────────
    "calc_percentage": {
        "tier": "calculator",
        "intent_description": "Basic percentage calculation: X% of Y, or X is what % of Y.",
        "intent_hints": {
            "keywords": [
                "porcentagem", "percentual", "percentage", "quanto e",
                "qual o percentual", "percent of",
            ],
            "examples": [
                "quanto e 20% de 500000?",
                "150 e qual porcentagem de 1200?",
                "what is 15% of 80000?",
            ],
        },
        "parameter_specs": {
            "value": {"type": "number"},
            "percentage": {"type": "number"},
            "part": {"type": "number"},
            "whole": {"type": "number"},
        },
    },
    "calc_average_cost": {
        "tier": "calculator",
        "intent_description": "Calculate weighted average cost basis from multiple purchases.",
        "intent_hints": {
            "keywords": [
                "preco medio", "custo medio", "average cost", "cost basis",
                "preço médio", "media ponderada",
            ],
            "examples": [
                "comprei 100 a 120 e 50 a 115, qual meu preco medio?",
                "average cost of 200 shares at 50 and 100 shares at 48",
            ],
        },
        "parameter_specs": {
            "lots": {"type": "array", "required": True, "description": "List of {quantity, price} dicts"},
            "premium_received": {"type": "number", "default": 0},
        },
        "schema": {"required": ["lots"]},
    },
    "calc_compound_growth": {
        "tier": "calculator",
        "intent_description": "Calculate compound growth projection with optional periodic contributions.",
        "intent_hints": {
            "keywords": [
                "juros compostos", "crescimento composto", "compound growth",
                "compound interest", "quanto vira", "projecao",
            ],
            "examples": [
                "se ganho 1.5% por semana, 100000 vira quanto em 1 ano?",
                "compound 2% monthly on 500000 for 12 months",
            ],
        },
        "parameter_specs": {
            "principal": {"type": "number", "required": True},
            "rate_pct": {"type": "number", "required": True},
            "periods": {"type": "integer", "required": True},
            "contribution_per_period": {"type": "number", "default": 0},
        },
        "schema": {"required": ["principal", "rate_pct", "periods"]},
    },
    "calc_risk_reward": {
        "tier": "calculator",
        "intent_description": "Calculate Risk/Reward ratio given entry, stop loss, and target price.",
        "intent_hints": {
            "keywords": [
                "risk reward", "risco retorno", "R:R", "ratio risco",
                "risk:reward", "stop loss target",
            ],
            "examples": [
                "risk reward de entrada 120, stop 115, alvo 135?",
                "R:R ratio for entry 50, stop 47, target 60",
            ],
        },
        "parameter_specs": {
            "entry_price": {"type": "number", "required": True},
            "stop_loss": {"type": "number", "required": True},
            "target_price": {"type": "number", "required": True},
        },
        "schema": {"required": ["entry_price", "stop_loss", "target_price"]},
    },
    # ─── Analysis Tier (LLM + Skills) ──────────────────────────────────
    "analyze_stock": {
        "tier": "analysis",
        "intent_description": (
            "Deep stock analysis using LLM — covers technical, fundamental, risk, "
            "and comparative analysis. The LLM adapts focus based on the user's question."
        ),
        "intent_hints": {
            "keywords": [
                "analise tecnica", "análise técnica", "analise fundamentalista",
                "análise fundamentalista", "analise de risco", "análise de risco",
                "analise completa", "análise completa", "deep analysis",
                "quero uma analise", "faca uma analise", "comparar", "compare",
                "versus", "vs", "qual melhor",
            ],
            "examples": [
                "faca uma analise tecnica da petr4",
                "analise fundamentalista da vale3",
                "qual o risco de investir em tsla?",
                "compare petr4 com vale3",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of symbols to compare (optional)",
            },
        },
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "analyze_options": {
        "tier": "analysis",
        "intent_description": (
            "Deep options/derivatives analysis using LLM — covers wheel strategy, "
            "Greeks, volatility, and strategy construction. Adapts focus based on the question."
        ),
        "intent_hints": {
            "keywords": [
                "analise de opcoes", "análise de opções", "analise wheel",
                "análise wheel", "analise derivativos", "derivatives analysis",
                "wheel strategy analysis", "greeks analysis",
                "analise completa opcoes", "quero analise de opcoes",
            ],
            "examples": [
                "faca uma analise wheel da nordea",
                "analise das opcoes da aapl",
                "wheel strategy analysis for nda-se.st",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
            "market": {"type": "string", "default": "sweden"},
        },
        "flow": SYMBOL_SEARCH_FLOW,
    },
}


def _parameter_specs_from_schema(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    raw_required = schema.get("required")
    required = set()
    if isinstance(raw_required, list):
        required = {str(item).strip() for item in raw_required if str(item).strip()}
    if not isinstance(properties, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in properties.items():
        if not isinstance(raw_spec, dict):
            continue
        name = str(raw_name).strip()
        if not name:
            continue
        spec: dict[str, Any] = {}
        type_name = raw_spec.get("type")
        if isinstance(type_name, str) and type_name.strip():
            spec["type"] = type_name.strip()
        if name in required:
            spec["required"] = True
        if "default" in raw_spec:
            spec["default"] = raw_spec.get("default")
        enum_values = raw_spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            spec["enum"] = enum_values
        examples = raw_spec.get("examples")
        if isinstance(examples, list) and examples:
            spec["examples"] = examples
        if isinstance(raw_spec.get("description"), str) and raw_spec["description"].strip():
            spec["description"] = raw_spec["description"].strip()
        items = raw_spec.get("items")
        if isinstance(items, dict):
            item_spec: dict[str, Any] = {}
            item_type = items.get("type")
            if isinstance(item_type, str) and item_type.strip():
                item_spec["type"] = item_type.strip()
            item_enum = items.get("enum")
            if isinstance(item_enum, list) and item_enum:
                item_spec["enum"] = item_enum
            item_examples = items.get("examples")
            if isinstance(item_examples, list) and item_examples:
                item_spec["examples"] = item_examples
            if item_spec:
                spec["items"] = item_spec
        out[name] = spec
    return out


def _merge_parameter_specs(
    schema_specs: dict[str, dict[str, Any]],
    override_specs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for key, value in schema_specs.items():
        if isinstance(value, dict):
            merged[str(key)] = dict(value)

    if not isinstance(override_specs, dict):
        return merged

    for key, value in override_specs.items():
        name = str(key).strip()
        if not name:
            continue
        current = merged.setdefault(name, {})
        if isinstance(value, dict):
            current.update(value)
        else:
            current["default"] = value
    return merged

@app.get("/manifest")
def get_manifest():
    """
    Return capabilities dynamically fetched from MCP with rich metadata.
    """
    if not mcp_adapter:
        raise HTTPException(status_code=503, detail="MCP Adapter not initialized")
    
    # 1. Fetch raw tools from MCP (Source of Truth)
    # Using a cache or fetching live? For now, fetch live to be truly dynamic.
    try:
        raw_tools = mcp_adapter.list_tools()
    except Exception as e:
        logger.error("Failed to fetch tools from MCP: %s", e)
        # Fallback to empty or cached?
        raw_tools = []

    capabilities = []
    for tool in raw_tools:
        name = tool["name"]
        raw_schema = tool.get("schema")
        schema = raw_schema if isinstance(raw_schema, dict) else {}
        
        # 2. Merge with Metadata Overrides
        overrides = METADATA_OVERRIDES.get(name, {})
        merged_metadata = dict(overrides)
        schema_parameter_specs = _parameter_specs_from_schema(schema)
        override_parameter_specs = (
            overrides.get("parameter_specs")
            if isinstance(overrides.get("parameter_specs"), dict)
            else {}
        )
        merged_parameter_specs = _merge_parameter_specs(
            schema_specs=schema_parameter_specs,
            override_specs=override_parameter_specs,
        )
        if merged_parameter_specs:
            merged_metadata["parameter_specs"] = merged_parameter_specs
        merged_metadata["schema"] = schema
        merged_metadata.setdefault(
            "composition",
            {
                "followup_roles": ["notifier"],
                "enabled_if": {
                    "path": "parameters.notify",
                    "equals": True,
                },
                "followup_required": False,
                "followup_output_key": "notification",
            },
        )
        
        description_override = overrides.get("intent_description")
        description = str(description_override).strip() if isinstance(description_override, str) else ""
        if not description:
            description = tool["description"]

        capabilities.append({
            "name": name,
            "description": description,
            "schema": schema, # Raw JSON Schema from MCP
            "metadata": merged_metadata
        })

    return {
        "domain": "finance",
        "domain_description": "Market and financial data queries for stocks, rankings, fundamentals, options, and pipelines.",
        "domain_intent_hints": {
            "keywords": [
                "acao",
                "ações",
                "ticker",
                "preco",
                "preço",
                "valor da acao",
                "valor da ação",
                "cotacao",
                "cotação",
                "historico",
                "histórico",
                "bolsa",
                "bovespa",
                "ibov",
                "fundamentos",
                "dividendos",
                "opcoes",
                "opções",
                "wheel",
                "puts",
                "calls",
                "analistas",
                "sentimento",
            ],
            "examples": [
                "qual o valor da petr4?",
                "quais as maiores altas do bovespa hoje?",
                "historico de cotacao da vale3 em 1 ano",
                "wheel put candidates para nordea",
            ],
        },
        "goals": [
            {
                "goal": "GET_QUOTE",
                "description": "Get current stock price/quote for one or more symbols",
                "capabilities": ["get_stock_price"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "qual o preco", "qual o preço", "qual o valor",
                        "cotacao", "cotação", "quanto esta", "quanto está",
                        "ticker", "stock price", "price of",
                    ],
                    "examples": [
                        "qual o valor da petr4?",
                        "quanto esta a vale3 hoje?",
                        "price of aapl",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user (e.g. 'Nordea', 'PETR4', 'Apple')",
                    },
                    "symbols_text": {
                        "type": "array",
                        "description": "Multiple stock symbols/company names if comparing (e.g. ['Vale', 'Petrobras'])",
                    },
                },
            },
            {
                "goal": "VIEW_HISTORY",
                "description": "Get historical OHLC price data for a symbol and period",
                "capabilities": ["get_historical_data"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "historico", "histórico", "grafico", "gráfico",
                        "serie historica", "dados historicos", "preco nos ultimos",
                    ],
                    "examples": [
                        "historico da petr4 no ultimo mes",
                        "grafico de vale3 em 1 ano",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                    "period_text": {
                        "type": "string",
                        "description": "Time period as mentioned by user (e.g. 'ultimo mes', '1 ano', 'ytd')",
                    },
                    "interval_text": {
                        "type": "string",
                        "description": "Data interval as mentioned by user (e.g. 'diário', 'semanal')",
                    },
                },
            },
            {
                "goal": "SCREEN_STOCKS",
                "description": "Screen or filter stocks by market criteria",
                "capabilities": ["get_stock_screener"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "screener", "filtrar acoes", "filtrar ações",
                        "ranking de acoes", "acoes por setor",
                    ],
                    "examples": [
                        "screener de acoes no brasil",
                        "melhores acoes de tecnologia nos eua",
                    ],
                },
                "entities_schema": {
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user (e.g. 'brasil', 'eua', 'suécia')",
                    },
                    "sector_text": {
                        "type": "string",
                        "description": "Sector as mentioned by user (e.g. 'tecnologia', 'financeiro')",
                    },
                },
            },
            {
                "goal": "TOP_MOVERS",
                "description": "Find top gaining or losing stocks for a market/period",
                "capabilities": ["get_top_gainers", "get_top_losers"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "maiores altas", "maiores baixas", "maiores ganhos", "maiores perdas",
                        "top gainers", "top losers",
                        "acoes que mais subiram", "ações que mais subiram",
                        "acoes que mais cairam", "ações que mais caíram",
                        "mais altas hoje", "mais baixas hoje",
                        "ranking de altas", "ranking de baixas",
                    ],
                    "examples": [
                        "quais as maiores altas de hoje do bovespa?",
                        "top losers no brasil hoje",
                        "quais as maiores baixas de hoje?",
                    ],
                },
                "entities_schema": {
                    "direction": {
                        "type": "enum",
                        "values": ["GAINERS", "LOSERS", "BOTH"],
                        "required": True,
                        "default": "BOTH",
                        "description": "Whether to show top gainers, losers, or both",
                        "capability_map": {
                            "GAINERS": "get_top_gainers",
                            "LOSERS": "get_top_losers",
                            "BOTH": ["get_top_gainers", "get_top_losers"],
                        },
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user (e.g. 'bovespa', 'brasil', 'suécia')",
                    },
                    "period_text": {
                        "type": "string",
                        "description": "Time period as mentioned by user (e.g. 'hoje', 'semana', 'mês')",
                    },
                },
            },
            {
                "goal": "DIVIDEND_ANALYSIS",
                "description": "Find high dividend yield stocks or dividend history for a symbol",
                "capabilities": ["get_top_dividend_payers", "get_dividend_history"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "dividendos", "dividend yield", "maiores dividendos",
                        "acoes pagadoras", "ações pagadoras",
                        "historico de dividendos", "dividendos da",
                    ],
                    "examples": [
                        "quais acoes pagam mais dividendos no brasil?",
                        "dividendos da petr4",
                    ],
                },
                "entities_schema": {
                    "focus": {
                        "type": "enum",
                        "values": ["RANKING", "HISTORY"],
                        "required": True,
                        "default": "RANKING",
                        "description": "RANKING for top payers by market, HISTORY for a specific symbol's dividend history",
                        "capability_map": {
                            "RANKING": "get_top_dividend_payers",
                            "HISTORY": "get_dividend_history",
                        },
                    },
                    "symbol_text": {
                        "type": "string",
                        "description": "Company name or ticker as mentioned by user",
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user",
                    },
                },
            },
            {
                "goal": "TECHNICAL_SCAN",
                "description": "Find stocks with technical signals (RSI, MACD) by market",
                "capabilities": ["get_technical_signals"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "sinal tecnico", "sinal técnico", "rsi", "macd",
                        "sobrevendido", "sobrecomprado",
                    ],
                    "examples": [
                        "acoes com rsi sobrevendido no brasil",
                        "sinais macd no mercado americano",
                    ],
                },
            },
            {
                "goal": "FUNDAMENTALS",
                "description": "Get fundamental data for a specific stock",
                "capabilities": ["get_fundamentals"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "fundamentos", "fundamental", "valuation",
                        "balanco da empresa", "balanço",
                    ],
                    "examples": [
                        "fundamentos da vale3",
                        "valuation da petr4",
                    ],
                },
            },
            {
                "goal": "COMPANY_PROFILE",
                "description": "Get company profile and business information",
                "capabilities": ["get_company_profile"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "sobre a empresa", "company info", "setor da empresa",
                        "perfil da empresa",
                    ],
                    "examples": [
                        "me fale sobre a empresa vale3",
                        "company info da aapl",
                    ],
                },
            },
            {
                "goal": "FINANCIAL_STATEMENTS",
                "description": "Get financial statements (DRE, balance sheet)",
                "capabilities": ["get_financial_statements"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "demonstracoes financeiras", "financial statements",
                        "dre", "balanco patrimonial",
                    ],
                    "examples": [
                        "demonstracoes financeiras da vale3",
                        "financial statements de msft",
                    ],
                },
            },
            {
                "goal": "OPTIONS_DATA",
                "description": "Get option chain or Greeks for a stock",
                "capabilities": ["get_option_chain", "get_option_greeks"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "option chain", "cadeia de opcoes", "opcoes da acao",
                        "opções da ação", "greeks", "delta gamma",
                        "gregas das opcoes", "gregas das opções",
                    ],
                    "examples": [
                        "option chain da aapl",
                        "gregas das opcoes de petr4",
                    ],
                },
                "entities_schema": {
                    "focus": {
                        "type": "enum",
                        "values": ["CHAIN", "GREEKS"],
                        "required": True,
                        "default": "CHAIN",
                        "description": "CHAIN for option chain, GREEKS for option Greeks (delta, gamma, etc.)",
                        "capability_map": {
                            "CHAIN": "get_option_chain",
                            "GREEKS": "get_option_greeks",
                        },
                    },
                    "symbol_text": {
                        "type": "string",
                        "description": "Company name or ticker as mentioned by user",
                    },
                },
            },
            {
                "goal": "SEARCH_SYMBOL",
                "description": "Search ticker symbols by company name",
                "capabilities": ["search_by_name", "yahoo_search"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "qual o ticker", "procurar ticker", "buscar simbolo",
                        "buscar símbolo",
                    ],
                    "examples": [
                        "qual o ticker da petrobras?",
                        "buscar simbolo da nordea",
                    ],
                },
                "entities_schema": {
                    "query_text": {
                        "type": "string",
                        "required": True,
                        "description": "Company name or keyword to search for (e.g. 'nordea', 'petrobras')",
                    },
                },
            },
            {
                "goal": "PIPELINE_STATUS",
                "description": "Check data pipeline jobs and status",
                "capabilities": ["list_jobs", "get_job_status"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "listar jobs", "jobs disponiveis", "jobs disponíveis",
                        "pipeline jobs", "status do job", "job status",
                    ],
                    "examples": [
                        "quais jobs de dados estao disponiveis?",
                        "qual o status do job earnings_sync?",
                    ],
                },
                "entities_schema": {
                    "focus": {
                        "type": "enum",
                        "values": ["LIST", "STATUS"],
                        "required": True,
                        "default": "LIST",
                        "description": "LIST to enumerate available jobs, STATUS for a specific job's health",
                        "capability_map": {
                            "LIST": "list_jobs",
                            "STATUS": "get_job_status",
                        },
                    },
                    "job_name_text": {
                        "type": "string",
                        "description": "Job name as mentioned by user (e.g. 'earnings_sync')",
                    },
                },
            },
            # ─── Expanded Screeners ───────────────────────────────────
            {
                "goal": "ACTIVE_STOCKS",
                "description": "Find most actively traded stocks by volume for a market",
                "capabilities": ["get_most_active_stocks"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "mais negociadas", "mais ativas", "volume alto",
                        "most active", "mais liquidez",
                    ],
                    "examples": [
                        "quais as acoes mais negociadas hoje?",
                        "most active stocks in sweden",
                    ],
                },
                "entities_schema": {
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user",
                    },
                    "period_text": {
                        "type": "string",
                        "description": "Period as mentioned by user (e.g. 'hoje', 'semana')",
                    },
                },
            },
            {
                "goal": "OVERSOLD_STOCKS",
                "description": "Find oversold stocks (low RSI) for a market",
                "capabilities": ["get_oversold_stocks"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "sobrevendido", "oversold", "rsi baixo",
                        "oportunidade de compra",
                    ],
                    "examples": [
                        "acoes sobrevendidas no brasil",
                        "oversold stocks in sweden",
                    ],
                },
                "entities_schema": {
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user",
                    },
                },
            },
            {
                "goal": "OVERBOUGHT_STOCKS",
                "description": "Find overbought stocks (high RSI) for a market",
                "capabilities": ["get_overbought_stocks"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "sobrecomprado", "overbought", "rsi alto",
                        "excessivamente comprado",
                    ],
                    "examples": [
                        "acoes sobrecompradas no brasil",
                        "overbought stocks in usa",
                    ],
                },
                "entities_schema": {
                    "market_text": {
                        "type": "string",
                        "description": "Market name as mentioned by user",
                    },
                },
            },
            # ─── Fundamentals Intelligence ────────────────────────────
            {
                "goal": "ANALYST_VIEW",
                "description": "Get analyst consensus recommendations for a stock",
                "capabilities": ["get_analyst_recommendations"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "recomendacao", "recomendação", "analistas",
                        "consenso", "buy sell hold", "analyst recommendation",
                    ],
                    "examples": [
                        "qual a recomendacao dos analistas para petr4?",
                        "analyst recommendations for aapl",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                },
            },
            {
                "goal": "TECH_ANALYSIS",
                "description": "Get comprehensive technical analysis indicators for a stock",
                "capabilities": ["get_technical_analysis"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "analise tecnica", "análise técnica", "indicadores tecnicos",
                        "technical analysis", "tecnica completa",
                    ],
                    "examples": [
                        "analise tecnica da vale3",
                        "technical analysis of aapl",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                    "period_text": {
                        "type": "string",
                        "description": "Analysis period as mentioned by user (e.g. '1 ano', '6 meses')",
                    },
                },
            },
            {
                "goal": "NEWS_SENTIMENT",
                "description": "Get news sentiment analysis for a stock",
                "capabilities": ["get_news_sentiment"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "sentimento", "noticias", "notícias",
                        "news sentiment", "humor do mercado",
                    ],
                    "examples": [
                        "qual o sentimento das noticias sobre petr4?",
                        "news sentiment for tsla",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                },
            },
            {
                "goal": "STOCK_DEEP_DIVE",
                "description": "Get comprehensive information about a stock (price, fundamentals, technicals, analyst views)",
                "capabilities": ["get_comprehensive_stock_info"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "tudo sobre", "analise completa", "análise completa",
                        "deep dive", "informacoes completas", "visao geral",
                        "resumo completo",
                    ],
                    "examples": [
                        "me fale tudo sobre a petr4",
                        "deep dive in aapl",
                        "analise completa da vale3",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                },
            },
            # ─── Wheel Strategy ───────────────────────────────────────
            {
                "goal": "WHEEL_PUT_SCAN",
                "description": "Find put option candidates for the wheel strategy",
                "capabilities": ["get_wheel_put_candidates"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "wheel puts", "puts para wheel", "candidatos wheel",
                        "vender puts", "sell puts", "put candidates",
                    ],
                    "examples": [
                        "quais os melhores puts para wheel na nordea?",
                        "wheel put candidates for aapl",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name as mentioned by user",
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market as mentioned by user (e.g. 'suécia', 'brasil')",
                    },
                },
            },
            {
                "goal": "WHEEL_PUT_RETURN",
                "description": "Calculate return for a specific wheel put position",
                "capabilities": ["get_wheel_put_return"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "retorno do put", "rendimento put", "quanto vou ganhar",
                        "put return", "wheel return",
                    ],
                    "examples": [
                        "qual o retorno do put de nordea strike 120?",
                        "wheel put return for aapl strike 180",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "strike_text": {
                        "type": "string",
                        "description": "Strike price as mentioned by user",
                    },
                    "expiry_text": {
                        "type": "string",
                        "description": "Expiry date as mentioned by user",
                    },
                    "premium_text": {
                        "type": "string",
                        "description": "Premium received as mentioned by user",
                    },
                },
            },
            {
                "goal": "WHEEL_CALL_SCAN",
                "description": "Find covered call candidates for the wheel strategy after assignment",
                "capabilities": ["get_wheel_covered_call_candidates"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "covered calls", "calls cobertas", "wheel calls",
                        "vender calls", "call candidates",
                    ],
                    "examples": [
                        "covered calls para nordea com custo medio de 120?",
                        "wheel call candidates for aapl cost basis 180",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "average_cost_text": {
                        "type": "string",
                        "description": "Average cost basis as mentioned by user",
                    },
                },
            },
            {
                "goal": "WHEEL_CAPACITY",
                "description": "Calculate how many wheel contracts you can sell given capital",
                "capabilities": ["get_wheel_contract_capacity"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "quantos contratos", "capacidade capital", "quanto posso vender",
                        "contract capacity", "capital allocation",
                    ],
                    "examples": [
                        "quantos contratos posso vender de nordea com 100000 sek?",
                        "wheel capacity for aapl with 50000",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "capital_text": {
                        "type": "string",
                        "description": "Capital amount as mentioned by user",
                    },
                },
            },
            {
                "goal": "WHEEL_MULTI_PLAN",
                "description": "Build a diversified wheel strategy plan across multiple stocks",
                "capabilities": ["build_wheel_multi_stock_plan"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "plano wheel multiplos", "carteira wheel", "distribuir capital",
                        "multi stock plan", "diversificar wheel",
                    ],
                    "examples": [
                        "monte um plano wheel com 500000 sek",
                        "build wheel plan across multiple stocks",
                    ],
                },
                "entities_schema": {
                    "capital_text": {
                        "type": "string",
                        "required": True,
                        "description": "Capital amount as mentioned by user",
                    },
                    "symbols_text": {
                        "type": "array",
                        "description": "List of specific symbols to consider (optional)",
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market as mentioned by user",
                    },
                },
            },
            {
                "goal": "WHEEL_RISK",
                "description": "Analyze risk of a wheel put position (downside scenarios)",
                "capabilities": ["analyze_wheel_put_risk"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "risco do put", "risco wheel", "analisar risco",
                        "put risk", "wheel risk analysis", "downside risk",
                    ],
                    "examples": [
                        "qual o risco do wheel put na nordea?",
                        "analyze wheel risk for aapl",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market as mentioned by user",
                    },
                },
            },
            # ─── Analysis Goals (LLM Tier) ─────────────────────────────
            {
                "goal": "ANALYZE_STOCK",
                "description": "Deep stock analysis (technical, fundamental, risk, comparison) using LLM",
                "capabilities": ["analyze_stock"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "analise tecnica", "análise técnica",
                        "analise fundamentalista", "análise fundamentalista",
                        "analise de risco", "análise de risco",
                        "analise completa", "análise completa",
                        "deep analysis", "quero uma analise",
                        "faca uma analise", "comparar", "compare",
                        "versus", "vs", "qual melhor",
                    ],
                    "examples": [
                        "faca uma analise tecnica da petr4",
                        "analise fundamentalista da vale3",
                        "qual o risco de investir em tsla?",
                        "compare petr4 com vale3",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "symbols_text": {
                        "type": "array",
                        "description": "Multiple symbols for comparison (optional)",
                    },
                },
            },
            {
                "goal": "ANALYZE_OPTIONS",
                "description": "Deep options/derivatives analysis (wheel, greeks, strategies) using LLM",
                "capabilities": ["analyze_options"],
                "requires_domains": [],
                "hints": {
                    "keywords": [
                        "analise de opcoes", "análise de opções",
                        "analise wheel", "análise wheel",
                        "analise derivativos", "derivatives analysis",
                        "wheel strategy analysis", "greeks analysis",
                        "analise completa opcoes",
                    ],
                    "examples": [
                        "faca uma analise wheel da nordea",
                        "analise das opcoes da aapl",
                        "derivatives analysis for petr4",
                    ],
                },
                "entities_schema": {
                    "symbol_text": {
                        "type": "string",
                        "required": True,
                        "description": "Stock symbol or company name",
                    },
                    "market_text": {
                        "type": "string",
                        "description": "Market as mentioned by user",
                    },
                },
            },
        ],
        "capabilities": capabilities
    }

@app.post("/execute", response_model=DomainOutput)
async def execute_intent(intent: ExecutionIntent):
    """
    Standard Domain Protocol Execution Endpoint.
    Receives ExecutionIntent -> Returns DomainOutput.
    """
    if not handler:
        raise HTTPException(status_code=503, detail="Handler not initialized")
    
    try:
        return await handler.execute(intent)
    except Exception as e:
        logger.error("Execution error: %s", e, exc_info=True)
        return DomainOutput(
            status="failure",
            explanation=f"Server error: {str(e)}",
            metadata={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
