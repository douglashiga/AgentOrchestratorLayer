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

from shared.models import IntentOutput, DomainOutput
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

    # Pass symbol_resolver with production aliases to handler
    handler = FinanceDomainHandler(
        skill_gateway=skill_gateway,
        registry=mock_registry,
        symbol_resolver=symbol_resolver,
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
    "SEMANA": "5d",
    "SEMANAL": "5d",
    "MÊS": "1mo",
    "MES": "1mo",
    "MENSAL": "1mo",
    "TRIMESTRE": "3mo",
    "ANO": "1y",
    "ANUAL": "1y",
}

HISTORICAL_PERIOD_ALIASES = {
    "HOJE": "1d",
    "1 DIA": "1d",
    "5 DIAS": "5d",
    "SEMANA": "5d",
    "1 MÊS": "1mo",
    "1 MES": "1mo",
    "MÊS": "1mo",
    "MES": "1mo",
    "3 MESES": "3mo",
    "6 MESES": "6mo",
    "1 ANO": "1y",
    "2 ANOS": "2y",
    "5 ANOS": "5y",
    "YTD": "ytd",
    "MAX": "max",
}

INTERVAL_ALIASES = {
    "DIARIO": "1d",
    "DIÁRIO": "1d",
    "DIA": "1d",
    "SEMANAL": "1wk",
    "SEMANA": "1wk",
    "MENSAL": "1mo",
    "MES": "1mo",
    "MÊS": "1mo",
}

SIGNAL_TYPE_ALIASES = {
    "RSI OVERSOLD": "rsi_oversold",
    "RSI SOBREVENDIDO": "rsi_oversold",
    "SOBREVENDIDO": "rsi_oversold",
    "RSI OVERBOUGHT": "rsi_overbought",
    "RSI SOBRECOMPRADO": "rsi_overbought",
    "SOBRECOMPRADO": "rsi_overbought",
    "MACD CROSS": "macd_cross",
    "CRUZAMENTO MACD": "macd_cross",
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
            "search_fallback_capabilities": ["search_symbol"],
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
            "search_fallback_capabilities": ["search_symbol"],
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
                    "param_name": "symbols",
                    "single_param_name": "symbol",
                    "max_concurrency": 4,
                }
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
                "examples": ["technology", "financials", "energy"],
            },
            "sort_by": {
                "type": "string",
                "default": "market_cap",
                "aliases": {
                    "VALOR DE MERCADO": "market_cap",
                    "MARKET CAP": "market_cap",
                    "P/L": "pe_ratio",
                    "DY": "dividend_yield",
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
    "compare_fundamentals": {
        "intent_description": (
            "Compare fundamental metrics for multiple stocks. "
            "Use when user asks to compare companies by valuation/profitability."
        ),
        "intent_hints": {
            "keywords": [
                "compare fundamentos",
                "comparar fundamentos",
                "comparar acoes",
                "comparar ações",
                "valuation comparativo",
            ],
            "examples": [
                "compare os fundamentos de vale3 e petr4",
                "comparar aapl e msft por pe ratio e roe",
            ],
        },
        "symbols_required": True,
        "parameter_specs": {
            "symbols": {
                "type": "array",
                "required": True,
                "items": {
                    "type": "string",
                    "normalization": {"case": "upper"},
                    "aliases": SYMBOL_ALIASES,
                },
                "examples": [["VALE3.SA", "PETR4.SA"]],
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "normalization": {"case": "lower"},
                    "aliases": {
                        "P/L": "pe_ratio",
                        "PE": "pe_ratio",
                        "ROE": "roe",
                        "MARGEM": "net_margin",
                    },
                },
                "examples": [["pe_ratio", "roe"]],
            },
        },
        "explanation_template": "Fundamental comparison.",
        "flow": SYMBOL_LIST_SEARCH_FLOW,
        "decomposition": {
            "array_params": [
                {
                    "param_name": "symbols",
                    "single_param_name": "symbol",
                    "max_concurrency": 4,
                }
            ]
        },
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
    "get_dividends": {
        "intent_description": "Get dividend history for a specific stock symbol.",
        "intent_hints": {
            "keywords": ["dividendos da", "historico de dividendos", "dividend history"],
            "examples": ["dividendos da petr4", "historico de dividendos da vale3"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Dividend history for {symbol} ({currency}).",
    },
    "get_company_info": {
        "intent_description": "Get company profile and business information for a symbol.",
        "intent_hints": {
            "keywords": ["sobre a empresa", "company info", "setor da empresa", "perfil da empresa"],
            "examples": ["me fale sobre a empresa vale3", "company info da aapl"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Company info for {result[name]}.",
    },
    "get_option_chain": {
        "intent_description": "Get option chain for a stock symbol.",
        "intent_hints": {
            "keywords": ["option chain", "cadeia de opcoes", "opcoes da acao", "opções da ação"],
            "examples": ["option chain da aapl", "cadeia de opcoes da petr4"],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
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
    "get_account_summary": {
        "intent_description": "Get brokerage account summary.",
        "intent_hints": {
            "keywords": ["resumo da conta", "account summary", "saldo da conta"],
            "examples": ["meu account summary"],
        },
        "explanation_template": "Account summary retrieved.",
    },
    "search_symbol": {
        "intent_description": "Search ticker symbols by company name or keyword.",
        "intent_hints": {
            "keywords": ["qual o ticker", "procurar ticker", "buscar simbolo", "buscar símbolo"],
            "examples": ["qual o ticker da petrobras?", "buscar simbolo da nordea"],
        },
        "parameter_specs": {
            "query": {
                "type": "string",
                "required": True,
                "examples": ["petrobras", "nordea"],
            }
        },
        "explanation_template": "Search results for '{params[query]}'.",
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
            ],
            "examples": [
                "qual o valor da petr4?",
                "quais as maiores altas do bovespa hoje?",
                "compare os fundamentos de vale3 e petr4",
                "historico de cotacao da vale3 em 1 ano",
            ],
        },
        "capabilities": capabilities
    }

@app.post("/execute", response_model=DomainOutput)
async def execute_intent(intent: IntentOutput):
    """
    Standard Domain Protocol Execution Endpoint.
    Receives IntentOutput -> Returns DomainOutput.
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
