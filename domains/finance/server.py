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
from skills.gateway import SkillGateway
from skills.registry import SkillRegistry
from skills.implementations.mcp_adapter import MCPAdapter

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_server")

# Environment
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/sse")

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

    # Initialize ModelSelector for specialist queries (optional — degrades gracefully).
    specialist_model_selector = None
    try:
        from models.selector import ModelSelector
        specialist_model_selector = ModelSelector(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
    except Exception as e:
        logger.warning("ModelSelector not available for specialist queries: %s", e)

    handler = FinanceDomainHandler(
        skill_gateway=skill_gateway,
        registry=mock_registry,
        model_selector=specialist_model_selector,
    )

    yield
    
    logger.info("Shutting down Finance Server...")
    if mcp_adapter:
        mcp_adapter.close()

app = FastAPI(title="Finance Domain Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Static Metadata Overrides (Templates & Rules)
# This enriches the raw MCP tools with Orchestrator-specific UI/Logic hints.
MARKET_ALIASES = {
    "BOVESPA": "BR",
    "IBOVESPA": "BR",
    "IBOV": "BR",
    "B3": "BR",
    "BRASIL": "BR",
    "BRAZIL": "BR",
    "NYSE": "US",
    "NASDAQ": "US",
    "EUA": "US",
    "USA": "US",
    "US": "US",
    "SUECIA": "SE",
    "SUÉCIA": "SE",
    "SWEDEN": "SE",
    "STOCKHOLM": "SE",
    "HONG KONG": "HK",
    "HONGKONG": "HK",
    "HKEX": "HK",
}

SYMBOL_ALIASES = {
    "PETRO": "PETR4.SA",
    "PETRO4": "PETR4.SA",
    "PETROBRAS": "PETR4.SA",
    "VALE": "VALE3.SA",
    "ITAU": "ITUB4.SA",
    "ITAU UNIBANCO": "ITUB4.SA",
    "BBAS": "BBAS3.SA",
    "BANCO DO BRASIL": "BBAS3.SA",
    "MGLU": "MGLU3.SA",
    "MAGAZINE LUIZA": "MGLU3.SA",
    "BBDC": "BBDC4.SA",
    "BRADESCO": "BBDC4.SA",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "NORDEA": "NDA-SE.ST",
}

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
                "quanto custa",
                "ticker",
                "stock price",
                "preco da acao",
                "preço da ação",
                "valor da acao",
                "valor da ação",
                "valor de mercado",
                "como esta a",
                "como está a",
                "me diga o valor",
                "me fala o preco",
                "current price",
                "quote",
            ],
            "examples": [
                "qual o valor da petr4?",
                "quanto esta a vale3 hoje?",
                "price of aapl",
                "me diga o valor da petrobras",
                "como esta a vale3?",
                "quanto custa a acao da itau?",
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
                "source_params": ["symbol", "symbols"],
                "infer_from_suffix": {
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
                "source_params": ["symbol", "symbols"],
                "infer_from_suffix": {
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
        "flow": SYMBOL_SEARCH_FLOW,
    },
    "get_historical_data": {
        "intent_description": (
            "Get historical OHLC price data for a symbol and period. "
            "Use for price history, chart, time-series, and performance-over-time requests."
        ),
        "intent_hints": {
            "keywords": [
                "historico da",
                "histórico da",
                "historico de cotacao",
                "histórico de cotação",
                "historico de preco",
                "histórico de preço",
                "grafico",
                "gráfico",
                "serie historica",
                "série histórica",
                "dados historicos",
                "dados históricos",
                "preco nos ultimos",
                "preço nos últimos",
                "evolucao da",
                "evolução da",
                "variacao da",
                "variação da",
                "desempenho da",
                "nos ultimos dias",
                "nos últimos dias",
                "ultimo mes",
                "último mês",
                "ultimo ano",
                "último ano",
                "historical data",
                "chart",
            ],
            "examples": [
                "historico da petr4 no ultimo mes",
                "grafico de vale3 em 1 ano",
                "como a petr4 se comportou nos ultimos 3 meses?",
                "evolucao da aapl neste ano",
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
                "ranking de ações",
                "acoes por setor",
                "ações por setor",
                "melhores acoes",
                "melhores ações",
                "listar acoes",
                "listar ações",
                "quais acoes",
                "quais ações",
                "stock screener",
                "buscar acoes",
                "buscar ações",
            ],
            "examples": [
                "screener de acoes no brasil",
                "melhores acoes de tecnologia nos eua",
                "quais acoes do setor financeiro no brasil?",
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
                "altas do dia",
                "quem mais subiu",
                "acoes em alta",
                "ações em alta",
                "valorizaram mais",
                "maiores valorizacoes",
                "maiores valorizações",
            ],
            "examples": [
                "quais as maiores altas de hoje do bovespa?",
                "top gainers no brasil hoje",
                "quais acoes mais subiram essa semana?",
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
                "baixas do dia",
                "quem mais caiu",
                "acoes em baixa",
                "ações em baixa",
                "desvalorizaram mais",
                "maiores quedas",
            ],
            "examples": [
                "quais as maiores baixas de hoje do bovespa?",
                "top losers no brasil hoje",
                "quais acoes mais cairam essa semana?",
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
                "melhores dividendos",
                "ranking dividendos",
                "dy",
                "maiores pagadoras",
                "renda passiva",
                "acoes de renda",
                "ações de renda",
            ],
            "examples": [
                "quais acoes pagam mais dividendos no brasil?",
                "top dividendos nos eua",
                "melhores acoes pagadoras de dividendos na bovespa",
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
            "Compare fundamental metrics for multiple stocks side by side. "
            "Use ONLY when user explicitly asks to compare two or more companies."
        ),
        "intent_hints": {
            "keywords": [
                "compare fundamentos",
                "comparar fundamentos",
                "comparar acoes",
                "comparar ações",
                "valuation comparativo",
                "comparar empresas",
                "diferenca entre",
                "diferença entre",
                "versus fundamentos",
                "qual melhor entre",
            ],
            "examples": [
                "compare os fundamentos de vale3 e petr4",
                "comparar aapl e msft por pe ratio e roe",
                "qual a diferenca entre vale3 e petr4 em fundamentos?",
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
        "intent_description": (
            "Get fundamental analysis data for a SINGLE stock symbol. "
            "Use when user asks about indicators, valuation, or fundamentals of one specific stock."
        ),
        "intent_hints": {
            "keywords": [
                "fundamentos da",
                "fundamentos de",
                "fundamental da",
                "valuation da",
                "valuation de",
                "balanco da empresa",
                "balanço da empresa",
                "indicadores da",
                "indicadores de",
                "p/l da",
                "pe ratio da",
                "roe da",
                "margem da",
                "ebitda da",
                "multiplos da",
                "múltiplos da",
                "analise fundamentalista da",
                "análise fundamentalista da",
            ],
            "examples": [
                "fundamentos da vale3",
                "valuation da petr4",
                "quais os indicadores da aapl?",
                "qual o p/l da petr4?",
            ],
        },
        "parameter_specs": {
            "symbol": dict(SYMBOL_PARAM_SPEC),
        },
        "flow": SYMBOL_SEARCH_FLOW,
        "explanation_template": "Fundamentals data for {symbol} ({market} market).",
    },
    "get_dividends": {
        "intent_description": (
            "Get dividend history for a SPECIFIC stock symbol. "
            "Use when user mentions a specific ticker/company and asks about dividends."
        ),
        "intent_hints": {
            "keywords": [
                "dividendos da",
                "dividendos de",
                "historico de dividendos",
                "histórico de dividendos",
                "dividend history",
                "proventos da",
                "proventos de",
                "jcp da",
                "juros sobre capital da",
                "pagamentos de dividendos da",
                "quando paga dividendos",
            ],
            "examples": [
                "dividendos da petr4",
                "historico de dividendos da vale3",
                "quando a itub4 paga dividendos?",
            ],
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
            "keywords": [
                "sobre a empresa",
                "company info",
                "setor da empresa",
                "perfil da empresa",
                "informacoes da empresa",
                "informações da empresa",
                "o que faz a empresa",
                "quem e a empresa",
                "quem é a empresa",
                "detalhes da empresa",
                "company profile",
            ],
            "examples": [
                "me fale sobre a empresa vale3",
                "company info da aapl",
                "o que a petrobras faz?",
            ],
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
    "ask_finance_expert": {
        "type": "specialist",
        "intent_description": (
            "Answer open-ended finance knowledge questions about strategies, concepts, "
            "analysis methodologies, derivatives, options strategies, and market theory. "
            "Use when the user asks HOW something works or WHAT a concept means in finance."
        ),
        "intent_hints": {
            "keywords": [
                "como funciona",
                "o que e",
                "o que é",
                "me explica",
                "me explique",
                "qual a diferenca entre",
                "qual a diferença entre",
                "estrategia de",
                "estratégia de",
                "the wheel",
                "covered call",
                "cash secured put",
                "iron condor",
                "straddle",
                "strangle",
                "analise tecnica",
                "análise técnica",
                "analise fundamentalista",
                "análise fundamentalista",
                "o que significa",
                "como calcular",
                "como analisar",
                "quando usar",
                "para que serve",
                "como interpretar",
                "o que sao",
                "o que são",
                "como investir",
                "como operar",
                "como montar",
                "suporte e resistencia",
                "suporte e resistência",
                "media movel",
                "média móvel",
                "bandas de bollinger",
                "fibonacci",
                "elliott",
                "candlestick",
                "grafico de velas",
                "gráfico de velas",
                "risco e retorno",
                "gestao de risco",
                "gestão de risco",
                "how does",
                "what is",
                "explain",
            ],
            "examples": [
                "como funciona o the wheel?",
                "me explica covered call",
                "o que e analise tecnica?",
                "qual a diferenca entre analise tecnica e fundamentalista?",
                "como funciona iron condor?",
                "o que significa rsi sobrevendido?",
                "como interpretar bandas de bollinger?",
                "como calcular o preco justo de uma acao?",
                "como montar uma estrategia de dividendos?",
                "para que serve o macd?",
            ],
        },
        "parameter_specs": {
            "question": {
                "type": "string",
                "required": True,
                "description": "The user's finance question",
            },
        },
        "specialist_config": {
            "default_system_prompt": (
                "You are a knowledgeable finance expert. Answer clearly and concisely "
                "in the user's language. Use examples when helpful."
            ),
            "temperature": 0.4,
            "timeout_seconds": 45,
            "experts": [
                {
                    "id": "technical_analysis",
                    "system_prompt": (
                        "You are an expert in technical analysis of financial markets. "
                        "You have deep knowledge of chart patterns, indicators (RSI, MACD, "
                        "Bollinger Bands, Moving Averages, Fibonacci, Elliott Waves), "
                        "support/resistance levels, volume analysis, and price action. "
                        "Explain concepts clearly with practical examples. "
                        "Answer in the user's language."
                    ),
                    "topics": [
                        "analise tecnica", "análise técnica", "technical analysis",
                        "rsi", "macd", "bollinger", "media movel", "média móvel",
                        "moving average", "fibonacci", "elliott", "candlestick",
                        "suporte e resistencia", "suporte e resistência",
                        "support and resistance", "volume", "price action",
                        "grafico", "gráfico", "chart", "tendencia", "tendência",
                        "indicador", "indicator", "overbought", "oversold",
                        "sobrecomprado", "sobrevendido",
                    ],
                    "data_capabilities": ["get_historical_data", "get_technical_signals"],
                },
                {
                    "id": "fundamental_analysis",
                    "system_prompt": (
                        "You are an expert in fundamental analysis of companies and stocks. "
                        "You have deep knowledge of financial statements, valuation metrics "
                        "(P/E, P/B, ROE, ROA, EV/EBITDA, dividend yield), DCF models, "
                        "competitive analysis, and intrinsic value calculation. "
                        "Explain concepts clearly with practical examples. "
                        "Answer in the user's language."
                    ),
                    "topics": [
                        "analise fundamentalista", "análise fundamentalista",
                        "fundamental analysis", "valuation", "p/l", "p/e",
                        "roe", "roa", "ebitda", "margem", "margin",
                        "balanco", "balanço", "balance sheet", "dre",
                        "fluxo de caixa", "cash flow", "dividendo", "dividend",
                        "preco justo", "preço justo", "fair value", "intrinsic value",
                        "valor intrinseco", "valor intrínseco", "multiplos", "múltiplos",
                        "lucro", "receita", "revenue", "earnings",
                    ],
                    "data_capabilities": ["get_fundamentals", "get_financial_statements"],
                },
                {
                    "id": "derivatives",
                    "system_prompt": (
                        "You are an expert in financial derivatives, especially stock options. "
                        "You have deep knowledge of options strategies (covered calls, "
                        "cash-secured puts, the wheel, iron condors, straddles, strangles, "
                        "spreads, butterflies), Greeks (delta, gamma, theta, vega), "
                        "implied volatility, and risk management for derivatives. "
                        "Explain concepts clearly with practical examples. "
                        "Answer in the user's language."
                    ),
                    "topics": [
                        "opcao", "opção", "opcoes", "opções", "option", "options",
                        "derivativo", "derivative", "covered call", "cash secured put",
                        "the wheel", "iron condor", "straddle", "strangle",
                        "spread", "butterfly", "collar", "protective put",
                        "greeks", "gregas", "delta", "gamma", "theta", "vega",
                        "volatilidade", "volatility", "implied volatility",
                        "strike", "exercicio", "exercício", "vencimento",
                        "expiration", "premium", "premio", "prêmio",
                    ],
                    "data_capabilities": ["get_option_chain", "get_option_greeks"],
                },
                {
                    "id": "general_finance",
                    "system_prompt": (
                        "You are a knowledgeable finance expert covering investment strategies, "
                        "portfolio management, risk management, asset allocation, "
                        "market structure, and financial concepts. "
                        "Explain concepts clearly with practical examples. "
                        "Answer in the user's language."
                    ),
                    "topics": [
                        "investimento", "investment", "carteira", "portfolio",
                        "diversificacao", "diversificação", "diversification",
                        "risco", "risk", "retorno", "return",
                        "alocacao", "alocação", "allocation",
                        "renda fixa", "fixed income", "renda variavel", "renda variável",
                        "etf", "fundo", "fund", "cdi", "selic", "ipca",
                        "estrategia", "estratégia", "strategy",
                    ],
                    "data_capabilities": [],
                },
            ],
        },
        "explanation_template": "{result[response]}",
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
                "triggering_parameter": "notify",
                "enabled_if": {
                    "path": "parameters.notify",
                    "equals": True,
                },
                "multi_instance": {
                    "list_param": "symbols",
                    "single_param": "symbol",
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

    # 3. Add non-MCP capabilities declared only in METADATA_OVERRIDES
    #    (e.g. specialist/knowledge capabilities that don't map to an MCP tool).
    mcp_tool_names = {tool["name"] for tool in raw_tools}
    for override_name, overrides in METADATA_OVERRIDES.items():
        if override_name in mcp_tool_names:
            continue  # Already handled above
        merged_metadata = dict(overrides)
        override_parameter_specs = (
            overrides.get("parameter_specs")
            if isinstance(overrides.get("parameter_specs"), dict)
            else {}
        )
        if override_parameter_specs:
            merged_metadata["parameter_specs"] = _merge_parameter_specs(
                schema_specs={},
                override_specs=override_parameter_specs,
            )
        merged_metadata.setdefault("schema", {})
        description = str(overrides.get("intent_description", "")).strip() or override_name
        capabilities.append({
            "name": override_name,
            "description": description,
            "schema": merged_metadata.get("schema", {}),
            "metadata": merged_metadata,
        })

    return {
        "domain": "finance",
        "domain_description": "Market and financial data queries for stocks, rankings, fundamentals, options, and pipelines.",
        "domain_intent_hints": {
            "keywords": [
                "acao",
                "ação",
                "acoes",
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
                "b3",
                "fundamentos",
                "dividendos",
                "opcoes",
                "opções",
                "mercado",
                "mercado financeiro",
                "investimento",
                "investimentos",
                "carteira",
                "portfolio",
                "ativo",
                "ativos",
                "renda variavel",
                "renda variável",
                "stock",
                "stocks",
                "market",
                "finance",
            ],
            "examples": [
                "qual o valor da petr4?",
                "quais as maiores altas do bovespa hoje?",
                "compare os fundamentos de vale3 e petr4",
                "historico de cotacao da vale3 em 1 ano",
                "me fale sobre os dividendos da itub4",
                "quanto custa a acao da petrobras?",
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
