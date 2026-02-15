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

    handler = FinanceDomainHandler(skill_gateway=skill_gateway, registry=mock_registry)
    
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
METADATA_OVERRIDES = {
    "get_stock_price": {
        "intent_description": (
            "Get current stock price/quote for one or more symbols. "
            "Use for price questions such as 'qual o preço', 'cotação', 'quanto está', and ticker/company mentions."
        ),
        "intent_hints": {
            "keywords": [
                "qual o preco",
                "qual o preço",
                "cotacao",
                "cotação",
                "quanto está",
                "quanto esta",
                "valor da acao",
                "price",
                "stock price",
            ],
            "examples": [
                "qual o preco da petro?",
                "cotação da vale3 hoje",
            ],
        },
        "explanation_template": "{symbol} is trading at {result[price]} {currency}.",
        "parameter_specs": {
            "currency": {
                "infer_from_symbol_suffix": {
                    ".SA": "BRL",
                    ".ST": "SEK",
                },
                "default": "USD",
                "value_labels": {
                    "USD": "dólar",
                    "BRL": "real",
                    "SEK": "coroa sueca",
                },
            },
            "exchange": {
                "infer_from_symbol_suffix": {
                    ".SA": "BOVESPA",
                    ".ST": "SFB",
                },
                "default": "SMART",
                "value_labels": {
                    "SMART": "mercado dos EUA",
                    "BOVESPA": "Bovespa",
                    "SFB": "bolsa da Suécia",
                },
            },
        },
        "flow": {
            "pre": [
                {
                    "type": "resolve_symbol",
                    "param": "symbol",
                    "search_capability": "yahoo_search",
                    "search_fallback_capabilities": ["search_symbol"],
                }
            ]
        },
    },
    "get_historical_data": {
        "explanation_template": "Historical data for {symbol} ({params[period]}).",
        "flow": {
            "pre": [
                {
                    "type": "resolve_symbol",
                    "param": "symbol",
                    "search_capability": "yahoo_search",
                    "search_fallback_capabilities": ["search_symbol"],
                }
            ]
        },
    },
    "get_stock_screener": {
        "market_required": True,
        "default_market": "SE",
        "valid_values": {"market": ["US", "BR", "SE", "HK"]},
        "explanation_template": "Stock screener results for {result[market]} market."
    },
    "get_top_gainers": {
        "intent_description": (
            "List top gainers for a market/period. Use for ranking/momentum queries such as "
            "'maiores altas', 'top gainers', 'ações que mais subiram', including Bovespa/IBOV context."
        ),
        "intent_hints": {
            "keywords": [
                "maiores altas",
                "top gainers",
                "ações que mais subiram",
                "acoes que mais subiram",
                "bovespa",
                "ibovespa",
                "ibov",
                "mais altas",
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
            "market": {
                "type": "string",
                "required": True,
                "examples": ["BR", "US", "SE"],
                "normalization": {"case": "upper"},
                "aliases": {
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
                    "SUECIA": "SE",
                    "SUÉCIA": "SE",
                    "SWEDEN": "SE",
                    "STOCKHOLM": "SE",
                },
            },
            "period": {
                "type": "string",
                "default": "1d",
                "examples": ["1d", "5d", "1mo"],
                "aliases": {
                    "HOJE": "1d",
                    "TODAY": "1d",
                    "SEMANA": "5d",
                    "MÊS": "1mo",
                    "MES": "1mo",
                    "ANO": "1y",
                },
            },
        },
        "explanation_template": "Top gainers in {result[market]} for {result[period]}."
    },
    "get_top_losers": {
        "intent_description": (
            "List top losers for a market/period. Use for ranking/momentum queries such as "
            "'maiores baixas', 'top losers', 'ações que mais caíram', including Bovespa/IBOV context."
        ),
        "intent_hints": {
            "keywords": [
                "maiores baixas",
                "top losers",
                "ações que mais caíram",
                "acoes que mais cairam",
                "bovespa",
                "ibovespa",
                "ibov",
                "mais baixas",
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
            "market": {
                "type": "string",
                "required": True,
                "examples": ["BR", "US", "SE"],
                "normalization": {"case": "upper"},
                "aliases": {
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
                    "SUECIA": "SE",
                    "SUÉCIA": "SE",
                    "SWEDEN": "SE",
                    "STOCKHOLM": "SE",
                },
            },
            "period": {
                "type": "string",
                "default": "1d",
                "examples": ["1d", "5d", "1mo"],
                "aliases": {
                    "HOJE": "1d",
                    "TODAY": "1d",
                    "SEMANA": "5d",
                    "MÊS": "1mo",
                    "MES": "1mo",
                    "ANO": "1y",
                },
            },
        },
        "explanation_template": "Top losers in {result[market]} for {result[period]}."
    },
    "get_top_dividend_payers": {
        "market_required": True,
        "default_market": "SE",
        "valid_values": {"market": ["US", "BR", "SE"]},
        "explanation_template": "High yield stocks in {result[market]}."
    },
    "get_technical_signals": {
        "market_required": True,
        "signal_required": True,
        "default_market": "SE",
        "default_signal_type": "rsi_oversold", 
        "valid_values": {"market": ["US", "BR", "SE"], "signal_type": ["rsi_oversold", "rsi_overbought", "macd_cross"]},
        "explanation_template": "Stocks showing {params[signal_type]} signals in {result[market]}."
    },
    "compare_fundamentals": {
        "symbols_required": True,
        "explanation_template": "Fundamental comparison.",
        "flow": {
            "pre": [
                {
                    "type": "resolve_symbol_list",
                    "param": "symbols",
                    "search_capability": "yahoo_search",
                    "search_fallback_capabilities": ["search_symbol"],
                }
            ]
        },
    },
    "list_jobs": {
        "explanation_template": "Data pipeline job list."
    },
    "get_job_status": {
        "explanation_template": "Pipeline health status overview."
    },
    "get_fundamentals": {
        "explanation_template": "Fundamentals data for {symbol} ({market} market)."
    },
    "get_dividends": {
        "explanation_template": "Dividend history for {symbol} ({currency})."
    },
    "get_company_info": {
        "explanation_template": "Company info for {result[name]}."
    },
    "get_option_chain": {
        "explanation_template": "Option chain for {symbol}."
    },
    "get_option_greeks": {
        "explanation_template": "Option Greeks for {symbol}."
    },
    "get_financial_statements": {
        "explanation_template": "Financial statements for {symbol}."
    },
    "get_account_summary": {
        "explanation_template": "Account summary retrieved."
    },
    "search_symbol": {
        "explanation_template": "Search results for '{params[query]}'."
    },
    "yahoo_search": {
        "explanation_template": "Yahoo search results for '{params[query]}'."
    }
}

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
        
        # 2. Merge with Metadata Overrides
        overrides = METADATA_OVERRIDES.get(name, {})
        merged_metadata = dict(overrides)
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
            "schema": tool["schema"], # Raw JSON Schema from MCP
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
                "cotacao",
                "cotação",
                "bolsa",
                "bovespa",
                "ibov",
                "mercado",
                "fundamentos",
                "dividendos",
            ],
            "examples": [
                "qual o valor da petr4?",
                "quais as maiores altas do bovespa hoje?",
                "compare os fundamentos de vale3 e petr4",
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
