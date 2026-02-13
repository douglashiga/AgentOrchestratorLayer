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
from pydantic import BaseModel

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
        mock_registry.register_capability(cap["name"], None, metadata=cap.get("metadata", {}))

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
        "default_market": "SE",
        "explanation_template": "{symbol} is trading at {result[price]} {currency}."
    },
    "get_historical_data": {
        "explanation_template": "Historical data for {symbol} ({params[duration]})."
    },
    "get_stock_screener": {
        "market_required": True,
        "default_market": "SE",
        "valid_values": {"market": ["US", "BR", "SE", "HK"]},
        "explanation_template": "Stock screener results for {result[market]} market."
    },
    "get_top_gainers": {
        "market_required": True,
        "default_market": "SE",
        "default_period": "1d",
        "valid_values": {"market": ["US", "BR", "SE"], "period": ["1d", "5d", "1mo", "3mo", "1y"]},
        "explanation_template": "Top gainers in {result[market]} for {result[period]}."
    },
    "get_top_losers": {
        "market_required": True,
        "default_market": "SE",
        "default_period": "1d",
        "valid_values": {"market": ["US", "BR", "SE"], "period": ["1d", "5d", "1mo", "3mo", "1y"]},
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
        "explanation_template": "Fundamental comparison."
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
        
        capabilities.append({
            "name": name,
            "description": tool["description"],
            "schema": tool["schema"], # Raw JSON Schema from MCP
            "metadata": overrides
        })

    return {
        "domain": "finance",
        "capabilities": capabilities
    }

@app.post("/execute", response_model=DomainOutput)
def execute_intent(intent: IntentOutput):
    """
    Standard Domain Protocol Execution Endpoint.
    Receives IntentOutput -> Returns DomainOutput.
    """
    if not handler:
        raise HTTPException(status_code=503, detail="Handler not initialized")
    
    try:
        # FinanceHandler is synchronous, FastAPI runs this in a threadpool
        return handler.execute(intent)
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
