import logging
import os

import pytest

from skills.implementations.mcp_adapter import MCPAdapter

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp_test")

def test_mcp_connection():
    mcp_url = os.getenv("MCP_URL", "http://localhost:8000/sse")
    logger.info(f"Testing connection to MCP at: {mcp_url}")

    adapter = MCPAdapter(mcp_url=mcp_url)
    test_params = {
        "_action": "get_stock_price",
        "symbol": "AAPL"
    }

    try:
        result = adapter.execute(test_params)
    finally:
        adapter.close()

    if not result.get("success"):
        pytest.skip(f"MCP unavailable for integration test: {result.get('error')}")

    assert result.get("success") is True
    assert "data" in result

if __name__ == "__main__":
    test_mcp_connection()
