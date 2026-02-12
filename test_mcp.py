import asyncio
import logging
import json
import os
from skills.implementations.mcp_adapter import MCPAdapter

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp_test")

async def test_mcp_connection():
    mcp_url = os.getenv("MCP_URL", "http://localhost:8000/sse")
    logger.info(f"Testing connection to MCP at: {mcp_url}")
    
    adapter = MCPAdapter(mcp_url=mcp_url)
    
    # Vamos tentar uma ferramenta simples: get_stock_price para AAPL
    test_params = {
        "_action": "get_stock_price",
        "symbol": "AAPL"
    }
    
    logger.info("Calling 'get_stock_price' for AAPL...")
    try:
        # A execução do adapter.execute é síncrona pois ele gerencia seu próprio loop em uma thread
        result = adapter.execute(test_params)
        
        if result.get("success"):
            logger.info("✅ SUCCESS! MCP is connected and working.")
            logger.info(f"Result Data: {json.dumps(result.get('data'), indent=2)}")
        else:
            logger.error("❌ FAILED! MCP returned an error.")
            logger.error(f"Error Detail: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR during test: {e}")
    finally:
        adapter.close()

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
