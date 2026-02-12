import asyncio
import logging
import json
import os
from registry.http_handler import HttpDomainHandler

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("manifest_test")

def test_manifest_sync():
    mcp_url = os.getenv("MCP_URL", "http://localhost:8001") # Note: Port 8001 is the HTTP server, not SSE
    logger.info(f"Testing manifest fetch from: {mcp_url}")
    
    handler = HttpDomainHandler(base_url=mcp_url)
    
    try:
        manifest = handler.fetch_manifest()
        logger.info("✅ Manifest fetched successfully!")
        
        # Check specific capability schema
        for cap in manifest.get("capabilities", []):
            if cap["name"] == "get_top_gainers":
                logger.info(f"Found 'get_top_gainers' schema: {json.dumps(cap.get('schema'), indent=2)}")
                break
        else:
            logger.warning("❌ 'get_top_gainers' NOT found in manifest.")
            
    except Exception as e:
        logger.error(f"💥 Failed to fetch manifest: {e}")

if __name__ == "__main__":
    test_manifest_sync()
