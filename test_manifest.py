import logging
import os

import pytest

from registry.http_handler import HttpDomainHandler

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("manifest_test")

def test_manifest_sync():
    mcp_url = os.getenv("MCP_URL", "http://localhost:8001")  # Port 8001 is the HTTP server, not SSE
    logger.info(f"Testing manifest fetch from: {mcp_url}")

    handler = HttpDomainHandler(base_url=mcp_url)

    try:
        manifest = handler.fetch_manifest()
    except Exception as e:
        pytest.skip(f"Finance domain unavailable for integration test: {e}")

    assert manifest.get("domain") == "finance"
    assert isinstance(manifest.get("capabilities"), list)
    assert any(cap.get("name") == "get_top_gainers" for cap in manifest["capabilities"])

if __name__ == "__main__":
    test_manifest_sync()
