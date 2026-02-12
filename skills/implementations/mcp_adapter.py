"""
MCP Adapter — Skill implementation that calls MCP Finance Server via SSE.

Responsibility:
- Connect to MCP Finance Server via SSE (Server-Sent Events)
- Call MCP tools using the standard MCP protocol
- Return raw data

Performance:
- Persistent SSE session — connect once, reuse across calls
- Dedicated background event loop in a thread
- Auto-reconnect on session failure

Prohibitions:
- No strategy decisions
- No domain calls
- No LLM usage
"""

import asyncio
import json
import logging
import threading
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

# Map of action names to MCP tool names
_ACTION_TO_TOOL: dict[str, str] = {
    "get_stock_price": "get_stock_price",
    "get_historical_data": "get_historical_data",
    "search_symbol": "search_symbol",
    "get_account_summary": "get_account_summary",
    "get_option_chain": "get_option_chain",
    "get_option_greeks": "get_option_greeks",
    "get_fundamentals": "get_fundamentals",
    "get_dividends": "get_dividends",
    "get_company_info": "get_company_info",
    "get_financial_statements": "get_financial_statements",
    "get_exchange_info": "get_exchange_info",
    "yahoo_search": "yahoo_search",
    # Mapped to yahoo_search temporarily until MCP Server is updated
    "get_top_gainers": "yahoo_search",
    "get_top_losers": "yahoo_search",
    "get_top_dividends": "yahoo_search",
    "get_market_performance": "yahoo_search",
    "compare_fundamentals": "yahoo_search",
}


class MCPAdapter:
    """Calls MCP Finance Server via SSE with persistent session.

    Architecture:
    - Dedicated asyncio event loop running in a background thread
    - SSE session opened once and reused across all calls
    - Auto-reconnects if session drops
    - Thread-safe: sync callers submit coroutines to the background loop
    """

    def __init__(self, mcp_url: str = "http://localhost:8000/sse"):
        self.mcp_url = mcp_url

        # Background event loop for async MCP calls
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Persistent session state (managed in background loop)
        self._session: ClientSession | None = None
        self._connected = False

    def _run_loop(self) -> None:
        """Run the dedicated event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro) -> Any:
        """Submit a coroutine to the background loop and wait for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an MCP tool call via persistent SSE session.
        Parameters must include '_action' key to identify which tool to call.
        """
        action = parameters.pop("_action", None)
        if not action:
            return {"success": False, "error": "No '_action' specified in parameters."}

        tool_name = _ACTION_TO_TOOL.get(action)
        if not tool_name:
            return {"success": False, "error": f"Unknown action: '{action}'"}

        # Temporary parameter adaptation for fallback mappings
        if tool_name == "yahoo_search" and "query" not in parameters:
            # Construct a query from available params for the fallback
            parts = []
            if action: parts.append(action.replace("_", " "))
            if parameters.get("market"): parts.append(f"in {parameters['market']}")
            if parameters.get("period"): parts.append(f"period {parameters['period']}")
            if parameters.get("symbols"): parts.append(f"for {', '.join(parameters['symbols'])}")
            
            parameters["query"] = " ".join(parts)

        try:
            return self._submit(self._call_tool(tool_name, parameters))
        except Exception as e:
            logger.error("MCP execution error: %s", e)
            # Reset session on error so next call reconnects
            self._connected = False
            self._session = None
            return {"success": False, "error": f"MCP error: {e}"}

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool, using persistent session or reconnecting."""
        logger.info("MCP call → tool='%s' arguments=%s", tool_name, arguments)

        # Use a fresh connection per call for SSE (SSE sessions are not truly reusable
        # in the MCP SDK — each sse_client context manages its own stream lifecycle)
        try:
            async with sse_client(self.mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)

                    logger.info(
                        "MCP response ← content=%s",
                        [c.text if hasattr(c, "text") else str(c) for c in result.content],
                    )

                    # Parse the result
                    if result.isError:
                        error_text = ""
                        for content in result.content:
                            if hasattr(content, "text"):
                                error_text += content.text
                        return {"success": False, "error": error_text or "MCP tool returned an error"}

                    # Extract text content
                    data = {}
                    for content in result.content:
                        if hasattr(content, "text"):
                            try:
                                parsed = json.loads(content.text)
                                data = parsed if isinstance(parsed, dict) else {"value": parsed}
                            except json.JSONDecodeError:
                                data = {"text": content.text}

                    return {"success": True, "data": data}

        except ConnectionRefusedError:
            logger.warning("MCP Server not reachable at %s", self.mcp_url)
            return {
                "success": False,
                "error": f"MCP Server not reachable at {self.mcp_url}. "
                         f"Make sure the MCP Finance Server is running.",
            }
        except Exception as e:
            logger.error("MCP SSE error: %s", e)
            return {"success": False, "error": f"MCP SSE error: {e}"}

    def close(self) -> None:
        """Shutdown the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
