"""
MCP Adapter — Skill implementation that calls MCP Finance Server via SSE.

Responsibility:
- Connect to MCP Finance Server via SSE (Server-Sent Events)
- Call MCP tools using the standard MCP protocol
- Return raw data

Prohibitions:
- No strategy decisions
- No domain calls
- No LLM usage
"""

import asyncio
import logging
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
}


class MCPAdapter:
    """Calls MCP Finance Server via SSE to execute finance skills."""

    def __init__(self, mcp_url: str = "http://localhost:8000/sse"):
        self.mcp_url = mcp_url

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an MCP tool call via SSE.
        Parameters must include '_action' key to identify which tool to call.
        """
        action = parameters.pop("_action", None)
        if not action:
            return {"success": False, "error": "No '_action' specified in parameters."}

        tool_name = _ACTION_TO_TOOL.get(action)
        if not tool_name:
            return {"success": False, "error": f"Unknown action: '{action}'"}

        # Run async MCP call in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self._call_mcp_tool(tool_name, parameters)
                    ).result(timeout=30)
                return result
            else:
                return loop.run_until_complete(self._call_mcp_tool(tool_name, parameters))
        except RuntimeError:
            return asyncio.run(self._call_mcp_tool(tool_name, parameters))
        except Exception as e:
            logger.error("MCP execution error: %s", e)
            return {"success": False, "error": f"MCP error: {e}"}

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via SSE transport."""
        logger.info("MCP call → tool='%s' arguments=%s", tool_name, arguments)
        try:
            async with sse_client(self.mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the MCP session
                    await session.initialize()

                    # Call the tool
                    result = await session.call_tool(tool_name, arguments=arguments)
                    logger.info("MCP response ← content=%s", [c.text if hasattr(c, 'text') else str(c) for c in result.content])

                    # Parse the result
                    if result.isError:
                        error_text = ""
                        for content in result.content:
                            if hasattr(content, "text"):
                                error_text += content.text
                        return {"success": False, "error": error_text or "MCP tool returned an error"}

                    # Extract text content from result
                    data = {}
                    for content in result.content:
                        if hasattr(content, "text"):
                            # Try to parse as JSON
                            import json
                            try:
                                parsed = json.loads(content.text)
                                if isinstance(parsed, dict):
                                    data = parsed
                                else:
                                    data = {"value": parsed}
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
