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
import concurrent.futures
import json
import logging
import os
import threading
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

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
        self.call_timeout_seconds = float(os.getenv("MCP_ADAPTER_CALL_TIMEOUT_SECONDS", "90"))

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
        try:
            return future.result(timeout=self.call_timeout_seconds)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"MCP call timed out after {self.call_timeout_seconds:.0f}s "
                f"(endpoint={self.mcp_url})"
            )

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an MCP tool call via persistent SSE session.
        Parameters must include '_action' key to identify which tool to call.
        
        Dynamic Resolution:
        - Uses '_action' as the tool name directly (Pass-Through).
        - Allows optional '_tool_map' in parameters for legacy translations.
        """
        action = parameters.pop("_action", None)
        if not action:
            return {"success": False, "error": "No '_action' specified in parameters."}

        # Dynamic Tool Resolution (Direct Mapping)
        tool_map = parameters.pop("_tool_map", {})
        tool_name = tool_map.get(action, action)

        try:
            return self._submit(self._call_tool(tool_name, parameters))
        except Exception as e:
            logger.error("MCP execution error: %r", e)
            # Reset session on error so next call reconnects
            self._connected = False
            self._session = None
            return {"success": False, "error": f"MCP error ({type(e).__name__}): {self._format_error_message(e)}"}

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
                        error_text_parts: list[str] = []
                        for content in result.content:
                            if hasattr(content, "text") and content.text:
                                error_text_parts.append(content.text)
                        error_text = " ".join(error_text_parts).strip()
                        return {"success": False, "error": error_text or "MCP tool returned an error"}

                    # Extract text content
                    data = {}
                    for content in result.content:
                        if hasattr(content, "text"):
                            try:
                                parsed = json.loads(content.text)
                                # Normalize MCP envelope: {"success": bool, "data": ..., "error": ...}
                                if isinstance(parsed, dict) and "success" in parsed:
                                    if parsed.get("success") is False:
                                        return {
                                            "success": False,
                                            "error": self._extract_embedded_error(parsed),
                                            "data": parsed.get("data"),
                                            "meta": parsed.get("meta"),
                                        }
                                    data = parsed
                                else:
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
            # Handle ExceptionGroup (common in mcp TaskGroups) to get the real cause
            error_detail = self._format_error_message(e)
            if hasattr(e, "exceptions") and e.exceptions:
                # Python 3.11+ ExceptionGroup or anyio ExceptionGroup
                sub_errors = [self._format_error_message(se) for se in e.exceptions]
                error_detail = f"{e} (Sub-errors: {', '.join(sub_errors)})"
                logger.error("MCP SSE TaskGroup failure: %s", error_detail)
            else:
                logger.error("MCP SSE error: %s", e, exc_info=True)
                
            return {"success": False, "error": f"MCP SSE error: {error_detail}"}

    def _extract_embedded_error(self, parsed: dict[str, Any]) -> str:
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            code = str(error_obj.get("code", "")).strip()
            message = str(error_obj.get("message", "")).strip()
            details = str(error_obj.get("details", "")).strip()
            parts = [part for part in (code, message, details) if part]
            if parts:
                return " | ".join(parts)
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()
        return "MCP tool reported failure without details"

    def _format_error_message(self, error: Exception) -> str:
        text = str(error).strip()
        if text:
            return text
        return error.__class__.__name__

    def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools from the MCP server.
        Returns a list of dicts with 'name', 'description', and 'schema'.
        """
        try:
            return self._submit(self._fetch_tools_async())
        except Exception as e:
            logger.error("Failed to list MCP tools: %s", e)
            return []

    async def _fetch_tools_async(self) -> list[dict[str, Any]]:
        """Async implementation of list_tools."""
        logger.info("Fetching tool list from MCP at %s...", self.mcp_url)
        try:
            async with sse_client(self.mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    
                    tools = []
                    for tool in result.tools:
                        tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema
                        })
                    
                    logger.info("Fetched %d tools from MCP", len(tools))
                    return tools
        except Exception as e:
            logger.error("Error fetching tools from MCP: %s", e)
            raise
    def close(self) -> None:
        """Shutdown the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
