"""
Calculator Tier — Local deterministic math + optional MCP data fetching.

Responsibility:
- Resolve calculator function for capability
- Fetch any required data from MCP via DataFetcher
- Execute local math (no LLM)
- Return DomainOutput with calculated results

Calculators are simple async functions: (params, DataFetcher) → dict
"""

import logging
from typing import Any, Awaitable, Callable

from shared.models import DomainOutput
from domains.finance.tiers.base import TierContext, TierProcessor
from skills.gateway import SkillGateway

logger = logging.getLogger(__name__)

# Calculator function signature: (params, DataFetcher) → dict
CalculatorFunction = Callable[[dict[str, Any], "DataFetcher"], Awaitable[dict[str, Any]]]


class DataFetcher:
    """
    Facade for calculators to fetch data from MCP.

    Provides a simple interface so calculator functions don't
    need to know about SkillGateway or MCP details.
    """

    def __init__(self, skill_gateway: SkillGateway):
        self._gateway = skill_gateway

    def fetch(self, capability: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Fetch data from MCP for a given capability.

        Args:
            capability: MCP capability name (e.g. "get_stock_price", "get_option_chain")
            params: Parameters to pass to the capability

        Returns:
            Raw MCP response dict with 'success' and 'data' or 'error'
        """
        skill_params = {
            **{k: v for k, v in params.items() if v is not None},
            "_action": capability,
        }
        result = self._gateway.execute("mcp_finance", skill_params)
        logger.debug("DataFetcher.fetch(%s): success=%s", capability, result.get("success"))
        return result

    def fetch_data(self, capability: str, params: dict[str, Any]) -> Any:
        """
        Fetch data and return only the data payload (or raise on error).

        Args:
            capability: MCP capability name
            params: Parameters to pass

        Returns:
            The 'data' portion of the MCP response

        Raises:
            RuntimeError: If MCP call failed
        """
        result = self.fetch(capability, params)
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"MCP call failed for {capability}: {error}")
        return result.get("data", {})


class CalculatorRegistry:
    """
    Registry of calculator functions by capability name.

    Each calculator is registered with the capability it handles.
    Multiple capabilities can map to the same calculator function.
    """

    def __init__(self):
        self._calculators: dict[str, CalculatorFunction] = {}

    def register(self, capability: str, calculator: CalculatorFunction) -> None:
        """Register a calculator for a capability."""
        self._calculators[capability] = calculator
        logger.info("Registered calculator: %s", capability)

    def resolve(self, capability: str) -> CalculatorFunction | None:
        """Find the calculator for a capability."""
        return self._calculators.get(capability)

    def list_capabilities(self) -> list[str]:
        """List all registered calculator capabilities."""
        return list(self._calculators.keys())


class CalculatorTier:
    """
    Tier 2: Calculator — Local deterministic math.

    Resolves the calculator function for the capability,
    creates a DataFetcher for optional MCP data access,
    and runs the math locally.
    """

    def __init__(
        self,
        calculator_registry: CalculatorRegistry,
        skill_gateway: SkillGateway,
    ):
        self._registry = calculator_registry
        self._data_fetcher = DataFetcher(skill_gateway)

    async def process(self, context: TierContext) -> DomainOutput:
        """
        Execute calculator pipeline:
        1. Resolve calculator function
        2. Run calculator with params + DataFetcher
        3. Return DomainOutput
        """
        capability = context.intent.capability
        params = dict(context.params)

        calculator = self._registry.resolve(capability)
        if calculator is None:
            logger.warning("No calculator registered for: %s", capability)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"No calculator implementation for '{capability}'.",
                confidence=0.0,
                metadata={"error": "calculator_not_found"},
            )

        try:
            result = await calculator(params, self._data_fetcher)

            if not isinstance(result, dict):
                result = {"value": result}

            return DomainOutput(
                status="success",
                result=result,
                explanation=result.get("explanation", f"Calculated {capability}."),
                confidence=1.0,
                metadata={"tier": "calculator", "capability": capability},
            )

        except RuntimeError as e:
            # MCP data fetch failures surface as RuntimeError from DataFetcher
            error_msg = str(e)
            logger.error("Calculator data fetch failed for %s: %s", capability, error_msg)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Não consegui obter os dados necessários: {error_msg}",
                confidence=0.0,
                metadata={"error": error_msg, "tier": "calculator"},
            )

        except Exception as e:
            logger.error("Calculator failed for %s: %s", capability, e, exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Erro no cálculo de '{capability}': {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "tier": "calculator"},
            )
