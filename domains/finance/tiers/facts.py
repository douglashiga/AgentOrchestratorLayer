"""
Facts Tier — MCP passthrough for data retrieval.

Responsibility:
- Resolve context (market detection from symbol)
- Call MCP via SkillGateway
- Format result via StrategyCore
- Return DomainOutput

This is a direct extraction of the _run_pipeline logic from handler.py.
Zero behavior change — same code path, just encapsulated in a tier processor.
"""

import logging
from typing import Any

from shared.models import Decision, DomainOutput, ExecutionContext, ExecutionIntent
from domains.finance.tiers.base import TierContext, TierProcessor
from domains.finance.context import ContextResolver
from domains.finance.core import StrategyCore
from skills.gateway import SkillGateway

logger = logging.getLogger(__name__)


class FactsTier:
    """
    Tier 1: Facts — Pure MCP data retrieval + StrategyCore formatting.

    Extracted from FinanceDomainHandler._run_pipeline.
    All existing capabilities route through this tier by default.
    """

    def __init__(
        self,
        skill_gateway: SkillGateway,
        context_resolver: ContextResolver,
        strategy_core: StrategyCore,
        registry: Any = None,
    ):
        self._gateway = skill_gateway
        self._context_resolver = context_resolver
        self._strategy_core = strategy_core
        self._registry = registry

    async def process(self, context: TierContext) -> DomainOutput:
        """
        Execute the facts pipeline:
        1. Context Resolution (market detection)
        2. Skill Execution (MCP call)
        3. Strategy Core (formatting)
        4. Output Generation
        """
        capability = context.intent.capability
        params = dict(context.params)
        metadata = context.metadata

        try:
            # ─── 1. Context Resolution ───────────────────────────────────
            multi_contexts = None
            if params.get("symbol"):
                domain_context = self._context_resolver.resolve(params["symbol"])
            elif params.get("symbols") and isinstance(params["symbols"], list) and params["symbols"]:
                domain_context = self._context_resolver.resolve(params["symbols"][0])
                if len(params["symbols"]) > 1:
                    multi_contexts = self._context_resolver.resolve_multiple(params["symbols"])
            elif params.get("market"):
                domain_context = self._context_resolver.get_market_profile(params["market"])
                if not domain_context:
                    domain_context = self._context_resolver.resolve("DEFAULT")
            else:
                domain_context = self._context_resolver.resolve("DEFAULT")

            logger.info("Facts tier context: %s (%s)", domain_context.market, domain_context.currency)

            # ─── 2. Skill Execution ──────────────────────────────────────
            sanitized_params = {k: v for k, v in params.items() if v is not None}
            skill_params = {
                **sanitized_params,
                "_action": capability,
            }

            # Inject Tool Map from Metadata
            if metadata:
                skill_params["_tool_map"] = metadata.get("tool_map", {})

            # Inject Context if needed
            if not skill_params.get("market") and domain_context.market != "US":
                skill_params["market"] = domain_context.market

            # Execute via Gateway
            skill_data = self._gateway.execute("mcp_finance", skill_params)
            logger.debug("Facts tier skill data: success=%s", skill_data.get("success"))

            # Stock price fallback (cache)
            if capability == "get_stock_price" and not skill_data.get("success", False):
                fallback = self._fallback_stock_price(params, domain_context, str(skill_data.get("error", "")))
                if fallback:
                    return fallback

            # ─── 3. Strategy Core ────────────────────────────────────────
            execution_context = ExecutionContext(
                domain_context=domain_context,
                skill_data=skill_data,
                multi_contexts=multi_contexts,
            )

            intent_for_decision = context.intent.model_copy(update={"parameters": params})
            decision = self._strategy_core.execute(
                intent_for_decision, execution_context, registry=self._registry
            )

            # Map operational errors to clarification
            clarification = self._map_operational_error(capability, decision)
            if clarification is not None:
                return clarification

            # ─── 4. Output Generation ────────────────────────────────────
            output_metadata = {"risk_metrics": decision.risk_metrics}
            if decision.error:
                output_metadata["error"] = decision.error

            return DomainOutput(
                status="success" if decision.success else "failure",
                result=decision.result,
                explanation=decision.explanation,
                confidence=1.0,
                metadata=output_metadata,
            )

        except Exception as e:
            logger.error("Facts tier failed for %s: %s", capability, e, exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing finance action: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    # ─── Helpers (extracted from handler) ────────────────────────────

    def _fallback_stock_price(
        self,
        params: dict[str, Any],
        domain_context: Any,
        original_error: str,
    ) -> DomainOutput | None:
        """Try cached historical data as fallback for stock price."""
        symbol = str(params.get("symbol", "")).strip()
        if not symbol:
            return None

        attempts = [symbol]
        if "." in symbol:
            attempts.append(symbol.split(".", 1)[0])

        seen: set[str] = set()
        ordered: list[str] = []
        for item in attempts:
            norm = item.strip().upper()
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)

        for candidate in ordered:
            trial = self._gateway.execute(
                "mcp_finance",
                {
                    "_action": "get_historical_data_cached",
                    "symbol": candidate,
                    "period": "1mo",
                    "interval": "1d",
                },
            )
            if not trial.get("success", False):
                continue

            rows = self._extract_cached_rows(trial.get("data"))
            if not rows:
                continue

            last = rows[-1] if isinstance(rows[-1], dict) else {}
            close = last.get("close", last.get("price"))
            if close in (None, ""):
                continue
            date = last.get("date") or last.get("datetime")

            result = {
                "symbol": symbol,
                "price": close,
                "currency": domain_context.currency,
                "date": date,
                "source": "cache_fallback",
            }
            return DomainOutput(
                status="success",
                result=result,
                explanation=f"{symbol} está em {close} {domain_context.currency} (cache local).",
                confidence=0.9,
                metadata={"source": "cache_fallback", "original_error": original_error},
            )

        return None

    def _extract_cached_rows(self, payload: Any) -> list[dict[str, Any]]:
        """Extract rows from heterogeneous cache responses."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return [item for item in payload.get("data", []) if isinstance(item, dict)]
            inner = payload.get("data")
            if isinstance(inner, dict) and isinstance(inner.get("data"), list):
                return [item for item in inner.get("data", []) if isinstance(item, dict)]
        return []

    def _map_operational_error(
        self,
        capability: str,
        decision: Decision,
    ) -> DomainOutput | None:
        """Map known operational errors to user-friendly clarification messages."""
        if decision.success:
            return None
        error_text = str(decision.error or "").strip().lower()
        if not error_text:
            return None

        operational_markers = (
            "ib_not_connected",
            "not connected to ib gateway",
            "timeout",
            "timed out",
            "network",
            "temporarily unavailable",
            "circuit_open",
            "tool_not_found",
            "runtime_error",
            "stock '",
            "job '",
            "not found",
            "no upcoming earnings",
        )
        if not any(marker in error_text for marker in operational_markers):
            return None

        if "stock '" in error_text and "not found" in error_text:
            message = (
                f"Não encontrei dados para o ativo em '{capability}'. "
                "Confirme o ticker/mercado para eu tentar novamente."
            )
        elif "job '" in error_text and "not found" in error_text:
            message = (
                "O job informado não existe no momento. "
                "Use list_jobs para ver os jobs disponíveis."
            )
        elif "circuit_open" in error_text:
            message = (
                f"A fonte de dados de '{capability}' está temporariamente indisponível "
                "(circuit breaker). Tente novamente em instantes."
            )
        else:
            message = (
                f"O serviço de dados para '{capability}' está temporariamente indisponível "
                "(fonte operacional). Tente novamente em instantes."
            )

        return DomainOutput(
            status="clarification",
            result={},
            explanation=message,
            confidence=0.9,
            metadata={"error": decision.error, "classification": "operational_unavailable"},
        )
