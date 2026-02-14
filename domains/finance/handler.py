"""
Finance Domain Handler.

Responsibility:
- Orchestrate internally: Context Resolver → Skill Gateway → Strategy Core
- Return Decision

Prohibitions:
- No LLM usage
- No bypassing gateway
- No global state
"""

import logging
import re
from typing import Any

from shared.models import Decision, DomainOutput, ExecutionContext, IntentOutput
from domains.finance.context import ContextResolver
from domains.finance.core import StrategyCore
from skills.gateway import SkillGateway
from domains.finance.schemas import (
    TopGainersInput, TopLosersInput, StockPriceInput, 
    HistoricalDataInput, StockScreenerInput, 
    TechnicalSignalsInput, CompareFundamentalsInput
)
from pydantic import ValidationError
from typing import get_type_hints

logger = logging.getLogger(__name__)


class FinanceDomainHandler:
    """Finance domain handler — orchestrates context, skills, and strategy."""

    def __init__(self, skill_gateway: SkillGateway, registry: Any = None):
        self.context_resolver = ContextResolver()
        self.strategy_core = StrategyCore()
        self.skill_gateway = skill_gateway
        self.registry = registry

    async def execute(self, intent: IntentOutput) -> DomainOutput:
        """
        Execute finance capability with type-safe dispatch.
        Attempts to find a specific handler method (e.g. get_top_gainers) 
        matching the intent capability.
        """
        try:
            # 0. Apply defaults without mutating shared state
            resolved_params = self._resolve_parameters(intent, dict(intent.parameters))

            # 1. Dynamic Dispatch to Typed Methods
            method_name = intent.capability
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                # Ensure it's a bound method, not a property
                if callable(method):
                    try:
                        # Inspect method hints to find the Pydantic model
                        type_hints = get_type_hints(method)
                        input_model = type_hints.get("params")
                        
                        if input_model:
                            # Validate inputs against schema
                            validated_params = input_model(**resolved_params)
                            # Call specific method with Intent AND Validated Params
                            return await method(intent, validated_params)
                    except ValidationError as ve:
                         # Pydantic validation error -> Ask for clarification
                         error_msg = str(ve).replace("\n", "; ")
                         return DomainOutput(
                            status="clarification",
                            result={},
                            explanation=f"I need more information to proceed. {error_msg}",
                            confidence=1.0,
                            metadata={"validation_error": str(ve)}
                        )
                    except Exception as e:
                         # Other execution error -> Failure
                         return DomainOutput(
                            status="failure",
                            explanation=f"Error executing {method_name}: {e}",
                            metadata={"error": str(e)}
                        )

            # 2. Fallback to Generic Execution (Legacy)
            return await self._generic_execute(intent, resolved_params)

        except Exception as e:
            logger.error("Finance execution failed: %s", e, exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing {intent.capability}: {str(e)}",
                confidence=0.0
            )

    # ─── Typed Capabilities ──────────────────────────────────────────

    async def get_top_gainers(self, intent: IntentOutput, params: TopGainersInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_top_losers(self, intent: IntentOutput, params: TopLosersInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_stock_price(self, intent: IntentOutput, params: StockPriceInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_historical_data(self, intent: IntentOutput, params: HistoricalDataInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_stock_screener(self, intent: IntentOutput, params: StockScreenerInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_technical_signals(self, intent: IntentOutput, params: TechnicalSignalsInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())
        
    async def compare_fundamentals(self, intent: IntentOutput, params: CompareFundamentalsInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    # ─── Unified Execution Pipeline ──────────────────────────────────

    async def _run_pipeline(self, intent: IntentOutput, params: dict) -> DomainOutput:
        """
        Unified pipeline for ALL finance executions (Typed & Generic).
        1. Metadata Check (Clarification overlap?) - Skipped if Typed (Pydantic handles valid structure)
        2. Context Resolution
        3. Skill Execution
        4. Strategy Core
        """
        capability = intent.capability
        metadata = self._get_capability_metadata(capability)
        
        # ─── 1. Pre-Flow Resolution (metadata/schema-driven) ───────
        pre_flow_result = self._apply_pre_flow(
            capability=capability,
            params=params,
            metadata=metadata,
            original_query=intent.original_query,
        )
        if isinstance(pre_flow_result, DomainOutput):
            return pre_flow_result
        params = pre_flow_result

        required_check = self._validate_required_params_from_schema(
            capability=capability,
            params=params,
            metadata=metadata,
        )
        if required_check is not None:
            return required_check

        # ─── 2. Context Resolution ───────────────────────────────────
        try:
            # Try to resolve from specific params first
            if params.get("symbol"):
                domain_context = self.context_resolver.resolve(params["symbol"])
            elif params.get("symbols") and isinstance(params["symbols"], list) and params["symbols"]:
                 domain_context = self.context_resolver.resolve(params["symbols"][0])
            elif params.get("market"):
                 domain_context = self.context_resolver.get_market_profile(params["market"])
                 if not domain_context:
                      domain_context = self.context_resolver.resolve("DEFAULT") 
            else:
                 domain_context = self.context_resolver.resolve("DEFAULT")

            logger.info("Context resolved: %s (%s)", domain_context.market, domain_context.currency)
            
            # ─── 3. Skill Execution ──────────────────────────────────
            
            # Prepare Skill Params
            sanitized_params = {k: v for k, v in params.items() if v is not None}
            skill_params = {
                **sanitized_params,
                "_action": capability,
            }
            
            # Inject Tool Map from Metadata
            if metadata:
                 skill_params["_tool_map"] = metadata.get("tool_map", {})
            
            # Inject Context if needed (e.g. market code from symbol resolution)
            if not skill_params.get("market") and domain_context.market != "US":
                skill_params["market"] = domain_context.market

            # Execute via Gateway
            skill_data = self.skill_gateway.execute("mcp_finance", skill_params)
            logger.debug("Skill data received: success=%s", skill_data.get("success"))

            if capability == "get_stock_price" and not skill_data.get("success", False):
                fallback_output = self._fallback_stock_price_from_cache(
                    params=params,
                    domain_context=domain_context,
                    original_error=str(skill_data.get("error", "")),
                )
                if fallback_output:
                    return fallback_output

            # ─── 4. Strategy Execution ───────────────────────────────
            execution_context = ExecutionContext(
                domain_context=domain_context,
                skill_data=skill_data,
            )

            intent_for_decision = intent.model_copy(update={"parameters": params})
            decision = self.strategy_core.execute(intent_for_decision, execution_context, registry=self.registry)

            clarification_output = self._map_operational_error_to_clarification(
                capability=capability,
                decision=decision,
            )
            if clarification_output is not None:
                return clarification_output
            
            # ─── 5. Output Generation ────────────────────────────────
            output_metadata = {"risk_metrics": decision.risk_metrics}
            if decision.error:
                output_metadata["error"] = decision.error

            return DomainOutput(
                status="success" if decision.success else "failure",
                result=decision.result,
                explanation=decision.explanation,
                confidence=1.0,
                metadata=output_metadata
            )

        except Exception as e:
            logger.error(f"Pipeline failed for {capability}: {e}", exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing finance action: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _validate_required_params_from_schema(
        self,
        capability: str,
        params: dict[str, Any],
        metadata: dict[str, Any],
    ) -> DomainOutput | None:
        schema = metadata.get("schema", {})
        if not isinstance(schema, dict):
            return None

        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list) or not required_fields:
            return None

        missing: list[str] = []
        for field in required_fields:
            field_name = str(field).strip()
            if not field_name:
                continue
            value = params.get(field_name)
            if value in (None, ""):
                missing.append(field_name)

        if not missing:
            return None

        fields_label = ", ".join(missing)
        return DomainOutput(
            status="clarification",
            result={},
            explanation=(
                f"Para executar '{capability}', faltam parâmetros obrigatórios: {fields_label}. "
                "Pode informar esses valores?"
            ),
            confidence=1.0,
            metadata={"missing_required_params": missing, "capability": capability},
        )

    def _map_operational_error_to_clarification(
        self,
        capability: str,
        decision: Decision,
    ) -> DomainOutput | None:
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

    async def _generic_execute(self, intent: IntentOutput, resolved_params: dict) -> DomainOutput:
        """Legacy generic execution path - delegates to unified pipeline."""
        if intent.capability == "chat":
            return DomainOutput(
                status="success",
                result={},
                explanation="This looks like a general question. Please ask a finance question."
            )
            
        # For generic execution, run manual metadata checks that Pydantic would have caught
        # (Legacy clarification logic could go here if we wanted to keep purely string-based checks)
        
        return await self._run_pipeline(intent, resolved_params)

    def _resolve_parameters(self, intent: IntentOutput, params: dict) -> dict:
        """
        Apply default values from metadata to missing parameters.
        Mutates `params` in-place.
        """
        metadata = self._get_capability_metadata(intent.capability)
        if not metadata:
            return params

        # Apply defaults
        for key, value in metadata.items():
            if key.startswith("default_"):
                param_name = key.replace("default_", "")
                if param_name not in params or params[param_name] is None:
                    params[param_name] = value
                    logger.info("Applied default parameter: %s=%s", param_name, value)
        return params

    def _get_capability_metadata(self, capability: str) -> dict[str, Any]:
        if not self.registry:
            return {}
        metadata = self.registry.get_metadata(capability)
        return metadata if isinstance(metadata, dict) else {}

    def _apply_pre_flow(
        self,
        capability: str,
        params: dict[str, Any],
        metadata: dict[str, Any],
        original_query: str,
    ) -> dict[str, Any] | DomainOutput:
        flow_steps = self._get_flow_steps(capability=capability, params=params, metadata=metadata)
        if not flow_steps:
            return params

        resolved = dict(params)
        for step in flow_steps:
            step_type = str(step.get("type", "")).strip()
            if step_type == "resolve_symbol":
                out = self._flow_resolve_symbol(step=step, params=resolved, original_query=original_query)
                if isinstance(out, DomainOutput):
                    return out
                resolved = out
            elif step_type == "resolve_symbol_list":
                out = self._flow_resolve_symbol_list(step=step, params=resolved)
                if isinstance(out, DomainOutput):
                    return out
                resolved = out
        return resolved

    def _get_flow_steps(self, capability: str, params: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
        schema = metadata.get("schema", {})
        if not isinstance(schema, dict):
            schema = {}
        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list):
            required_fields = []
        required_set = {str(item).strip() for item in required_fields if str(item).strip()}

        flow_block = metadata.get("flow", {})
        if isinstance(flow_block, dict):
            pre = flow_block.get("pre")
            if isinstance(pre, list):
                normalized_pre: list[dict[str, Any]] = []
                for step in pre:
                    if not isinstance(step, dict):
                        continue
                    step_copy = dict(step)
                    field = str(step_copy.get("param", "")).strip()
                    if field and "required" not in step_copy:
                        step_copy["required"] = field in required_set
                    normalized_pre.append(step_copy)
                return normalized_pre

        # Default inference from schema/params when explicit flow is absent.
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        steps: list[dict[str, Any]] = []
        if "symbol" in params or "symbol" in properties:
            steps.append({"type": "resolve_symbol", "param": "symbol", "required": "symbol" in required_set})
        if "symbols" in params or "symbols" in properties:
            steps.append({"type": "resolve_symbol_list", "param": "symbols", "required": "symbols" in required_set})
        return steps

    def _flow_resolve_symbol(
        self,
        step: dict[str, Any],
        params: dict[str, Any],
        original_query: str,
    ) -> dict[str, Any] | DomainOutput:
        field = str(step.get("param", "symbol")).strip() or "symbol"
        raw_value = params.get(field)
        if raw_value in (None, ""):
            inferred = self._infer_symbol_from_query_text(original_query)
            if inferred:
                updated = dict(params)
                updated[field] = inferred
                logger.info("Inferred symbol from query text: %s", inferred)
                return updated
            if step.get("required") is True:
                return DomainOutput(
                    status="clarification",
                    result={},
                    explanation=f"Qual ticker você quer consultar? (campo: {field})",
                    confidence=1.0,
                    metadata={"flow_step": "resolve_symbol", "field": field},
                )
            return params

        resolved_symbol = self._resolve_symbol_value(str(raw_value), step=step)
        if isinstance(resolved_symbol, DomainOutput):
            return resolved_symbol

        updated = dict(params)
        updated[field] = resolved_symbol
        return updated

    def _infer_symbol_from_query_text(self, query: str) -> str | None:
        text = (query or "").strip().upper()
        if not text:
            return None

        # Explicit exchange suffix first (e.g. PETR4.SA, VOLV-B.ST, AAPL)
        explicit_matches = re.findall(r"\b([A-Z0-9-]{1,12}\.[A-Z]{1,4})\b", text)
        for candidate in explicit_matches:
            normalized = self._normalize_canonical_symbol(candidate)
            if normalized:
                return normalized

        # B3 pattern without suffix (e.g. PETR4, VALE3, BOVA11)
        b3_matches = re.findall(r"\b([A-Z]{4}(?:3|4|5|6|11)(?:F)?)\b", text)
        for candidate in b3_matches:
            normalized = self._normalize_canonical_symbol(candidate)
            if normalized:
                return normalized

        # Common noisy B3 token (e.g. PETRO4 -> PETR4.SA)
        quasi_b3_matches = re.findall(r"\b([A-Z]{5,8}(?:3|4|5|6|11)(?:F)?)\b", text)
        for candidate in quasi_b3_matches:
            if candidate.endswith("11"):
                compact = f"{candidate[:4]}11"
            else:
                compact = f"{candidate[:4]}{candidate[-1]}"
            normalized = self._normalize_canonical_symbol(compact)
            if normalized:
                return normalized

        return None

    def _flow_resolve_symbol_list(self, step: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | DomainOutput:
        field = str(step.get("param", "symbols")).strip() or "symbols"
        raw_values = params.get(field)
        if not isinstance(raw_values, list) or not raw_values:
            return params

        resolved_list: list[str] = []
        for idx, item in enumerate(raw_values):
            resolved = self._resolve_symbol_value(str(item), step=step)
            if isinstance(resolved, DomainOutput):
                return resolved.model_copy(
                    update={
                        "metadata": {
                            **resolved.metadata,
                            "flow_step": "resolve_symbol_list",
                            "field": field,
                            "index": idx,
                        }
                    }
                )
            resolved_list.append(resolved)

        updated = dict(params)
        updated[field] = resolved_list
        return updated

    def _resolve_symbol_value(self, raw_symbol: str, step: dict[str, Any]) -> str | DomainOutput:
        symbol_norm = (raw_symbol or "").strip().upper()
        if not symbol_norm:
            return DomainOutput(
                status="clarification",
                result={},
                explanation="Informe um ticker válido (ex: VALE3.SA, PETR4.SA, AAPL).",
                confidence=1.0,
            )

        normalized_canonical = self._normalize_canonical_symbol(symbol_norm)
        is_plain_alpha = bool(re.fullmatch(r"[A-Z]{1,5}", symbol_norm))
        if normalized_canonical and not is_plain_alpha:
            return normalized_canonical

        search_capability = str(step.get("search_capability", "search_symbol")).strip() or "search_symbol"
        search_param = str(step.get("search_param", "query")).strip() or "query"
        search_fallbacks = step.get("search_fallback_capabilities")
        search_capabilities = [search_capability]
        if isinstance(search_fallbacks, list):
            for item in search_fallbacks:
                cap = str(item).strip()
                if cap and cap not in search_capabilities:
                    search_capabilities.append(cap)
        if "yahoo_search" not in search_capabilities:
            search_capabilities.append("yahoo_search")

        search_result = None
        used_search_capability = ""
        for capability_name in search_capabilities:
            trial = self.skill_gateway.execute(
                "mcp_finance",
                {
                    "_action": capability_name,
                    search_param: symbol_norm,
                },
            )
            if trial.get("success"):
                search_result = trial
                used_search_capability = capability_name
                break
            logger.warning("%s failed for symbol '%s': %s", capability_name, symbol_norm, trial.get("error"))

        if not search_result:
            # Deterministic fallback for plain alpha tickers (e.g., AAPL)
            # when lookup backend is unavailable.
            if normalized_canonical and is_plain_alpha:
                return normalized_canonical
            return DomainOutput(
                status="clarification",
                result={},
                explanation=(
                    f"Não consegui identificar o ticker para '{raw_symbol}'. "
                    "Pode informar o código exato? Ex: PETR4.SA, VALE3.SA, AAPL."
                ),
                confidence=1.0,
                metadata={"resolution": "symbol_lookup_failed"},
            )

        data = search_result.get("data", {})
        candidates = self._extract_symbol_candidates(data)
        if not candidates:
            if normalized_canonical and is_plain_alpha:
                return normalized_canonical
            return DomainOutput(
                status="clarification",
                result={},
                explanation=(
                    f"Não encontrei ticker para '{raw_symbol}'. "
                    "Pode informar o código exato?"
                ),
                confidence=1.0,
                metadata={"resolution": "symbol_not_found"},
            )

        if len(candidates) == 1:
            chosen = self._normalize_canonical_symbol(candidates[0]["symbol"]) or candidates[0]["symbol"]
            logger.info("Resolved symbol '%s' -> '%s' via %s", symbol_norm, chosen, used_search_capability)
            return chosen

        top = candidates[:5]
        options = ", ".join(item["symbol"] for item in top)
        return DomainOutput(
            status="clarification",
            result={"candidates": top},
            explanation=(
                f"Encontrei mais de um ticker para '{raw_symbol}': {options}. "
                "Qual deles você quer consultar?"
            ),
            confidence=1.0,
            metadata={"resolution": "ambiguous_symbol"},
        )

    def _looks_canonical_symbol(self, symbol: str) -> bool:
        """
        Canonical patterns:
        - AAPL
        - VALE3.SA
        - VOLV-B.ST
        """
        return self._normalize_canonical_symbol(symbol) is not None

    def _normalize_canonical_symbol(self, symbol: str) -> str | None:
        """
        Normalize trusted canonical patterns:
        - Explicit exchange suffix: PETR4.SA, VOLV-B.ST
        - B3 base format: PETR4, VALE3, BOVA11 -> add .SA
        - US base format: AAPL, TSLA
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        if bool(re.fullmatch(r"[A-Z0-9-]{1,12}\.[A-Z]{1,4}", sym)):
            return sym
        # B3 base symbols: 4 letters + class (3/4/5/6/11)
        if bool(re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", sym)):
            return f"{sym}.SA"
        # US tickers (letters only, short)
        if bool(re.fullmatch(r"[A-Z]{1,5}", sym)):
            return sym
        return None

    def _extract_symbol_candidates(self, payload: Any) -> list[dict[str, str]]:
        """
        Parse symbol candidates from heterogeneous MCP responses.
        """
        raw_items: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            for key in ("results", "symbols", "matches", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    raw_items = value
                    break
            if not raw_items:
                # Single-object fallback
                raw_items = [payload]
        elif isinstance(payload, list):
            raw_items = payload

        candidates: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            symbol = (
                item.get("symbol")
                or item.get("ticker")
                or item.get("code")
                or item.get("instrument")
            )
            if not symbol:
                continue
            symbol_norm = str(symbol).strip().upper()
            if not symbol_norm or symbol_norm in seen:
                continue
            seen.add(symbol_norm)
            name = str(item.get("name") or item.get("company") or item.get("description") or "").strip()
            candidates.append({"symbol": symbol_norm, "name": name})
        return candidates

    def _fallback_stock_price_from_cache(
        self,
        params: dict[str, Any],
        domain_context: Any,
        original_error: str,
    ) -> DomainOutput | None:
        symbol = str(params.get("symbol", "")).strip()
        if not symbol:
            return None

        attempts: list[str] = []
        attempts.append(symbol)
        if "." in symbol:
            attempts.append(symbol.split(".", 1)[0])

        seen: set[str] = set()
        ordered_attempts: list[str] = []
        for item in attempts:
            norm = item.strip().upper()
            if norm and norm not in seen:
                seen.add(norm)
                ordered_attempts.append(norm)

        for candidate in ordered_attempts:
            trial = self.skill_gateway.execute(
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
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            # Envelope: {"success": true, "data": [...]}
            if isinstance(payload.get("data"), list):
                return [item for item in payload.get("data", []) if isinstance(item, dict)]
            inner = payload.get("data")
            if isinstance(inner, dict) and isinstance(inner.get("data"), list):
                return [item for item in inner.get("data", []) if isinstance(item, dict)]
        return []
