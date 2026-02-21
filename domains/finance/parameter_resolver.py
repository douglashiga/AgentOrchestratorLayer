"""
Parameter Resolver — Two-tier resolution with sub-resolver registry.

Architecture:
  ParameterResolver
    ├── _sub_resolvers: dict[str, SubResolver]  (specialized per parameter)
    ├── _default_resolver: DefaultParameterSubResolver  (DB → enum → LLM → passthrough)
    └── resolve_all()  (orchestrates resolution for all params of a capability)

Tier 1 (Deterministic): SQLite lookup via ParameterResolverDB
Tier 2 (LLM Fallback):  Metadata-driven prompt with auto-learning
"""

import json
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from domains.finance.parameter_resolver_db import ParameterResolverDB
from shared.models import DomainOutput, ModelPolicy

logger = logging.getLogger(__name__)


# ─── Models ──────────────────────────────────────────────────────────────────


class ResolvedParameter(BaseModel):
    """Result of resolving a single parameter."""

    model_config = {"frozen": True}

    parameter_name: str
    original_value: Any = None
    resolved_value: Any = None
    confidence: float = 1.0
    source: str = "unknown"  # deterministic_db | metadata_default | enum_match |
    #                          type_coercion | llm | sub_resolver | passthrough
    needs_clarification: bool = False
    clarification_message: str = ""


class ParameterResolutionConfig(BaseModel):
    """Configuration for the parameter resolver."""

    model_config = {"frozen": True}

    llm_confidence_threshold: float = Field(
        default=0.9, description="Auto-learn if LLM confidence >= this"
    )
    llm_clarification_threshold: float = Field(
        default=0.6, description="Ask user if LLM confidence < this"
    )
    enable_llm: bool = True
    enable_auto_learn: bool = True
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 300


class ParameterResolverContext(BaseModel):
    """Context shared across sub-resolvers during a single resolve_all call."""

    model_config = {"frozen": True}

    capability: str = ""
    original_query: str = ""
    all_params: dict[str, Any] = Field(default_factory=dict)
    parameter_specs: dict[str, Any] = Field(default_factory=dict)


# ─── Sub-Resolver Interface ─────────────────────────────────────────────────


class SubResolver(ABC):
    """Interface for specialized per-parameter resolvers."""

    @abstractmethod
    def resolve(
        self,
        parameter_name: str,
        raw_value: Any,
        spec: dict[str, Any],
        context: ParameterResolverContext,
    ) -> ResolvedParameter:
        """Resolve a value for this parameter."""
        ...


# ─── Symbol Sub-Resolvers ───────────────────────────────────────────────────


class SymbolSubResolver(SubResolver):
    """Wraps the existing SymbolResolver as a sub-resolver."""

    def __init__(self, symbol_resolver: Any):
        self._symbol_resolver = symbol_resolver

    def resolve(
        self,
        parameter_name: str,
        raw_value: Any,
        spec: dict[str, Any],
        context: ParameterResolverContext,
    ) -> ResolvedParameter:
        if raw_value in (None, ""):
            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=raw_value,
                confidence=0.0,
                source="sub_resolver",
                needs_clarification=False,  # symbol flow in handler handles this
            )

        result = self._symbol_resolver.resolve(str(raw_value))
        if result:
            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=result.symbol,
                confidence=result.confidence,
                source="sub_resolver",
            )

        # Could not resolve — return as-is, let the handler's symbol flow handle it
        return ResolvedParameter(
            parameter_name=parameter_name,
            original_value=raw_value,
            resolved_value=raw_value,
            confidence=0.5,
            source="sub_resolver",
        )


class SymbolListSubResolver(SubResolver):
    """Resolves lists of symbols by delegating each item to SymbolResolver."""

    def __init__(self, symbol_resolver: Any):
        self._symbol_resolver = symbol_resolver

    def resolve(
        self,
        parameter_name: str,
        raw_value: Any,
        spec: dict[str, Any],
        context: ParameterResolverContext,
    ) -> ResolvedParameter:
        if not isinstance(raw_value, list) or not raw_value:
            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=raw_value,
                confidence=1.0,
                source="sub_resolver",
            )

        resolved_list: list[str] = []
        min_confidence = 1.0
        for item in raw_value:
            result = self._symbol_resolver.resolve(str(item))
            if result:
                resolved_list.append(result.symbol)
                min_confidence = min(min_confidence, result.confidence)
            else:
                resolved_list.append(str(item))
                min_confidence = min(min_confidence, 0.5)

        return ResolvedParameter(
            parameter_name=parameter_name,
            original_value=raw_value,
            resolved_value=resolved_list,
            confidence=min_confidence,
            source="sub_resolver",
        )


# ─── Default Parameter Sub-Resolver ─────────────────────────────────────────


class DefaultParameterSubResolver(SubResolver):
    """
    Generic pipeline for any textual/enum parameter:
    1. Apply default if missing
    2. Normalize input
    3. Deterministic DB lookup
    4. Enum match (case-insensitive)
    5. infer_from_symbol_suffix
    6. LLM fallback (metadata-driven prompt)
    7. Passthrough
    """

    def __init__(
        self,
        db: ParameterResolverDB,
        model_selector: Any | None = None,
        config: ParameterResolutionConfig | None = None,
    ):
        self._db = db
        self._model_selector = model_selector
        self._config = config or ParameterResolutionConfig()

    def resolve(
        self,
        parameter_name: str,
        raw_value: Any,
        spec: dict[str, Any],
        context: ParameterResolverContext,
    ) -> ResolvedParameter:
        # Step 1: Apply default if missing
        if raw_value in (None, ""):
            default = spec.get("default")
            if default is not None:
                return ResolvedParameter(
                    parameter_name=parameter_name,
                    original_value=raw_value,
                    resolved_value=default,
                    confidence=1.0,
                    source="metadata_default",
                )
            # No default and no value — passthrough
            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=raw_value,
                confidence=1.0,
                source="passthrough",
            )

        # Only process string values through the resolution pipeline
        if not isinstance(raw_value, str):
            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=raw_value,
                confidence=1.0,
                source="passthrough",
            )

        # Step 2: Normalize input
        normalized = self._normalize_input(raw_value, spec)

        # Step 3: Deterministic DB lookup (with enum guard)
        result = self._resolve_via_db(parameter_name, normalized, spec)
        if result:
            return result

        # Step 4: Enum match (case-insensitive)
        result = self._resolve_via_enum(parameter_name, normalized, raw_value, spec)
        if result:
            return result

        # Step 4b: Nearest covering period — if the enum contains time
        # periods and the user asked for a duration not in the enum, pick the
        # smallest enum value that fully covers the requested duration.
        # Example: user says "3 dias", enum is [1d, 5d, 1mo] → returns 5d.
        result = self._resolve_via_nearest_period(parameter_name, normalized, raw_value, spec)
        if result:
            return result

        # Step 5: infer_from_symbol_suffix
        result = self._resolve_via_symbol_suffix(
            parameter_name, raw_value, spec, context.all_params
        )
        if result:
            return result

        # Step 6: LLM fallback
        if self._config.enable_llm and self._model_selector:
            result = self._resolve_via_llm(
                parameter_name, raw_value, spec, context.capability
            )
            if result:
                return result

        # Step 7: Passthrough — apply normalization rules if any
        final_value = self._apply_normalization(raw_value, spec)

        # If an enum is defined and the value doesn't match any entry,
        # request clarification instead of silently passing through.
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_lower = {str(v).lower() for v in enum_values}
            if str(final_value).lower() not in enum_lower:
                options = ", ".join(str(v) for v in enum_values)
                return ResolvedParameter(
                    parameter_name=parameter_name,
                    original_value=raw_value,
                    resolved_value=None,
                    confidence=0.0,
                    source="passthrough",
                    needs_clarification=True,
                    clarification_message=(
                        f"I couldn't resolve '{raw_value}' for '{parameter_name}'. "
                        f"Valid options are: {options}"
                    ),
                )

        return ResolvedParameter(
            parameter_name=parameter_name,
            original_value=raw_value,
            resolved_value=final_value,
            confidence=0.5,
            source="passthrough",
        )

    def _normalize_input(self, value: str, spec: dict[str, Any]) -> str:
        """Strip and apply case normalization from spec."""
        text = value.strip()
        normalization = spec.get("normalization")
        if isinstance(normalization, dict):
            case_mode = str(normalization.get("case", "")).strip().lower()
            if case_mode == "upper":
                text = text.upper()
            elif case_mode == "lower":
                text = text.lower()
        return text

    def _apply_normalization(self, value: str, spec: dict[str, Any]) -> str:
        """Apply full normalization rules (case + suffix)."""
        text = value.strip()
        normalization = spec.get("normalization")
        if isinstance(normalization, dict):
            case_mode = str(normalization.get("case", "")).strip().lower()
            if case_mode == "upper":
                text = text.upper()
            elif case_mode == "lower":
                text = text.lower()
            suffix = normalization.get("suffix")
            if isinstance(suffix, str) and suffix.strip():
                suffix_val = suffix.strip()
                if not text.upper().endswith(suffix_val.upper()):
                    text = f"{text}{suffix_val}"
        return text

    def _resolve_via_db(
        self,
        parameter_name: str,
        normalized_value: str,
        spec: dict[str, Any] | None = None,
    ) -> ResolvedParameter | None:
        """Tier 1: Deterministic DB lookup with enum guard.

        If the capability defines an enum for this parameter, the DB-resolved
        value MUST be in that enum. Otherwise the DB result is discarded so
        the pipeline can fall through to later steps (enum match, LLM, etc.).
        This prevents cross-capability pollution (e.g. '5 ANOS' -> '5y' being
        accepted for a capability that only supports ['1d','5d','1mo','3mo','1y']).
        """
        mapping = self._db.lookup(parameter_name, normalized_value)
        if mapping:
            resolved = mapping["resolved_value"]

            # Enum guard: if spec defines an enum, verify the resolved value is valid
            if spec:
                enum_values = spec.get("enum")
                if isinstance(enum_values, list) and enum_values:
                    enum_lower = {str(v).lower() for v in enum_values}
                    if str(resolved).lower() not in enum_lower:
                        logger.debug(
                            "DB resolved '%s' -> '%s' for '%s', but value not in enum %s — skipping",
                            normalized_value, resolved, parameter_name, enum_values,
                        )
                        return None  # Fall through to next resolution step

            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=normalized_value,
                resolved_value=resolved,
                confidence=mapping["confidence"],
                source="deterministic_db",
            )
        return None

    def _resolve_via_enum(
        self,
        parameter_name: str,
        normalized_value: str,
        original_value: str,
        spec: dict[str, Any],
    ) -> ResolvedParameter | None:
        """Match against spec['enum'] case-insensitively."""
        enum_values = spec.get("enum")
        if not isinstance(enum_values, list) or not enum_values:
            return None

        lowered = normalized_value.lower()
        for item in enum_values:
            if str(item).lower() == lowered:
                return ResolvedParameter(
                    parameter_name=parameter_name,
                    original_value=original_value,
                    resolved_value=item,
                    confidence=1.0,
                    source="enum_match",
                )
        return None

    # ── Nearest-covering period resolution helpers ──────────────────

    # Canonical period tokens → approximate days
    _PERIOD_TOKEN_DAYS: dict[str, float] = {
        "1d": 1, "2d": 2, "3d": 3, "5d": 5,
        "1D": 1, "2D": 2, "3D": 3, "5D": 5,
        "1wk": 7,
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825,
        "ytd": 180,   # approximate mid-year
        "max": 36500,  # ~100 years sentinel
    }

    # User-facing duration words → days-per-unit
    _DURATION_UNIT_DAYS: list[tuple[str, float]] = [
        # Portuguese
        ("dias", 1), ("dia", 1),
        ("semanas", 7), ("semana", 7),
        ("meses", 30), ("mes", 30), ("mês", 30),
        ("anos", 365), ("ano", 365),
        ("trimestres", 90), ("trimestre", 90),
        ("semestres", 180), ("semestre", 180),
        # English
        ("days", 1), ("day", 1),
        ("weeks", 7), ("week", 7),
        ("months", 30), ("month", 30),
        ("years", 365), ("year", 365),
        ("quarters", 90), ("quarter", 90),
    ]

    def _period_to_days(self, token: str) -> float | None:
        """Convert a canonical period token (e.g. '5d', '1mo') to approx days."""
        return self._PERIOD_TOKEN_DAYS.get(token)

    def _parse_user_duration_to_days(self, text: str) -> float | None:
        """
        Parse a user-provided duration string into approximate days.

        Handles patterns like:
          '3 dias', '2 semanas', '4 months', '1 ano',
          '3d', '2w', '6m', '1y'
        """
        cleaned = unicodedata.normalize("NFKC", text.strip()).lower()

        # Pattern 1: number + word  ('3 dias', '2 weeks', etc.)
        for unit_word, days_per in self._DURATION_UNIT_DAYS:
            pattern = rf"(\d+)\s*{re.escape(unit_word)}"
            match = re.search(pattern, cleaned)
            if match:
                return float(match.group(1)) * days_per

        # Pattern 2: compact format  ('3d', '2w', '6m', '1y')
        compact_match = re.fullmatch(r"(\d+)\s*([dwmy])", cleaned)
        if compact_match:
            num = float(compact_match.group(1))
            unit = compact_match.group(2)
            unit_map = {"d": 1, "w": 7, "m": 30, "y": 365}
            return num * unit_map[unit]

        return None

    def _resolve_via_nearest_period(
        self,
        parameter_name: str,
        normalized_value: str,
        original_value: str,
        spec: dict[str, Any],
    ) -> ResolvedParameter | None:
        """
        Find the smallest enum period that fully covers the user's requested
        duration. Only activates when ALL enum values are parseable as time
        periods.

        Example: user asks '3 dias', enum=[1d, 5d, 1mo] → returns '5d'
        because 5d is the smallest range >= 3 days.
        """
        enum_values = spec.get("enum")
        if not isinstance(enum_values, list) or len(enum_values) < 2:
            return None

        # Check that all enum values are time periods
        enum_days: list[tuple[str, float]] = []
        for val in enum_values:
            days = self._period_to_days(str(val))
            if days is None:
                return None  # Not a time-period enum — bail out
            enum_days.append((str(val), days))

        # Parse the user's input to days
        requested_days = self._parse_user_duration_to_days(original_value)
        if requested_days is None:
            requested_days = self._parse_user_duration_to_days(normalized_value)
        if requested_days is None:
            return None  # Can't parse input — bail out

        # Sort enum by days ascending
        enum_days.sort(key=lambda x: x[1])

        # Find smallest covering value (>= requested days)
        for enum_val, enum_d in enum_days:
            if enum_d >= requested_days:
                logger.info(
                    "Nearest period: '%s' (~%.0f days) → '%s' (~%.0f days) for '%s'",
                    original_value, requested_days, enum_val, enum_d, parameter_name,
                )
                return ResolvedParameter(
                    parameter_name=parameter_name,
                    original_value=original_value,
                    resolved_value=enum_val,
                    confidence=0.9,
                    source="nearest_period",
                )

        # Requested duration exceeds all enum values — return the largest
        largest_val, largest_d = enum_days[-1]
        logger.info(
            "Nearest period (capped): '%s' (~%.0f days) → '%s' (~%.0f days) for '%s'",
            original_value, requested_days, largest_val, largest_d, parameter_name,
        )
        return ResolvedParameter(
            parameter_name=parameter_name,
            original_value=original_value,
            resolved_value=largest_val,
            confidence=0.85,
            source="nearest_period",
        )

    def _resolve_via_symbol_suffix(
        self,
        parameter_name: str,
        raw_value: str,
        spec: dict[str, Any],
        all_params: dict[str, Any],
    ) -> ResolvedParameter | None:
        """Infer value from symbol suffix (e.g., .SA -> BRL for currency)."""
        infer_map = spec.get("infer_from_symbol_suffix")
        if not isinstance(infer_map, dict):
            return None

        # Find a symbol value in params
        symbol_value: str | None = None
        raw_symbol = all_params.get("symbol")
        if isinstance(raw_symbol, str) and raw_symbol.strip():
            symbol_value = raw_symbol.strip().upper()
        else:
            raw_symbols = all_params.get("symbols")
            if isinstance(raw_symbols, list):
                for item in raw_symbols:
                    if isinstance(item, str) and item.strip():
                        symbol_value = item.strip().upper()
                        break

        if not symbol_value:
            return None

        for suffix_raw, inferred in infer_map.items():
            suffix = str(suffix_raw).strip().upper()
            if suffix and symbol_value.endswith(suffix):
                return ResolvedParameter(
                    parameter_name=parameter_name,
                    original_value=raw_value,
                    resolved_value=inferred,
                    confidence=1.0,
                    source="infer_from_symbol",
                )
        return None

    def _resolve_via_llm(
        self,
        parameter_name: str,
        raw_value: str,
        spec: dict[str, Any],
        capability: str = "",
    ) -> ResolvedParameter | None:
        """
        Tier 2: LLM resolution with metadata-driven prompt.
        Only for string params that have enum or aliases to resolve against.
        """
        if not self._model_selector:
            return None

        # Only use LLM if there are constraints to resolve against
        has_enum = isinstance(spec.get("enum"), list) and spec["enum"]
        has_aliases = isinstance(spec.get("aliases"), dict) and spec["aliases"]
        if not has_enum and not has_aliases:
            return None

        try:
            prompt = self._build_llm_prompt(parameter_name, raw_value, spec, capability)

            policy = ModelPolicy(
                model_name=self._config.llm_model,
                temperature=self._config.llm_temperature,
                timeout_seconds=10.0,
                max_retries=2,
                json_mode=True,
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a parameter resolver. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]

            result = self._model_selector.generate(messages=messages, policy=policy)

            # Parse response
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.warning(
                        "LLM returned non-JSON for param '%s': %s",
                        parameter_name,
                        result[:100],
                    )
                    return None

            if not isinstance(result, dict):
                return None

            resolved_value = result.get("resolved_value")
            confidence = float(result.get("confidence", 0.0))

            if resolved_value is None:
                return None

            # Validate against enum if present
            enum_values = spec.get("enum", [])
            if isinstance(enum_values, list) and enum_values:
                if resolved_value not in enum_values:
                    logger.warning(
                        "LLM returned invalid enum value '%s' for param '%s'. "
                        "Valid: %s",
                        resolved_value,
                        parameter_name,
                        enum_values,
                    )
                    return ResolvedParameter(
                        parameter_name=parameter_name,
                        original_value=raw_value,
                        resolved_value=raw_value,
                        confidence=0.0,
                        source="llm",
                        needs_clarification=True,
                        clarification_message=(
                            f"Nao consegui resolver '{raw_value}' para {parameter_name}. "
                            f"Opcoes validas: {', '.join(str(v) for v in enum_values)}"
                        ),
                    )

            # Auto-learn if high confidence
            if (
                confidence >= self._config.llm_confidence_threshold
                and self._config.enable_auto_learn
            ):
                self._auto_learn(
                    parameter_name, raw_value, str(resolved_value), confidence
                )

            # Check if needs clarification
            needs_clarification = (
                confidence < self._config.llm_clarification_threshold
            )

            clarification_msg = ""
            if needs_clarification:
                if isinstance(enum_values, list) and enum_values:
                    options_str = ", ".join(str(v) for v in enum_values)
                    clarification_msg = (
                        f"Nao tenho certeza sobre '{raw_value}' para {parameter_name}. "
                        f"Opcoes disponiveis: {options_str}"
                    )
                else:
                    clarification_msg = (
                        f"Nao tenho certeza sobre '{raw_value}' para {parameter_name}."
                    )

            return ResolvedParameter(
                parameter_name=parameter_name,
                original_value=raw_value,
                resolved_value=resolved_value,
                confidence=confidence,
                source="llm",
                needs_clarification=needs_clarification,
                clarification_message=clarification_msg,
            )

        except Exception as e:
            logger.warning("LLM resolution error for '%s': %s", parameter_name, e)
            return None

    def _build_llm_prompt(
        self,
        parameter_name: str,
        raw_value: str,
        spec: dict[str, Any],
        capability: str = "",
    ) -> str:
        """Build metadata-driven prompt with explicit formats and rules."""
        parts: list[str] = []

        parts.append(
            "You are a parameter resolver for a finance domain system. "
            "Given a user input value for a specific parameter, resolve it to the "
            "correct canonical value."
        )

        parts.append(f"\nParameter: {parameter_name}")

        description = spec.get("description", "")
        if description:
            parts.append(f"Description: {description}")

        if capability:
            parts.append(f"Capability context: {capability}")

        param_type = spec.get("type", "string")
        parts.append(f"Expected type: {param_type}")

        # Enum values (most important constraint)
        enum_values = spec.get("enum", [])
        if isinstance(enum_values, list) and enum_values:
            parts.append(
                f"Valid values (MUST be one of): {json.dumps(enum_values, ensure_ascii=False)}"
            )

        # Known aliases (show pattern for reference)
        aliases = spec.get("aliases", {})
        if isinstance(aliases, dict) and aliases:
            sample = dict(list(aliases.items())[:10])
            parts.append(
                f"Known alias examples (for reference): {json.dumps(sample, ensure_ascii=False)}"
            )

        # Normalization rules
        normalization = spec.get("normalization", {})
        if isinstance(normalization, dict):
            case_mode = normalization.get("case", "")
            if case_mode:
                parts.append(f"Case normalization: {case_mode}")
            suffix = normalization.get("suffix", "")
            if suffix:
                parts.append(f"Required suffix: {suffix}")

        # Examples
        examples = spec.get("examples", [])
        if isinstance(examples, list) and examples:
            parts.append(
                f"Examples of correct values: {json.dumps(examples, ensure_ascii=False)}"
            )

        # Default value
        default = spec.get("default")
        if default is not None:
            parts.append(f"Default value: {default}")

        parts.append(f'\nUser input: "{raw_value}"')

        parts.append(
            "\nReturn ONLY a JSON object with:\n"
            "{\n"
            '  "resolved_value": "<canonical value>",\n'
            "  \"confidence\": <0.0 to 1.0>,\n"
            '  "reasoning": "<brief explanation>"\n'
            "}"
        )

        if isinstance(enum_values, list) and enum_values:
            parts.append(
                f"\nIMPORTANT: resolved_value MUST be exactly one of: "
                f"{json.dumps(enum_values, ensure_ascii=False)}. "
                "If the input does not clearly match any valid value, set confidence below 0.6."
            )

        return "\n".join(parts)

    def _auto_learn(
        self,
        parameter_name: str,
        input_value: str,
        resolved_value: str,
        confidence: float,
    ) -> None:
        """Insert LLM-learned mapping into deterministic DB for future lookups."""
        try:
            self._db.insert_mapping(
                parameter_name=parameter_name,
                input_value=input_value,
                resolved_value=resolved_value,
                source="llm_learned",
                confidence=confidence,
            )
            logger.info(
                "Auto-learned mapping: %s['%s'] -> '%s' (confidence=%.2f)",
                parameter_name,
                input_value,
                resolved_value,
                confidence,
            )
        except Exception as e:
            logger.warning("Failed to auto-learn mapping: %s", e)


# ─── Main ParameterResolver ─────────────────────────────────────────────────

# Numeric types that should be skipped (passed through without resolution)
_NUMERIC_TYPES = {"number", "float", "integer", "int"}


class ParameterResolver:
    """
    Finance domain parameter resolver with sub-resolver registry.

    For each parameter:
      1. Numeric type? → skip (passthrough)
      2. Has registered sub-resolver? → delegate to it
      3. Otherwise → use DefaultParameterSubResolver (DB → enum → LLM → passthrough)
    """

    def __init__(
        self,
        db: ParameterResolverDB,
        model_selector: Any | None = None,
        config: ParameterResolutionConfig | None = None,
    ):
        self._db = db
        self._model_selector = model_selector
        self._config = config or ParameterResolutionConfig()
        self._sub_resolvers: dict[str, SubResolver] = {}
        self._default_resolver = DefaultParameterSubResolver(
            db=db, model_selector=model_selector, config=self._config
        )
        logger.info(
            "ParameterResolver initialized (llm=%s, auto_learn=%s)",
            self._config.enable_llm,
            self._config.enable_auto_learn,
        )

    def register_resolver(
        self, parameter_name: str, resolver: SubResolver
    ) -> None:
        """Register a specialized sub-resolver for a parameter."""
        self._sub_resolvers[parameter_name] = resolver
        logger.info(
            "Registered sub-resolver for '%s': %s",
            parameter_name,
            type(resolver).__name__,
        )

    def resolve_all(
        self,
        params: dict[str, Any],
        parameter_specs: dict[str, dict[str, Any]],
        capability: str = "",
        original_query: str = "",
    ) -> dict[str, Any] | DomainOutput:
        """
        Resolve all textual/enum parameters for a capability.

        Returns resolved params dict or DomainOutput(status='clarification').
        """
        if not parameter_specs:
            return params

        context = ParameterResolverContext(
            capability=capability,
            original_query=original_query,
            all_params=dict(params),
            parameter_specs=parameter_specs,
        )
        resolved = dict(params)

        for param_name, spec in parameter_specs.items():
            if not isinstance(spec, dict):
                continue

            # Skip numeric types
            param_type = str(spec.get("type", "string")).strip().lower()
            if param_type in _NUMERIC_TYPES:
                continue

            # Skip boolean types
            if param_type in ("boolean", "bool"):
                continue

            # Get raw value, trying _text variant as fallback
            raw_value = resolved.get(param_name)
            if raw_value in (None, ""):
                text_key = f"{param_name}_text"
                raw_value = resolved.get(text_key)

            # Select resolver
            if param_name in self._sub_resolvers:
                resolver = self._sub_resolvers[param_name]
            else:
                resolver = self._default_resolver

            result = resolver.resolve(param_name, raw_value, spec, context)

            if result.needs_clarification:
                return DomainOutput(
                    status="clarification",
                    result={},
                    explanation=result.clarification_message,
                    confidence=1.0,
                    metadata={
                        "parameter": param_name,
                        "original_value": str(raw_value) if raw_value else "",
                        "source": result.source,
                    },
                )

            resolved[param_name] = result.resolved_value

        return resolved

    def get_stats(self) -> dict[str, Any]:
        """Get resolver statistics."""
        return {
            "sub_resolvers": list(self._sub_resolvers.keys()),
            "db_stats": self._db.get_stats(),
            "llm_enabled": self._config.enable_llm,
            "auto_learn_enabled": self._config.enable_auto_learn,
        }
