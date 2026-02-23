"""
Analysis Tier — LLM-powered analysis using specialized skills.

Responsibility:
- Resolve the analysis skill for the capability
- Gather required data via Facts and Calculator tiers
- Build specialized prompt from gathered data
- Call LLM via ModelSelector
- Return structured analysis as DomainOutput

This tier orchestrates: SkillRegistry → DataGathering → PromptBuilding → LLM → Output
"""

import json
import logging
from typing import Any

from shared.models import DomainOutput, ModelPolicy
from domains.finance.tiers.base import TierContext, TierProcessor
from domains.finance.tiers.facts import FactsTier
from domains.finance.tiers.calculators import CalculatorTier
from domains.finance.analysis_skills.base import AnalysisSkill, DataRequirement
from domains.finance.analysis_skills.registry import FinanceSkillRegistry

logger = logging.getLogger(__name__)

# Default model policy for analysis skills
DEFAULT_ANALYSIS_POLICY = ModelPolicy(
    model_name="claude-3-5-haiku-latest",
    temperature=0.3,
    timeout_seconds=60.0,
    max_retries=2,
    json_mode=False,
)


class AnalysisTier:
    """
    Tier 3: Analysis — LLM-powered financial analysis.

    Orchestrates:
    1. Skill resolution (which analysis persona to use)
    2. Data gathering (fetch required data via Facts/Calculator tiers)
    3. Prompt construction (build specialized prompt)
    4. LLM execution (call model with analysis prompt)
    5. Output formatting (structure response)
    """

    def __init__(
        self,
        skill_registry: FinanceSkillRegistry,
        facts_tier: FactsTier,
        calculator_tier: CalculatorTier | None = None,
        model_selector: Any = None,
    ):
        self._skill_registry = skill_registry
        self._facts_tier = facts_tier
        self._calculator_tier = calculator_tier
        self._model_selector = model_selector

    async def process(self, context: TierContext) -> DomainOutput:
        """
        Execute analysis pipeline:
        1. Resolve skill
        2. Gather data
        3. Build prompt
        4. Call LLM
        5. Return analysis
        """
        capability = context.intent.capability

        # 1. Resolve skill
        skill = self._skill_registry.resolve(capability)
        if skill is None:
            logger.warning("No analysis skill for: %s", capability)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"No analysis skill registered for '{capability}'.",
                confidence=0.0,
                metadata={"error": "skill_not_found", "tier": "analysis"},
            )

        # 2. Check model_selector availability
        if self._model_selector is None:
            logger.error("AnalysisTier: model_selector not configured")
            return DomainOutput(
                status="failure",
                result={},
                explanation="Serviço de análise LLM não está configurado.",
                confidence=0.0,
                metadata={"error": "model_selector_not_configured", "tier": "analysis"},
            )

        try:
            # 3. Gather required data
            gathered = await self._gather_data(skill, context)

            # 4. Build prompt
            user_prompt = skill.build_user_prompt(
                params=dict(context.params),
                gathered_data=gathered,
                original_query=context.original_query,
            )

            # 5. Call LLM
            policy = skill.model_policy or DEFAULT_ANALYSIS_POLICY
            messages = [
                {"role": "system", "content": skill.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self._model_selector.generate(messages=messages, policy=policy)

            # Handle response (str or dict depending on json_mode)
            if isinstance(response, dict):
                analysis_text = response.get("analysis", json.dumps(response, ensure_ascii=False))
            else:
                analysis_text = str(response)

            return DomainOutput(
                status="success",
                result={
                    "analysis": analysis_text,
                    "skill": skill.name,
                    "data_sources": list(gathered.keys()),
                },
                explanation=analysis_text,
                confidence=0.95,
                metadata={
                    "tier": "analysis",
                    "skill": skill.name,
                    "capability": capability,
                    "data_gathered": list(gathered.keys()),
                },
            )

        except Exception as e:
            logger.error("Analysis tier failed for %s: %s", capability, e, exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Erro na análise: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "tier": "analysis", "skill": getattr(skill, "name", "unknown")},
            )

    async def _gather_data(
        self,
        skill: AnalysisSkill,
        context: TierContext,
    ) -> dict[str, Any]:
        """
        Gather all data required by the skill.

        Iterates over skill.data_requirements, fetches each one
        via the appropriate tier, and collects results keyed by output_key.

        Args:
            skill: The analysis skill with data_requirements
            context: The tier context with user params

        Returns:
            Dict of {output_key: fetched_data}
        """
        gathered: dict[str, Any] = {}
        user_params = dict(context.params)

        for req in skill.data_requirements:
            try:
                resolved_params = req.resolve_params(user_params)
                data = await self._fetch_requirement(req, resolved_params, context)
                gathered[req.output_key] = data
                logger.debug(
                    "Gathered data for %s.%s: %s",
                    skill.name,
                    req.output_key,
                    type(data).__name__,
                )
            except Exception as e:
                if req.required:
                    raise RuntimeError(
                        f"Required data '{req.output_key}' failed: {e}"
                    ) from e
                logger.warning(
                    "Optional data '%s' failed for skill %s: %s",
                    req.output_key,
                    skill.name,
                    e,
                )
                gathered[req.output_key] = None

        return gathered

    async def _fetch_requirement(
        self,
        req: DataRequirement,
        resolved_params: dict[str, Any],
        context: TierContext,
    ) -> Any:
        """
        Fetch a single data requirement via the appropriate tier.

        For facts tier: creates a sub-context and runs FactsTier.process
        For calculator tier: creates a sub-context and runs CalculatorTier.process
        """
        from shared.models import ExecutionIntent

        # Build a sub-intent for the data fetch
        sub_intent = ExecutionIntent(
            domain=context.intent.domain,
            capability=req.capability,
            confidence=1.0,
            parameters=resolved_params,
            original_query=context.original_query,
        )

        sub_context = TierContext(
            intent=sub_intent,
            params=resolved_params,
            metadata={},  # Sub-calls don't need full metadata
            original_query=context.original_query,
        )

        if req.tier == "calculator" and self._calculator_tier is not None:
            output = await self._calculator_tier.process(sub_context)
        else:
            output = await self._facts_tier.process(sub_context)

        if output.status == "failure":
            raise RuntimeError(f"{req.capability} failed: {output.explanation}")

        return output.result
