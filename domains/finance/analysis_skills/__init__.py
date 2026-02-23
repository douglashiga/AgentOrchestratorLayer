"""
Finance Analysis Skills — Specialized LLM prompts for financial analysis.

Each skill defines:
- System prompt (specializes the LLM persona)
- Data requirements (what data to gather before analysis)
- User prompt builder (constructs prompt from gathered data)
- Model policy (optional LLM configuration)
"""

from domains.finance.analysis_skills.base import AnalysisSkill, DataRequirement
from domains.finance.analysis_skills.registry import FinanceSkillRegistry

__all__ = ["AnalysisSkill", "DataRequirement", "FinanceSkillRegistry"]
