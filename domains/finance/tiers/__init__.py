"""
Tier processors for the Finance domain.

Three tiers:
- Facts: MCP passthrough (get data, return data)
- Calculator: Local math + optional MCP data fetching
- Analysis: LLM-powered analysis using specialized skills
"""

from domains.finance.tiers.base import Tier, TierContext, TierProcessor

__all__ = ["Tier", "TierContext", "TierProcessor"]
