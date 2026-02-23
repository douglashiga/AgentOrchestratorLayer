"""
Finance Calculators — Basic deterministic math for general finance.

Functions:
- calc_percentage: Basic percentage operations (X% of Y, X is what % of Y)
- calc_average_cost: Weighted average cost basis with premium adjustment
- calc_compound_growth: Compound growth projection with optional contributions
- calc_risk_reward: Risk/Reward ratio calculation
"""

import logging
from typing import Any

from domains.finance.tiers.calculators import CalculatorRegistry, DataFetcher

logger = logging.getLogger(__name__)


async def calc_percentage(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Basic percentage calculations.

    Mode 1: "quanto é X% de Y?" → value + percentage → result
    Mode 2: "X é qual % de Y?" → part + whole → percentage
    """
    value = params.get("value")
    percentage = params.get("percentage")
    part = params.get("part")
    whole = params.get("whole")

    # Mode 1: X% of Y
    if value is not None and percentage is not None:
        val = float(value)
        pct = float(percentage)
        computed = val * (pct / 100)
        return {
            "mode": "percentage_of_value",
            "value": val,
            "percentage": pct,
            "result": round(computed, 2),
            "explanation": f"{pct}% de {val} = {computed:.2f}",
        }

    # Mode 2: X is what % of Y
    if part is not None and whole is not None:
        p = float(part)
        w = float(whole)
        if w == 0:
            return {"error": "whole cannot be zero", "explanation": "O valor total não pode ser zero."}
        pct = (p / w) * 100
        return {
            "mode": "what_percentage",
            "part": p,
            "whole": w,
            "percentage": round(pct, 4),
            "explanation": f"{p} é {pct:.4f}% de {w}",
        }

    return {
        "error": "Provide (value + percentage) or (part + whole)",
        "explanation": "Informe (value + percentage) para calcular X% de Y, ou (part + whole) para saber qual %.",
    }


async def calc_average_cost(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate weighted average cost basis.

    Required: lots — list of {quantity, price} dicts
    Optional: premium_received (adjusts cost basis downward)
    """
    lots = params.get("lots", [])
    premium_received = float(params.get("premium_received", 0))

    if not isinstance(lots, list) or not lots:
        return {"error": "lots must be a non-empty list", "explanation": "Informe as compras no formato [{quantity, price}, ...]."}

    total_qty = 0
    total_cost = 0.0
    lot_details = []

    for i, lot in enumerate(lots):
        if not isinstance(lot, dict):
            return {"error": f"lot[{i}] must be a dict with quantity and price", "explanation": f"O lote {i} deve ter quantity e price."}
        qty = float(lot.get("quantity", 0))
        price = float(lot.get("price", 0))
        if qty <= 0 or price <= 0:
            return {"error": f"lot[{i}] has invalid quantity or price", "explanation": f"O lote {i} tem quantidade ou preço inválido."}
        total_qty += qty
        total_cost += qty * price
        lot_details.append({"quantity": qty, "price": price, "subtotal": round(qty * price, 2)})

    avg_cost = total_cost / total_qty if total_qty > 0 else 0

    result: dict[str, Any] = {
        "lots": lot_details,
        "total_quantity": total_qty,
        "total_cost": round(total_cost, 2),
        "average_cost": round(avg_cost, 4),
    }

    if premium_received > 0:
        adjusted = avg_cost - (premium_received / total_qty) if total_qty > 0 else avg_cost
        result["premium_received"] = premium_received
        result["adjusted_cost"] = round(adjusted, 4)
        result["explanation"] = (
            f"Preço médio: {avg_cost:.4f} ({total_qty:.0f} ações, custo total {total_cost:.2f}). "
            f"Com prêmio de {premium_received:.2f}, custo ajustado: {adjusted:.4f}."
        )
    else:
        result["explanation"] = (
            f"Preço médio: {avg_cost:.4f} ({total_qty:.0f} ações, custo total {total_cost:.2f})."
        )

    return result


async def calc_compound_growth(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate compound growth with optional periodic contributions.

    Required: principal, rate_pct, periods
    Optional: contribution_per_period (default 0)
    """
    principal = float(params.get("principal", 0))
    rate_pct = float(params.get("rate_pct", 0))
    periods = int(params.get("periods", 0))
    contribution = float(params.get("contribution_per_period", 0))

    if principal <= 0:
        return {"error": "principal must be positive", "explanation": "Informe o capital inicial."}
    if periods <= 0:
        return {"error": "periods must be positive", "explanation": "Informe o número de períodos."}

    balance = principal
    timeline = []
    snapshot_interval = max(1, periods // 10)  # ~10 snapshots

    for i in range(1, periods + 1):
        balance = balance * (1 + rate_pct / 100) + contribution
        if i % snapshot_interval == 0 or i == periods:
            timeline.append({"period": i, "balance": round(balance, 2)})

    total_contributions = contribution * periods
    total_invested = principal + total_contributions
    total_return = balance - total_invested
    total_return_pct = (total_return / principal) * 100 if principal > 0 else 0

    result: dict[str, Any] = {
        "principal": principal,
        "rate_pct": rate_pct,
        "periods": periods,
        "contribution_per_period": contribution,
        "final_balance": round(balance, 2),
        "total_invested": round(total_invested, 2),
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "timeline": timeline,
    }

    contrib_label = f" + {contribution:.0f}/período" if contribution > 0 else ""
    result["explanation"] = (
        f"{principal:.0f} a {rate_pct:.2f}%/período por {periods} períodos{contrib_label}: "
        f"saldo final {balance:.2f} (retorno {total_return_pct:.2f}%)."
    )
    return result


async def calc_risk_reward(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate Risk/Reward ratio.

    Required: entry_price, stop_loss, target_price
    """
    entry = float(params.get("entry_price", 0))
    stop_loss = float(params.get("stop_loss", 0))
    target = float(params.get("target_price", 0))

    if entry <= 0:
        return {"error": "entry_price must be positive", "explanation": "Informe o preço de entrada."}
    if stop_loss <= 0:
        return {"error": "stop_loss must be positive", "explanation": "Informe o stop loss."}
    if target <= 0:
        return {"error": "target_price must be positive", "explanation": "Informe o preço alvo."}

    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    ratio = reward / risk if risk > 0 else 0
    risk_pct = (risk / entry) * 100 if entry > 0 else 0
    reward_pct = (reward / entry) * 100 if entry > 0 else 0

    # Determine direction
    direction = "long" if target > entry else "short"

    result: dict[str, Any] = {
        "entry_price": entry,
        "stop_loss": stop_loss,
        "target_price": target,
        "direction": direction,
        "risk_per_unit": round(risk, 4),
        "reward_per_unit": round(reward, 4),
        "risk_pct": round(risk_pct, 2),
        "reward_pct": round(reward_pct, 2),
        "risk_reward_ratio": round(ratio, 2),
        "explanation": (
            f"Entrada {entry:.2f}, stop {stop_loss:.2f}, alvo {target:.2f} ({direction}): "
            f"risco {risk:.2f} ({risk_pct:.1f}%) / ganho {reward:.2f} ({reward_pct:.1f}%) = "
            f"R:R {ratio:.2f}:1."
        ),
    }
    return result


# ─── Registration ────────────────────────────────────────────────────────────


def register_finance_calculators(registry: CalculatorRegistry) -> None:
    """Register all basic finance calculators."""
    registry.register("calc_percentage", calc_percentage)
    registry.register("calc_average_cost", calc_average_cost)
    registry.register("calc_compound_growth", calc_compound_growth)
    registry.register("calc_risk_reward", calc_risk_reward)
    logger.info("Registered %d finance calculators", 4)
