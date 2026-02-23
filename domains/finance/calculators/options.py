"""
Options Calculators — Generic deterministic math for options trading.

Functions:
- calc_put_return: Return metrics for a put position (ROC, breakeven, cushion)
- calc_contract_capacity: How many contracts given capital + allocation %
- calc_put_risk: Downside scenario analysis for put positions
- calc_required_premium: Inverse — target return % → required premium
- calc_income_target: Monthly/weekly income targets from capital + return goal
- calc_annualized_return: Annualize a period return
- calc_margin_collateral: Total margin/collateral required
"""

import logging
from typing import Any

from domains.finance.tiers.calculators import CalculatorRegistry, DataFetcher

logger = logging.getLogger(__name__)


# ─── Migrated from wheel.py (renamed, generalized) ───────────────────────────


async def calc_put_return(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate return metrics for a put position.

    Required: symbol, strike, premium
    Optional: lot_size (default 100)
    """
    symbol = str(params.get("symbol", "")).strip()
    strike = float(params.get("strike", 0))
    premium = float(params.get("premium", 0))
    lot_size = int(params.get("lot_size", 100))

    if strike <= 0:
        return {"error": "Strike price must be positive", "explanation": "Informe um strike válido."}
    if premium <= 0:
        return {"error": "Premium must be positive", "explanation": "Informe um prêmio válido."}

    # Fetch current price for context
    current_price = _fetch_current_price(fetcher, symbol)

    # Core calculations
    collateral = strike * lot_size
    total_premium = premium * lot_size
    return_on_collateral = (premium / strike) * 100
    breakeven = strike - premium
    downside_cushion = ((strike - breakeven) / strike) * 100

    result: dict[str, Any] = {
        "symbol": symbol,
        "strike": strike,
        "premium": premium,
        "lot_size": lot_size,
        "collateral_required": round(collateral, 2),
        "total_premium_received": round(total_premium, 2),
        "return_on_collateral_pct": round(return_on_collateral, 4),
        "breakeven_price": round(breakeven, 2),
        "downside_cushion_pct": round(downside_cushion, 2),
    }

    if current_price is not None:
        otm_pct = ((current_price - strike) / current_price) * 100 if current_price > 0 else 0
        result["current_price"] = round(current_price, 2)
        result["otm_pct"] = round(otm_pct, 2)

    result["explanation"] = (
        f"Put {symbol} strike {strike}: prêmio {premium}, "
        f"retorno {return_on_collateral:.2f}% sobre colateral de {collateral:.0f}. "
        f"Breakeven em {breakeven:.2f}."
    )
    return result


async def calc_contract_capacity(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate how many contracts you can sell given capital.

    Required: symbol, capital
    Optional: allocation_pct (1.0), strike, margin_requirement_pct (1.0), lot_size (100)
    """
    symbol = str(params.get("symbol", "")).strip()
    capital = float(params.get("capital", 0) or params.get("capital_sek", 0))
    allocation_pct = float(params.get("allocation_pct", 1.0))
    margin_req = float(params.get("margin_requirement_pct", 1.0))
    lot_size = int(params.get("lot_size", 100))

    if capital <= 0:
        return {"error": "Capital must be positive", "explanation": "Informe um valor de capital válido."}
    if not 0 < allocation_pct <= 1.0:
        return {"error": "allocation_pct must be between 0 and 1", "explanation": "Informe um percentual válido (ex: 0.2 para 20%)."}

    # Effective capital after allocation
    effective_capital = capital * allocation_pct

    # Get strike from params or fetch current price
    strike = params.get("strike")
    current_price = _fetch_current_price(fetcher, symbol)

    if strike is None or float(strike) <= 0:
        if current_price:
            strike = current_price * 0.95  # Default: 5% OTM
        else:
            return {"error": "Strike required when price unavailable", "explanation": "Informe o strike."}
    else:
        strike = float(strike)

    # Core calculations
    collateral_per_contract = strike * lot_size * margin_req
    max_contracts = int(effective_capital / collateral_per_contract) if collateral_per_contract > 0 else 0
    total_collateral = max_contracts * collateral_per_contract
    remaining = effective_capital - total_collateral

    result: dict[str, Any] = {
        "symbol": symbol,
        "total_capital": round(capital, 2),
        "allocation_pct": allocation_pct,
        "effective_capital": round(effective_capital, 2),
        "strike": round(strike, 2),
        "lot_size": lot_size,
        "margin_requirement_pct": margin_req,
        "collateral_per_contract": round(collateral_per_contract, 2),
        "max_contracts": max_contracts,
        "total_collateral_used": round(total_collateral, 2),
        "remaining_capital": round(remaining, 2),
        "utilization_pct": round((total_collateral / effective_capital) * 100, 2) if effective_capital > 0 else 0,
    }

    if current_price:
        result["current_price"] = round(current_price, 2)

    alloc_label = f" ({allocation_pct*100:.0f}% do capital)" if allocation_pct < 1.0 else ""
    result["explanation"] = (
        f"Com {effective_capital:.0f}{alloc_label} de capital, você pode vender {max_contracts} contratos "
        f"de {symbol} no strike {strike:.2f} (colateral total: {total_collateral:.0f})."
    )
    return result


async def calc_put_risk(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Analyze downside risk scenarios for put positions.

    Required: symbol
    Optional: pct_below_spot (5.0), lot_size (100)
    """
    symbol = str(params.get("symbol", "")).strip()
    pct_below = float(params.get("pct_below_spot", 5.0))
    lot_size = int(params.get("lot_size", 100))

    current_price = _fetch_current_price(fetcher, symbol)
    if not current_price:
        return {
            "error": "Could not determine current price",
            "explanation": f"Não consegui obter o preço atual de {symbol} para análise de risco.",
        }

    # Calculate risk scenarios
    strike = current_price * (1 - pct_below / 100)
    scenarios = []
    for drawdown_pct in [5, 10, 15, 20, 30]:
        price_at_drawdown = current_price * (1 - drawdown_pct / 100)
        loss_per_share = max(0, strike - price_at_drawdown)
        loss_total = loss_per_share * lot_size
        scenarios.append({
            "drawdown_pct": drawdown_pct,
            "price_at_drawdown": round(price_at_drawdown, 2),
            "loss_per_share": round(loss_per_share, 2),
            "loss_total": round(loss_total, 2),
            "in_the_money": price_at_drawdown < strike,
        })

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "strike": round(strike, 2),
        "pct_below_spot": pct_below,
        "lot_size": lot_size,
        "scenarios": scenarios,
        "explanation": (
            f"Risco do put {symbol}: strike {strike:.2f} "
            f"({pct_below:.1f}% abaixo do spot {current_price:.2f}). "
            f"Cenários de drawdown analisados."
        ),
    }


# ─── New calculators ─────────────────────────────────────────────────────────


async def calc_required_premium(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate required premium for a target return percentage.

    Inverse of calc_put_return: given target % and strike → required premium.

    Required: strike, target_return_pct
    Optional: lot_size (100), days_to_expiry (for annualization)
    """
    strike = float(params.get("strike", 0))
    target_return_pct = float(params.get("target_return_pct", 0))
    lot_size = int(params.get("lot_size", 100))
    days_to_expiry = params.get("days_to_expiry")

    if strike <= 0:
        return {"error": "Strike must be positive", "explanation": "Informe um strike válido."}
    if target_return_pct <= 0:
        return {"error": "Target return must be positive", "explanation": "Informe um percentual de retorno válido."}

    premium = strike * (target_return_pct / 100)
    total_premium = premium * lot_size
    collateral = strike * lot_size

    result: dict[str, Any] = {
        "strike": strike,
        "target_return_pct": target_return_pct,
        "required_premium_per_share": round(premium, 4),
        "total_premium": round(total_premium, 2),
        "collateral": round(collateral, 2),
        "lot_size": lot_size,
    }

    if days_to_expiry is not None and int(days_to_expiry) > 0:
        dte = int(days_to_expiry)
        annualized = ((1 + target_return_pct / 100) ** (365 / dte) - 1) * 100
        result["days_to_expiry"] = dte
        result["annualized_return_pct"] = round(annualized, 2)

    result["explanation"] = (
        f"Para {target_return_pct:.2f}% de retorno no strike {strike:.2f}, "
        f"o prêmio mínimo é {premium:.4f} por ação ({total_premium:.2f} total por contrato)."
    )
    return result


async def calc_income_target(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate weekly/monthly income targets given capital and return goal.

    Required: capital, and one of (target_monthly_pct, target_annual_pct)
    Optional: num_contracts (for per-contract breakdown)
    """
    capital = float(params.get("capital", 0))
    target_monthly_pct = params.get("target_monthly_pct")
    target_annual_pct = params.get("target_annual_pct")
    num_contracts = params.get("num_contracts")

    if capital <= 0:
        return {"error": "Capital must be positive", "explanation": "Informe o capital disponível."}

    # Determine monthly target
    if target_annual_pct is not None:
        monthly_pct = float(target_annual_pct) / 12
    elif target_monthly_pct is not None:
        monthly_pct = float(target_monthly_pct)
    else:
        return {"error": "Provide target_monthly_pct or target_annual_pct", "explanation": "Informe a meta de retorno (mensal ou anual)."}

    if monthly_pct <= 0:
        return {"error": "Target must be positive", "explanation": "Informe uma meta positiva."}

    weeks_per_month = 4.33
    trading_days_per_month = 21

    monthly_income = capital * (monthly_pct / 100)
    weekly_income = monthly_income / weeks_per_month
    daily_income = monthly_income / trading_days_per_month
    annual_income = monthly_income * 12
    annual_pct = monthly_pct * 12

    result: dict[str, Any] = {
        "capital": round(capital, 2),
        "target_monthly_pct": round(monthly_pct, 4),
        "target_annual_pct": round(annual_pct, 2),
        "monthly_income": round(monthly_income, 2),
        "weekly_income": round(weekly_income, 2),
        "daily_income_trading": round(daily_income, 2),
        "annual_income": round(annual_income, 2),
    }

    if num_contracts is not None and int(num_contracts) > 0:
        nc = int(num_contracts)
        result["num_contracts"] = nc
        result["income_per_contract_weekly"] = round(weekly_income / nc, 2)
        result["income_per_contract_monthly"] = round(monthly_income / nc, 2)

    result["explanation"] = (
        f"Com {capital:.0f} de capital e meta de {monthly_pct:.2f}%/mês ({annual_pct:.1f}%/ano): "
        f"ganhar {monthly_income:.2f}/mês, {weekly_income:.2f}/semana, {daily_income:.2f}/dia."
    )
    return result


async def calc_annualized_return(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Annualize a return from a specific period.

    Required: return_pct, period_days
    Optional: num_periods (for cumulative calc)
    """
    return_pct = float(params.get("return_pct", 0))
    period_days = int(params.get("period_days", 0))
    num_periods = params.get("num_periods")

    if period_days <= 0:
        return {"error": "period_days must be positive", "explanation": "Informe o número de dias do período."}

    annualized = ((1 + return_pct / 100) ** (365 / period_days) - 1) * 100
    periods_per_year = 365 / period_days

    result: dict[str, Any] = {
        "return_pct": return_pct,
        "period_days": period_days,
        "annualized_return_pct": round(annualized, 2),
        "periods_per_year": round(periods_per_year, 1),
    }

    if num_periods is not None and int(num_periods) > 0:
        np_val = int(num_periods)
        cumulative = ((1 + return_pct / 100) ** np_val - 1) * 100
        result["num_periods"] = np_val
        result["cumulative_return_pct"] = round(cumulative, 2)

    result["explanation"] = (
        f"{return_pct:.2f}% em {period_days} dias = {annualized:.2f}% anualizado "
        f"({periods_per_year:.1f} períodos/ano)."
    )
    return result


async def calc_margin_collateral(params: dict[str, Any], fetcher: DataFetcher) -> dict[str, Any]:
    """
    Calculate total margin/collateral required for option positions.

    Required: strike
    Optional: lot_size (100), num_contracts (1), margin_pct (1.0)
    """
    strike = float(params.get("strike", 0))
    lot_size = int(params.get("lot_size", 100))
    num_contracts = int(params.get("num_contracts", 1))
    margin_pct = float(params.get("margin_pct", 1.0))

    if strike <= 0:
        return {"error": "Strike must be positive", "explanation": "Informe um strike válido."}
    if num_contracts <= 0:
        return {"error": "num_contracts must be positive", "explanation": "Informe o número de contratos."}

    collateral_per_contract = strike * lot_size * margin_pct
    total_collateral = collateral_per_contract * num_contracts
    notional_per_contract = strike * lot_size

    result: dict[str, Any] = {
        "strike": strike,
        "lot_size": lot_size,
        "num_contracts": num_contracts,
        "margin_pct": margin_pct,
        "notional_per_contract": round(notional_per_contract, 2),
        "collateral_per_contract": round(collateral_per_contract, 2),
        "total_collateral": round(total_collateral, 2),
    }

    margin_label = "cash-secured" if margin_pct >= 1.0 else f"{margin_pct*100:.0f}% margem"
    result["explanation"] = (
        f"{num_contracts} contrato(s) no strike {strike:.2f} ({margin_label}): "
        f"colateral total = {total_collateral:.2f}."
    )
    return result


# ─── Helper ──────────────────────────────────────────────────────────────────


def _fetch_current_price(fetcher: DataFetcher, symbol: str) -> float | None:
    """Fetch current price from MCP, return None on failure."""
    if not symbol:
        return None
    try:
        price_data = fetcher.fetch_data("get_stock_price", {"symbol": symbol})
        if isinstance(price_data, dict):
            raw = price_data.get("price") or price_data.get("regularMarketPrice")
            if raw is not None:
                return float(raw)
    except Exception as e:
        logger.warning("Could not fetch price for %s: %s", symbol, e)
    return None


# ─── Registration ────────────────────────────────────────────────────────────


def register_options_calculators(registry: CalculatorRegistry) -> None:
    """Register all generic options calculators."""
    registry.register("calc_put_return", calc_put_return)
    registry.register("calc_contract_capacity", calc_contract_capacity)
    registry.register("calc_put_risk", calc_put_risk)
    registry.register("calc_required_premium", calc_required_premium)
    registry.register("calc_income_target", calc_income_target)
    registry.register("calc_annualized_return", calc_annualized_return)
    registry.register("calc_margin_collateral", calc_margin_collateral)
    logger.info("Registered %d options calculators", 7)
