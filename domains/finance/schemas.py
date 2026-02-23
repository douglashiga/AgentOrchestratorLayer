from typing import Literal
from pydantic import BaseModel, Field

class StockPriceInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g. AAPL, VALE3.SA)")

class HistoricalDataInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: str = Field(default="1mo", description="Period (e.g. 1d, 5d, 1mo, 1y)")
    interval: str = Field(default="1d", description="Interval (e.g. 1d, 1wk, 1mo)")

class TopGainersInput(BaseModel):
    market: str = Field(..., description="Market code (US, BR, SE, HK)")
    period: str = Field(default="1d", description="Period to calculate gain")
    limit: int = Field(default=10, description="Number of items to return")

class TopLosersInput(BaseModel):
    market: str = Field(..., description="Market code (US, BR, SE, HK)")
    period: str = Field(default="1d", description="Period to calculate loss")
    limit: int = Field(default=10, description="Number of items to return")

class TechnicalSignalsInput(BaseModel):
    market: str = Field(..., description="Market code (US, BR, SE)")
    signal_type: str = Field(..., description="Signal type (rsi_oversold, rsi_overbought, macd_cross)")
    limit: int = Field(default=10, description="Number of items")

class StockScreenerInput(BaseModel):
    market: str = Field(..., description="Market code")
    sector: str | None = Field(default=None, description="Sector filter")
    sort_by: str = Field(default="market_cap", description="Sort criteria")
    limit: int = Field(default=10, description="Number of items")

class DividendHistoryInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: str = Field(default="2y", description="Period to fetch dividend history (e.g. 1y, 2y)")

class TriggerJobInput(BaseModel):
    job_name: str = Field(..., description="Name of the job to trigger")

class EmptyInput(BaseModel):
    pass

# ─── Expanded Screeners ────────────────────────────────────────────────────────

class MostActiveInput(BaseModel):
    market: str = Field(..., description="Market code (US, BR, SE)")
    period: str = Field(default="1D", description="Period (1D, 5D)")
    limit: int = Field(default=10, description="Number of items")

class OversoldOverboughtInput(BaseModel):
    market: str = Field(..., description="Market code (US, BR, SE)")
    limit: int = Field(default=10, description="Number of items")

# ─── Fundamentals Intelligence ─────────────────────────────────────────────────

class AnalystRecommendationsInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")

class TechnicalAnalysisInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: str = Field(default="1y", description="Analysis period (e.g. 1y, 6mo)")

class NewsSentimentInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")

class ComprehensiveStockInfoInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")

# ─── Wheel Strategy ────────────────────────────────────────────────────────────

class WheelPutCandidatesInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    market: str = Field(default="sweden", description="Market (sweden, brazil, usa)")
    delta_min: float = Field(default=0.25, description="Minimum delta for put selection")
    delta_max: float = Field(default=0.35, description="Maximum delta for put selection")
    dte_min: int = Field(default=4, description="Minimum days to expiration")
    dte_max: int = Field(default=10, description="Maximum days to expiration")
    limit: int = Field(default=5, description="Number of candidates to return")
    require_liquidity: bool = Field(default=True, description="Filter for liquid contracts only")

class WheelPutReturnInput(BaseModel):
    """Legacy alias — use PutReturnInput instead."""
    symbol: str = Field(..., description="Stock symbol")
    strike: float = Field(..., description="Put strike price")
    expiry: str = Field(..., description="Expiry date (YYYY-MM-DD)")
    premium: float = Field(..., description="Premium received per share")

# ─── Generic Options Math ─────────────────────────────────────────────────────

class PutReturnInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g. AAPL, NDA.ST)")
    strike: float = Field(..., description="Put strike price")
    premium: float = Field(..., description="Premium received per share")
    lot_size: int = Field(default=100, description="Shares per contract")

class ContractCapacityInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    capital: float = Field(..., description="Available capital")
    allocation_pct: float = Field(default=1.0, description="Fraction of capital to use (0.2 = 20%)")
    strike: float | None = Field(default=None, description="Specific strike (optional, defaults to ~5% OTM)")
    margin_requirement_pct: float = Field(default=1.0, description="Margin requirement fraction")
    lot_size: int = Field(default=100, description="Shares per contract")

class PutRiskInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    pct_below_spot: float = Field(default=5.0, description="Percentage below spot for strike calculation")
    lot_size: int = Field(default=100, description="Shares per contract")

class RequiredPremiumInput(BaseModel):
    strike: float = Field(..., description="Strike price")
    target_return_pct: float = Field(..., description="Target return on collateral (%)")
    lot_size: int = Field(default=100, description="Shares per contract")
    days_to_expiry: int | None = Field(default=None, description="DTE for annualization")

class IncomeTargetInput(BaseModel):
    capital: float = Field(..., description="Total capital available")
    target_monthly_pct: float | None = Field(default=None, description="Target monthly return (%)")
    target_annual_pct: float | None = Field(default=None, description="Target annual return (%)")
    num_contracts: int | None = Field(default=None, description="Number of contracts for per-contract breakdown")

class AnnualizedReturnInput(BaseModel):
    return_pct: float = Field(..., description="Period return (%)")
    period_days: int = Field(..., description="Period length in days")
    num_periods: int | None = Field(default=None, description="Number of periods for cumulative calc")

class MarginCollateralInput(BaseModel):
    strike: float = Field(..., description="Strike price")
    lot_size: int = Field(default=100, description="Shares per contract")
    num_contracts: int = Field(default=1, description="Number of contracts")
    margin_pct: float = Field(default=1.0, description="1.0=cash-secured, 0.2=margin")

# ─── Basic Finance Math ───────────────────────────────────────────────────────

class PercentageInput(BaseModel):
    value: float | None = Field(default=None, description="Base value (for 'X% of Y')")
    percentage: float | None = Field(default=None, description="Percentage to apply")
    part: float | None = Field(default=None, description="Part value (for 'X is what % of Y')")
    whole: float | None = Field(default=None, description="Whole value")

class AverageCostInput(BaseModel):
    lots: list[dict] = Field(..., description="List of {quantity, price} dicts")
    premium_received: float = Field(default=0, description="Total premium received (adjusts cost basis)")

class CompoundGrowthInput(BaseModel):
    principal: float = Field(..., description="Initial capital")
    rate_pct: float = Field(..., description="Return per period (%)")
    periods: int = Field(..., description="Number of periods")
    contribution_per_period: float = Field(default=0, description="Additional contribution each period")

class RiskRewardInput(BaseModel):
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    target_price: float = Field(..., description="Target/take profit price")

class WheelCoveredCallCandidatesInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol (must be assigned/owned)")
    average_cost: float = Field(..., description="Average cost basis per share")
    market: str = Field(default="sweden", description="Market (sweden, brazil, usa)")
    delta_min: float = Field(default=0.25, description="Minimum delta for call selection")
    delta_max: float = Field(default=0.35, description="Maximum delta for call selection")
    dte_min: int = Field(default=4, description="Minimum days to expiration")
    dte_max: int = Field(default=21, description="Maximum days to expiration")
    min_upside_pct: float = Field(default=1.0, description="Minimum upside percentage required")
    limit: int = Field(default=5, description="Number of candidates to return")

class WheelContractCapacityInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    capital_sek: float = Field(..., description="Available capital in SEK")
    market: str = Field(default="sweden", description="Market")
    strike: float | None = Field(default=None, description="Specific strike price (optional)")
    margin_requirement_pct: float = Field(default=1.0, description="Margin requirement as fraction of notional")
    cash_buffer_pct: float = Field(default=0.0, description="Cash buffer percentage to keep aside")
    target_dte: int = Field(default=7, description="Target days to expiration")

class WheelMultiStockPlanInput(BaseModel):
    capital_sek: float = Field(..., description="Total capital to allocate in SEK")
    symbols: list[str] | None = Field(default=None, description="List of symbols to consider (None = auto-select)")
    market: str = Field(default="sweden", description="Market")
    delta_min: float = Field(default=0.25, description="Minimum delta")
    delta_max: float = Field(default=0.35, description="Maximum delta")
    dte_min: int = Field(default=4, description="Minimum DTE")
    dte_max: int = Field(default=10, description="Maximum DTE")
    margin_requirement_pct: float = Field(default=1.0, description="Margin requirement fraction")
    cash_buffer_pct: float = Field(default=0.10, description="Cash buffer percentage")

class WheelPutRiskInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    market: str = Field(default="sweden", description="Market")
    pct_below_spot: float = Field(default=5.0, description="Percentage below spot to analyze risk")
    target_dte: int = Field(default=7, description="Target days to expiration")
