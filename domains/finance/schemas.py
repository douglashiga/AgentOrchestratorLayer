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

class CompareFundamentalsInput(BaseModel):
    symbols: list[str] = Field(..., description="List of symbols to compare")
    metrics: list[str] | None = Field(default=None, description="Specific metrics to compare")

class TriggerJobInput(BaseModel):
    job_name: str = Field(..., description="Name of the job to trigger")

class EmptyInput(BaseModel):
    pass
