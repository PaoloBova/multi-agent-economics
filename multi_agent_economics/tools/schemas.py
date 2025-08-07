"""
Pydantic response schemas for all tool types.

These schemas ensure type safety and consistent return formats across
both economic analysis and artifact management tools.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# Economic Tool Response Schemas

class SectorForecastResponse(BaseModel):
    """Response model for sector forecast tool."""
    sector: str = Field(..., description="Sector that was forecasted")
    horizon: int = Field(..., description="Number of periods forecasted", ge=1)
    forecast: List[float] = Field(..., description="Forecasted returns for each period")
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)


class MonteCarloVarResponse(BaseModel):
    """Response model for Monte Carlo VaR tool."""
    portfolio_value: float = Field(..., description="Portfolio value analyzed", gt=0)
    volatility: float = Field(..., description="Portfolio volatility used", ge=0)
    confidence_level: float = Field(..., description="Confidence level for VaR calculation", ge=0, le=1)
    var_estimate: float = Field(..., description="Value at Risk estimate", ge=0)
    expected_shortfall: float = Field(..., description="Expected shortfall (CVaR)", ge=0)
    max_loss: float = Field(..., description="Maximum simulated loss", ge=0)
    expected_loss: float = Field(..., description="Expected loss from simulation")
    n_simulations: int = Field(..., description="Number of simulations performed", gt=0)
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_requested: float = Field(..., description="Originally requested effort level", ge=0)
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or constraints applied")


class PriceNoteResponse(BaseModel):
    """Response model for structured note pricing tool."""
    notional: float = Field(..., description="Notional amount of the note", gt=0)
    payoff_type: str = Field(..., description="Type of payoff structure")
    fair_value: float = Field(..., description="Calculated fair value", ge=0)
    quoted_price: float = Field(..., description="Final quoted price with error adjustment", ge=0)
    pricing_error: float = Field(..., description="Pricing error applied based on quality")
    pricing_accuracy: float = Field(..., description="Pricing accuracy metric", ge=0, le=1)
    expected_return: float = Field(..., description="Expected return used in calculation")
    discount_rate: float = Field(..., description="Discount rate used", ge=0)
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_requested: float = Field(..., description="Originally requested effort level", ge=0)
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or constraints applied")


# Artifact Tool Response Schemas

class ArtifactLoadResponse(BaseModel):
    """Response model for artifact load tool."""
    status: Literal["loaded", "error"] = Field(..., description="Operation status")
    artifact_id: str = Field(..., description="ID of the artifact")
    message: str = Field(..., description="Human-readable status message")
    version: Optional[int] = Field(None, description="Version of loaded artifact")


class ArtifactUnloadResponse(BaseModel):
    """Response model for artifact unload tool."""
    status: Literal["unloaded", "error"] = Field(..., description="Operation status")
    artifact_id: str = Field(..., description="ID of the artifact")
    message: str = Field(..., description="Human-readable status message")


class ArtifactWriteResponse(BaseModel):
    """Response model for artifact write tool."""
    status: Literal["written", "error"] = Field(..., description="Operation status")
    artifact_id: str = Field(..., description="ID of the artifact")
    version: Optional[int] = Field(None, description="Version of written artifact")
    path: Optional[str] = Field(None, description="Storage path of artifact")
    message: str = Field(..., description="Human-readable status message")
    size_chars: Optional[int] = Field(None, description="Size of written content in characters")



class ArtifactListResponse(BaseModel):
    """Response model for artifact list tool."""
    status: Literal["success", "error"] = Field(..., description="Operation status")
    workspace_listing: str = Field(..., description="Formatted workspace listing")
    loaded_artifacts: List[str] = Field(..., description="List of currently loaded artifact IDs")
    loaded_status: Dict[str, Any] = Field(..., description="Status details for loaded artifacts")
    message: str = Field(..., description="Human-readable instruction message")
    total_artifacts: Optional[int] = Field(None, description="Total number of available artifacts")


# Utility Models for Complex Tool Inputs

class PayoffFunction(BaseModel):
    """Model for structured note payoff function parameters."""
    type: Literal["linear", "barrier", "autocall", "digital"] = Field(..., description="Payoff type")
    notional: float = Field(..., description="Notional amount", gt=0)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Type-specific parameters")


class ForecastInput(BaseModel):
    """Model for forecast input data."""
    asset: str = Field(..., description="Asset or sector identifier")
    horizon: int = Field(..., description="Forecast horizon", ge=1)
    confidence_level: float = Field(0.95, description="Desired confidence level", ge=0.5, le=0.99)


# Error Response Model (for consistency)

class ToolErrorResponse(BaseModel):
    """Standard error response for any tool."""
    status: Literal["error"] = Field("error", description="Error status")
    error_type: str = Field(..., description="Type of error encountered")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")