"""
Pydantic response schemas for all tool types.

These schemas ensure type safety and consistent return formats across
both economic analysis and artifact management tools.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from ..models.market_for_finance import ForecastData


# Economic Tool Response Schemas

class SectorForecastResponse(BaseModel):
    """Response model for sector forecast tool."""
    sector: str = Field(..., description="Sector that was forecasted")
    forecast: ForecastData = Field(..., description="Forecast data with regime prediction and confidence")
    quality_attributes: Dict[str, Any] = Field(..., description="Quality attributes of the forecast (methodology, coverage)")
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)

class PostToMarketResponse(BaseModel):
    """Response model for posting offers to market."""
    offer: Any = Field(..., description="The Offer object that was posted to the market") 
    status: Literal["success", "error"] = Field(..., description="Operation status")
    message: str = Field(..., description="Human-readable status message")


# Market Research Tool Response Schemas

class HistoricalPerformanceResponse(BaseModel):
    """Response model for historical performance analysis tool - returns raw trade data."""
    sector: str = Field(..., description="Sector that was analyzed")
    trade_data: List[Dict[str, Any]] = Field(..., description="Raw historical trade data for agent analysis")
    sample_size: int = Field(..., description="Number of trades returned", ge=0)
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or data limitations")
    recommendation: str = Field(..., description="Brief summary of data available for analysis")


class AttributeAnalysis(BaseModel):
    """Analysis result for a single attribute."""
    attribute_name: str = Field(..., description="Name of the marketing attribute")
    marginal_wtp_impact: Optional[float] = Field(None, description="Estimated marginal impact on WTP (regression coefficient)")
    average_feature_value: Optional[float] = Field(None, description="Average feature value observed")
    sample_size: int = Field(..., description="Number of observations for this attribute", ge=0)
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Statistical confidence in the estimate")

class WTPDataPoint(BaseModel):
    """Individual willingness-to-pay observation."""
    buyer_id: str = Field(..., description="Identifier for the buyer")
    offer_attributes: Dict[str, Any] = Field(..., description="Marketing attributes of the test offer")
    willingness_to_pay: float = Field(..., description="Calculated WTP for this buyer-offer combination")

class BuyerPreferenceResponse(BaseModel):
    """Response model for buyer preference analysis tool."""
    sector: str = Field(..., description="Sector that was analyzed")
    analysis_method: Literal["regression", "descriptive", "insufficient_data"] = Field(..., description="Method used for analysis")
    attribute_insights: List[AttributeAnalysis] = Field(default_factory=list, description="Statistical insights for each attribute")
    regression_r_squared: Optional[float] = Field(None, description="R-squared value for regression analysis (if applicable)")
    raw_wtp_data: List[WTPDataPoint] = Field(default_factory=list, description="Raw WTP observations (if requested)")
    sample_size: int = Field(..., description="Number of buyers analyzed", ge=0)
    total_observations: int = Field(..., description="Total buyer-offer combinations tested", ge=0)
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or data limitations")
    recommendation: str = Field(..., description="Actionable recommendation based on analysis")


class CompetitivePricingResponse(BaseModel):
    """Response model for competitive pricing research tool."""
    sector: str = Field(..., description="Sector that was analyzed")
    price_simulations: List[Dict[str, Any]] = Field(..., description="Competitive simulation results showing market share at different price points")
    recommended_price: float = Field(..., description="Recommended optimal price based on competitive simulation", ge=0)
    quality_tier: Literal["low", "medium", "high"] = Field(..., description="Quality tier based on effort")
    effort_used: float = Field(..., description="Actual effort used after budget constraints", ge=0)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or data limitations")
    recommendation: str = Field(..., description="Competitive pricing recommendation with market share projections")


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