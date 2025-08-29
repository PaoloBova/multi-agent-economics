from pydantic import BaseModel, Field


class RegimeParameters(BaseModel):
    """Parameters for a single regime."""
    mu: float = Field(..., description="Expected return for this regime")
    sigma: float = Field(..., description="Volatility for this regime", gt=0)


