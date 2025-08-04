"""
Scenario Templates for Multi-Agent Economics Simulation

This module implements template-based scenario generation following the design
from simulation_backend_plan.md. Templates create structured economic scenarios
with specific regime configurations, correlations, and market dynamics.

Templates available:
- Crisis: Flight to quality with high correlations
- Boom: All sectors bullish with low correlations
- Tech Shock: Tech sector divergence from others
- Decoupling: Opposite regimes across sectors
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """Configuration for a scenario template."""
    name: str
    description: str
    sectors: List[str]
    initial_regimes: Dict[str, int]
    regime_parameters: Dict[str, Dict[int, Dict[str, float]]]
    transition_matrices: Dict[str, np.ndarray]
    correlation_matrix: np.ndarray
    market_volatility_level: float
    correlation_regime: str
    expected_patterns: Dict[str, Any]


def generate_crisis_scenario(sectors: Optional[List[str]] = None, 
                           intensity: float = 1.0) -> ScenarioConfig:
    """
    Generate a financial crisis scenario.
    
    Crisis characteristics:
    - Most sectors in low-return, high-volatility regimes
    - High correlations (flight to quality, contagion effects)
    - Persistent crisis regime with slow recovery
    
    Args:
        sectors: List of sector names. Defaults to ["tech", "finance", "healthcare"]
        intensity: Crisis intensity [0.5, 1.5]. Higher = more severe crisis
        
    Returns:
        ScenarioConfig with crisis parameters
    """
    if sectors is None:
        sectors = ["tech", "finance", "healthcare"]
    
    n_sectors = len(sectors)
    
    # Crisis regimes: Most sectors start in crisis (regime 1)
    initial_regimes = {}
    for i, sector in enumerate(sectors):
        # Healthcare less affected by financial crisis
        if sector == "healthcare":
            initial_regimes[sector] = 0  # Defensive sector stays stable
        else:
            initial_regimes[sector] = 1  # Crisis regime
    
    # Regime parameters: Crisis regime has low returns, high volatility
    base_crisis_return = -0.02 * intensity  # Negative returns in crisis
    base_normal_return = 0.06
    crisis_volatility = 0.30 * intensity   # High volatility in crisis
    normal_volatility = 0.15
    
    regime_parameters = {}
    for sector in sectors:
        if sector == "tech":
            # Tech is most volatile in crisis
            regime_parameters[sector] = {
                0: {"mu": base_normal_return * 1.5, "sigma": normal_volatility * 1.2},  # Tech boom
                1: {"mu": base_crisis_return * 1.8, "sigma": crisis_volatility * 1.4}   # Tech bust
            }
        elif sector == "finance":
            # Finance is epicenter of crisis
            regime_parameters[sector] = {
                0: {"mu": base_normal_return, "sigma": normal_volatility},
                1: {"mu": base_crisis_return * 2.0, "sigma": crisis_volatility * 1.5}  # Severe crisis
            }
        elif sector == "healthcare":
            # Healthcare is defensive
            regime_parameters[sector] = {
                0: {"mu": base_normal_return * 0.8, "sigma": normal_volatility * 0.7},  # Stable
                1: {"mu": base_crisis_return * 0.3, "sigma": crisis_volatility * 0.6}   # Less affected
            }
        else:
            # Default parameters for other sectors
            regime_parameters[sector] = {
                0: {"mu": base_normal_return, "sigma": normal_volatility},
                1: {"mu": base_crisis_return, "sigma": crisis_volatility}
            }
    
    # Transition matrices: Crisis is persistent, recovery is slow
    transition_matrices = {}
    for sector in sectors:
        if sector == "healthcare":
            # Healthcare recovers faster
            transition_matrices[sector] = np.array([
                [0.85, 0.15],  # Normal -> Crisis (less likely)
                [0.40, 0.60]   # Crisis -> Normal (faster recovery)
            ])
        else:
            # Other sectors: crisis persists
            transition_matrices[sector] = np.array([
                [0.70, 0.30],  # Normal -> Crisis
                [0.20, 0.80]   # Crisis -> Normal (slow recovery)
            ])
    
    # High correlations during crisis (flight to quality, contagion)
    base_correlation = 0.3 + 0.4 * intensity  # 0.3 to 0.9 based on intensity
    correlation_matrix = np.ones((n_sectors, n_sectors)) * base_correlation
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Ensure positive definite by adjusting off-diagonal elements
    max_corr = 0.95
    correlation_matrix = np.clip(correlation_matrix, -max_corr, max_corr)
    
    return ScenarioConfig(
        name="Crisis",
        description=f"Financial crisis scenario with {intensity:.1f}x intensity",
        sectors=sectors,
        initial_regimes=initial_regimes,
        regime_parameters=regime_parameters,
        transition_matrices=transition_matrices,
        correlation_matrix=correlation_matrix,
        market_volatility_level=crisis_volatility,
        correlation_regime="high_correlation",
        expected_patterns={
            "volatility_clustering": True,
            "flight_to_quality": True,
            "regime_persistence": True,
            "negative_returns": True,
            "high_correlations": True
        }
    )


def generate_boom_scenario(sectors: Optional[List[str]] = None,
                          growth_factor: float = 1.0) -> ScenarioConfig:
    """
    Generate an economic boom scenario.
    
    Boom characteristics:
    - All sectors in high-return, low-volatility regimes
    - Low to moderate correlations (diversification benefits)
    - Optimistic agents with persistent growth expectations
    
    Args:
        sectors: List of sector names
        growth_factor: Boom intensity [0.5, 2.0]. Higher = stronger boom
        
    Returns:
        ScenarioConfig with boom parameters
    """
    if sectors is None:
        sectors = ["tech", "finance", "healthcare"]
    
    n_sectors = len(sectors)
    
    # Boom regimes: All sectors start in bull market (regime 0)
    initial_regimes = {sector: 0 for sector in sectors}
    
    # Regime parameters: Boom regime has high returns, low volatility
    base_boom_return = 0.12 * growth_factor    # High returns in boom
    base_normal_return = 0.04
    boom_volatility = 0.12  # Low volatility in boom
    normal_volatility = 0.20
    
    regime_parameters = {}
    for sector in sectors:
        if sector == "tech":
            # Tech leads the boom
            regime_parameters[sector] = {
                0: {"mu": base_boom_return * 1.6, "sigma": boom_volatility * 1.1},  # Tech boom
                1: {"mu": base_normal_return, "sigma": normal_volatility * 1.2}     # Tech correction
            }
        elif sector == "finance":
            # Finance benefits from boom
            regime_parameters[sector] = {
                0: {"mu": base_boom_return * 1.2, "sigma": boom_volatility},
                1: {"mu": base_normal_return * 0.7, "sigma": normal_volatility}
            }
        elif sector == "healthcare":
            # Healthcare steady growth
            regime_parameters[sector] = {
                0: {"mu": base_boom_return * 0.8, "sigma": boom_volatility * 0.9},
                1: {"mu": base_normal_return * 0.8, "sigma": normal_volatility * 0.8}
            }
        else:
            # Default boom parameters
            regime_parameters[sector] = {
                0: {"mu": base_boom_return, "sigma": boom_volatility},
                1: {"mu": base_normal_return, "sigma": normal_volatility}
            }
    
    # Transition matrices: Boom is persistent but can correct
    transition_matrices = {}
    for sector in sectors:
        if sector == "tech":
            # Tech more volatile (higher chance of correction)
            transition_matrices[sector] = np.array([
                [0.80, 0.20],  # Boom -> Correction
                [0.50, 0.50]   # Correction -> Boom
            ])
        else:
            # Other sectors more stable
            transition_matrices[sector] = np.array([
                [0.85, 0.15],  # Boom -> Correction
                [0.40, 0.60]   # Correction -> Boom
            ])
    
    # Low to moderate correlations (diversification works)
    base_correlation = max(0.1, 0.4 - 0.2 * growth_factor)  # Lower corr in stronger booms
    correlation_matrix = np.ones((n_sectors, n_sectors)) * base_correlation
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return ScenarioConfig(
        name="Boom",
        description=f"Economic boom scenario with {growth_factor:.1f}x growth factor",
        sectors=sectors,
        initial_regimes=initial_regimes,
        regime_parameters=regime_parameters,
        transition_matrices=transition_matrices,
        correlation_matrix=correlation_matrix,
        market_volatility_level=boom_volatility,
        correlation_regime="low_correlation",
        expected_patterns={
            "sustained_growth": True,
            "low_volatility": True,
            "diversification_benefits": True,
            "positive_returns": True,
            "low_correlations": True
        }
    )


def generate_tech_shock_scenario(sectors: Optional[List[str]] = None,
                                shock_magnitude: float = 1.0) -> ScenarioConfig:
    """
    Generate a tech sector shock scenario.
    
    Tech Shock characteristics:
    - Tech sector in extreme regime (boom or bust)
    - Strong tech-nontech divergence in correlations
    - Idiosyncratic volatility concentrated in tech
    
    Args:
        sectors: List of sector names (must include "tech")
        shock_magnitude: Shock intensity [0.5, 2.0]. Higher = more extreme
        
    Returns:
        ScenarioConfig with tech shock parameters
    """
    if sectors is None:
        sectors = ["tech", "finance", "healthcare"]
    
    if "tech" not in sectors:
        raise ValueError("Tech shock scenario requires 'tech' sector")
    
    n_sectors = len(sectors)
    
    # Tech shock: Tech in extreme regime, others normal/defensive
    initial_regimes = {}
    for sector in sectors:
        if sector == "tech":
            initial_regimes[sector] = 0 if shock_magnitude > 0 else 1  # Boom if positive shock
        else:
            initial_regimes[sector] = 0  # Others remain normal initially
    
    # Regime parameters: Extreme tech parameters, normal others
    base_return = 0.06
    base_volatility = 0.15
    
    regime_parameters = {}
    for sector in sectors:
        if sector == "tech":
            # Extreme tech parameters
            tech_boom_return = 0.25 * abs(shock_magnitude)
            tech_boom_vol = 0.35 * abs(shock_magnitude)
            tech_bust_return = -0.15 * abs(shock_magnitude)
            tech_bust_vol = 0.45 * abs(shock_magnitude)
            
            regime_parameters[sector] = {
                0: {"mu": tech_boom_return, "sigma": tech_boom_vol},    # Tech boom
                1: {"mu": tech_bust_return, "sigma": tech_bust_vol}     # Tech bust
            }
        elif sector == "finance":
            # Finance somewhat correlated with tech
            regime_parameters[sector] = {
                0: {"mu": base_return * 1.1, "sigma": base_volatility},
                1: {"mu": base_return * 0.6, "sigma": base_volatility * 1.2}
            }
        else:
            # Other sectors less affected
            regime_parameters[sector] = {
                0: {"mu": base_return, "sigma": base_volatility},
                1: {"mu": base_return * 0.8, "sigma": base_volatility * 1.1}
            }
    
    # Transition matrices: Tech regime persists, others more stable
    transition_matrices = {}
    for sector in sectors:
        if sector == "tech":
            # Tech shock persists
            transition_matrices[sector] = np.array([
                [0.90, 0.10],  # Boom persists (regime 0 -> regime 0 with prob 0.90)
                [0.15, 0.85]   # Bust persists (regime 1 -> regime 1 with prob 0.85)
            ])
        else:
            # Others more mean-reverting
            transition_matrices[sector] = np.array([
                [0.75, 0.25],
                [0.40, 0.60]
            ])
    
    # Correlation structure: Strong tech-nontech divergence
    correlation_matrix = np.eye(n_sectors) * 0.8 + np.ones((n_sectors, n_sectors)) * 0.1
    
    # Adjust tech correlations based on shock direction
    tech_idx = sectors.index("tech")
    for i, sector in enumerate(sectors):
        if sector != "tech":
            if sector == "finance":
                # Finance moderately correlated with tech
                correlation_matrix[tech_idx, i] = 0.4 * np.sign(shock_magnitude)
                correlation_matrix[i, tech_idx] = 0.4 * np.sign(shock_magnitude)
            else:
                # Other sectors weakly correlated or inverse
                correlation_matrix[tech_idx, i] = 0.1 * np.sign(shock_magnitude)
                correlation_matrix[i, tech_idx] = 0.1 * np.sign(shock_magnitude)
    
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return ScenarioConfig(
        name="Tech_Shock",
        description=f"Tech sector shock with {shock_magnitude:.1f}x magnitude",
        sectors=sectors,
        initial_regimes=initial_regimes,
        regime_parameters=regime_parameters,
        transition_matrices=transition_matrices,
        correlation_matrix=correlation_matrix,
        market_volatility_level=base_volatility,
        correlation_regime="sector_divergence",
        expected_patterns={
            "tech_divergence": True,
            "idiosyncratic_volatility": True,
            "sector_rotation": True,
            "correlation_breakdown": True
        }
    )


def generate_decoupling_scenario(sectors: Optional[List[str]] = None,
                               decoupling_strength: float = 1.0) -> ScenarioConfig:
    """
    Generate a sector decoupling scenario.
    
    Decoupling characteristics:
    - Sectors in opposite regimes
    - Mixed correlations (some positive, some negative)
    - Opportunity for dispersion trades
    
    Args:
        sectors: List of sector names
        decoupling_strength: Decoupling intensity [0.5, 2.0]
        
    Returns:
        ScenarioConfig with decoupling parameters
    """
    if sectors is None:
        sectors = ["tech", "finance", "healthcare"]
    
    n_sectors = len(sectors)
    
    # Decoupling: Alternate regimes across sectors
    initial_regimes = {}
    for i, sector in enumerate(sectors):
        initial_regimes[sector] = i % 2  # Alternate 0, 1, 0, 1, ...
    
    # Regime parameters: Enhanced differences between regimes
    base_return_high = 0.10 * decoupling_strength
    base_return_low = 0.02
    base_vol_high = 0.18
    base_vol_low = 0.12
    
    regime_parameters = {}
    for sector in sectors:
        # Amplify regime differences
        regime_parameters[sector] = {
            0: {"mu": base_return_high, "sigma": base_vol_low},   # High return, low vol
            1: {"mu": base_return_low, "sigma": base_vol_high}   # Low return, high vol
        }
    
    # Transition matrices: Regimes tend to persist (maintain decoupling)
    transition_matrices = {}
    for sector in sectors:
        persistence = 0.75 + 0.15 * decoupling_strength
        transition_matrices[sector] = np.array([
            [persistence, 1 - persistence],
            [1 - persistence, persistence]
        ])
    
    # Mixed correlation structure
    correlation_matrix = np.zeros((n_sectors, n_sectors))
    
    # Create block structure with mixed correlations
    for i in range(n_sectors):
        for j in range(n_sectors):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Alternate positive and negative correlations
                if (i + j) % 2 == 0:
                    correlation_matrix[i, j] = 0.3 * decoupling_strength
                else:
                    correlation_matrix[i, j] = -0.2 * decoupling_strength
    
    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Renormalize diagonal
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return ScenarioConfig(
        name="Decoupling",
        description=f"Sector decoupling scenario with {decoupling_strength:.1f}x strength",
        sectors=sectors,
        initial_regimes=initial_regimes,
        regime_parameters=regime_parameters,
        transition_matrices=transition_matrices,
        correlation_matrix=correlation_matrix,
        market_volatility_level=(base_vol_high + base_vol_low) / 2,
        correlation_regime="mixed_correlation",
        expected_patterns={
            "sector_rotation": True,
            "dispersion_opportunities": True,
            "mixed_correlations": True,
            "regime_persistence": True
        }
    )


def generate_scenario_grid(base_scenarios: List[str], 
                          parameter_ranges: Dict[str, List[float]]) -> Dict[str, ScenarioConfig]:
    """
    Generate a grid of scenario variations for systematic exploration.
    
    Args:
        base_scenarios: List of scenario names to vary
        parameter_ranges: Dict mapping parameter names to ranges
        
    Returns:
        Dict mapping scenario_name -> ScenarioConfig
    """
    scenarios = {}
    
    scenario_generators = {
        "crisis": generate_crisis_scenario,
        "boom": generate_boom_scenario, 
        "tech_shock": generate_tech_shock_scenario,
        "decoupling": generate_decoupling_scenario
    }
    
    for scenario_name in base_scenarios:
        if scenario_name not in scenario_generators:
            continue
            
        generator = scenario_generators[scenario_name]
        
        # Get parameter name for this scenario
        param_mapping = {
            "crisis": "intensity",
            "boom": "growth_factor",
            "tech_shock": "shock_magnitude", 
            "decoupling": "decoupling_strength"
        }
        
        param_name = param_mapping[scenario_name]
        param_values = parameter_ranges.get(param_name, [1.0])
        
        for param_value in param_values:
            scenario_key = f"{scenario_name}_{param_value:.1f}"
            
            # Generate scenario with this parameter value
            if scenario_name == "crisis":
                scenarios[scenario_key] = generator(intensity=param_value)
            elif scenario_name == "boom":
                scenarios[scenario_key] = generator(growth_factor=param_value)
            elif scenario_name == "tech_shock":
                scenarios[scenario_key] = generator(shock_magnitude=param_value)
            elif scenario_name == "decoupling":
                scenarios[scenario_key] = generator(decoupling_strength=param_value)
    
    return scenarios


def get_scenario_summary(config: ScenarioConfig) -> Dict[str, Any]:
    """
    Generate a summary of scenario characteristics for analysis.
    
    Args:
        config: Scenario configuration
        
    Returns:
        Dict with scenario summary statistics
    """
    # Calculate aggregate statistics
    all_returns = []
    all_volatilities = []
    
    for sector_params in config.regime_parameters.values():
        for regime_params in sector_params.values():
            all_returns.append(regime_params["mu"])
            all_volatilities.append(regime_params["sigma"])
    
    avg_correlation = np.mean(config.correlation_matrix[np.triu_indices_from(config.correlation_matrix, k=1)])
    
    return {
        "name": config.name,
        "description": config.description,
        "n_sectors": len(config.sectors),
        "avg_expected_return": np.mean(all_returns),
        "avg_volatility": np.mean(all_volatilities),
        "avg_correlation": avg_correlation,
        "volatility_range": (np.min(all_volatilities), np.max(all_volatilities)),
        "return_range": (np.min(all_returns), np.max(all_returns)),
        "correlation_regime": config.correlation_regime,
        "expected_patterns": config.expected_patterns
    }