"""
Test suite for scenario templates.

Following TDD principles: comprehensive tests for all scenario generation
functions, parameter validation, and integration testing.
"""

import pytest
import numpy as np
from typing import Dict, List

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.models.scenario_templates import (
    ScenarioConfig,
    generate_crisis_scenario,
    generate_boom_scenario,
    generate_tech_shock_scenario,
    generate_decoupling_scenario,
    generate_scenario_grid,
    get_scenario_summary
)


class TestScenarioConfig:
    """Test ScenarioConfig dataclass."""
    
    def test_scenario_config_creation(self):
        """Test basic ScenarioConfig creation."""
        config = ScenarioConfig(
            name="Test",
            description="Test scenario",
            sectors=["tech", "finance"],
            initial_regimes={"tech": 0, "finance": 1},
            regime_parameters={"tech": {0: {"mu": 0.1, "sigma": 0.2}}},
            transition_matrices={"tech": np.array([[0.8, 0.2], [0.3, 0.7]])},
            correlation_matrix=np.eye(2),
            market_volatility_level=0.15,
            correlation_regime="normal",
            expected_patterns={"test": True}
        )
        
        assert config.name == "Test"
        assert len(config.sectors) == 2
        assert config.initial_regimes["tech"] == 0
        assert config.expected_patterns["test"] is True


class TestCrisisScenario:
    """Test crisis scenario generation."""
    
    def test_crisis_scenario_default(self):
        """Test crisis scenario with default parameters."""
        config = generate_crisis_scenario()
        
        assert config.name == "Crisis"
        assert len(config.sectors) == 3
        assert "tech" in config.sectors
        assert "finance" in config.sectors
        assert "healthcare" in config.sectors
        
        # Most sectors should start in crisis (regime 1)
        crisis_sectors = sum(1 for regime in config.initial_regimes.values() if regime == 1)
        assert crisis_sectors >= 2  # At least 2 sectors in crisis
        
        # Healthcare should be defensive (regime 0)
        assert config.initial_regimes["healthcare"] == 0
        
    def test_crisis_scenario_intensity_effects(self):
        """Test that crisis intensity affects parameters correctly."""
        mild_crisis = generate_crisis_scenario(intensity=0.5)
        severe_crisis = generate_crisis_scenario(intensity=1.5)
        
        # Severe crisis should have more negative returns
        tech_mild_return = mild_crisis.regime_parameters["tech"][1]["mu"]
        tech_severe_return = severe_crisis.regime_parameters["tech"][1]["mu"]
        assert tech_severe_return < tech_mild_return
        
        # Severe crisis should have higher volatility
        tech_mild_vol = mild_crisis.regime_parameters["tech"][1]["sigma"]
        tech_severe_vol = severe_crisis.regime_parameters["tech"][1]["sigma"]
        assert tech_severe_vol > tech_mild_vol
        
        # Severe crisis should have higher correlations
        mild_avg_corr = np.mean(mild_crisis.correlation_matrix[np.triu_indices_from(mild_crisis.correlation_matrix, k=1)])
        severe_avg_corr = np.mean(severe_crisis.correlation_matrix[np.triu_indices_from(severe_crisis.correlation_matrix, k=1)])
        assert severe_avg_corr > mild_avg_corr
        
    def test_crisis_scenario_custom_sectors(self):
        """Test crisis scenario with custom sectors."""
        custom_sectors = ["energy", "utilities", "materials"]
        config = generate_crisis_scenario(sectors=custom_sectors)
        
        assert config.sectors == custom_sectors
        assert len(config.initial_regimes) == 3
        assert len(config.regime_parameters) == 3
        assert len(config.transition_matrices) == 3
        assert config.correlation_matrix.shape == (3, 3)
        
    def test_crisis_regime_parameters_structure(self):
        """Test that crisis regime parameters have correct structure."""
        config = generate_crisis_scenario()
        
        for sector in config.sectors:
            assert sector in config.regime_parameters
            regime_params = config.regime_parameters[sector]
            
            # Should have 2 regimes (0: normal, 1: crisis)
            assert 0 in regime_params
            assert 1 in regime_params
            
            # Each regime should have mu and sigma
            for regime in [0, 1]:
                assert "mu" in regime_params[regime]
                assert "sigma" in regime_params[regime]
                assert isinstance(regime_params[regime]["mu"], float)
                assert isinstance(regime_params[regime]["sigma"], float)
                assert regime_params[regime]["sigma"] > 0  # Positive volatility
    
    def test_crisis_transition_matrices(self):
        """Test crisis transition matrices are valid."""
        config = generate_crisis_scenario()
        
        for sector in config.sectors:
            matrix = config.transition_matrices[sector]
            
            # Should be 2x2 matrix
            assert matrix.shape == (2, 2)
            
            # Rows should sum to 1 (probabilities)
            assert np.allclose(matrix.sum(axis=1), 1.0)
            
            # All entries should be non-negative
            assert np.all(matrix >= 0)
            
            # Crisis should be persistent (high probability of staying in crisis)
            assert matrix[1, 1] > 0.5  # P(crisis -> crisis) > 0.5


class TestBoomScenario:
    """Test boom scenario generation."""
    
    def test_boom_scenario_default(self):
        """Test boom scenario with default parameters."""
        config = generate_boom_scenario()
        
        assert config.name == "Boom"
        assert config.correlation_regime == "low_correlation"
        
        # All sectors should start in boom (regime 0)
        assert all(regime == 0 for regime in config.initial_regimes.values())
        
        # Expected patterns should include boom characteristics
        assert config.expected_patterns["sustained_growth"] is True
        assert config.expected_patterns["low_volatility"] is True
        assert config.expected_patterns["positive_returns"] is True
        
    def test_boom_scenario_growth_factor_effects(self):
        """Test that growth factor affects boom parameters."""
        modest_boom = generate_boom_scenario(growth_factor=0.7)
        strong_boom = generate_boom_scenario(growth_factor=1.5)
        
        # Strong boom should have higher returns
        tech_modest_return = modest_boom.regime_parameters["tech"][0]["mu"]
        tech_strong_return = strong_boom.regime_parameters["tech"][0]["mu"]
        assert tech_strong_return > tech_modest_return
        
        # Strong boom should have lower correlations (more diversification)
        modest_avg_corr = np.mean(modest_boom.correlation_matrix[np.triu_indices_from(modest_boom.correlation_matrix, k=1)])
        strong_avg_corr = np.mean(strong_boom.correlation_matrix[np.triu_indices_from(strong_boom.correlation_matrix, k=1)])
        assert strong_avg_corr < modest_avg_corr
        
    def test_boom_positive_returns(self):
        """Test that boom scenario generates positive expected returns."""
        config = generate_boom_scenario()
        
        for sector in config.sectors:
            boom_return = config.regime_parameters[sector][0]["mu"]
            assert boom_return > 0, f"Boom return for {sector} should be positive"
            
            # Boom volatility should be lower than normal regime
            boom_vol = config.regime_parameters[sector][0]["sigma"]
            normal_vol = config.regime_parameters[sector][1]["sigma"]
            assert boom_vol < normal_vol, f"Boom volatility for {sector} should be lower"


class TestTechShockScenario:
    """Test tech shock scenario generation."""
    
    def test_tech_shock_scenario_default(self):
        """Test tech shock scenario with default parameters."""
        config = generate_tech_shock_scenario()
        
        assert config.name == "Tech_Shock"
        assert config.correlation_regime == "sector_divergence"
        
        # Tech should be in regime 0 (positive shock by default)
        assert config.initial_regimes["tech"] == 0
        
        # Expected patterns should include tech-specific characteristics
        assert config.expected_patterns["tech_divergence"] is True
        assert config.expected_patterns["idiosyncratic_volatility"] is True
        
    def test_tech_shock_requires_tech_sector(self):
        """Test that tech shock requires tech sector."""
        with pytest.raises(ValueError, match="Tech shock scenario requires 'tech' sector"):
            generate_tech_shock_scenario(sectors=["finance", "healthcare"])
            
    def test_tech_shock_magnitude_effects(self):
        """Test tech shock magnitude effects."""
        positive_shock = generate_tech_shock_scenario(shock_magnitude=1.5)
        negative_shock = generate_tech_shock_scenario(shock_magnitude=-1.0)
        
        # Positive shock: tech in boom regime
        assert positive_shock.initial_regimes["tech"] == 0
        
        # Negative shock: tech in bust regime  
        assert negative_shock.initial_regimes["tech"] == 1
        
        # Tech parameters should be extreme
        tech_boom_return = positive_shock.regime_parameters["tech"][0]["mu"]
        tech_bust_return = negative_shock.regime_parameters["tech"][1]["mu"]
        
        assert tech_boom_return > 0.15  # Very high boom return
        assert tech_bust_return < -0.05  # Negative bust return
        
    def test_tech_shock_correlation_structure(self):
        """Test tech shock creates appropriate correlation structure."""
        config = generate_tech_shock_scenario()
        
        tech_idx = config.sectors.index("tech")
        correlation_matrix = config.correlation_matrix
        
        # Tech should have different correlations with other sectors
        tech_correlations = correlation_matrix[tech_idx, :]
        non_tech_correlations = tech_correlations[tech_correlations != 1.0]  # Exclude diagonal
        
        # Should have varying correlations (not all the same)
        assert len(set(np.round(non_tech_correlations, 2))) > 1
        
    def test_tech_shock_persistence(self):
        """Test that tech shock regimes are persistent."""
        config = generate_tech_shock_scenario()
        
        tech_transition = config.transition_matrices["tech"]
        
        # Tech regimes should be very persistent
        assert tech_transition[0, 0] > 0.8  # Boom persists
        assert tech_transition[1, 1] > 0.8  # Bust persists


class TestDecouplingScenario:
    """Test decoupling scenario generation."""
    
    def test_decoupling_scenario_default(self):
        """Test decoupling scenario with default parameters."""
        config = generate_decoupling_scenario()
        
        assert config.name == "Decoupling"
        assert config.correlation_regime == "mixed_correlation"
        
        # Should have alternating initial regimes
        regimes = list(config.initial_regimes.values())
        assert len(set(regimes)) > 1  # Not all in same regime
        
        # Expected patterns
        assert config.expected_patterns["sector_rotation"] is True
        assert config.expected_patterns["dispersion_opportunities"] is True
        assert config.expected_patterns["mixed_correlations"] is True
        
    def test_decoupling_alternating_regimes(self):
        """Test that decoupling creates alternating regimes."""
        config = generate_decoupling_scenario(sectors=["A", "B", "C", "D"])
        
        regimes = [config.initial_regimes[sector] for sector in config.sectors]
        expected = [0, 1, 0, 1]  # Alternating pattern
        assert regimes == expected
        
    def test_decoupling_mixed_correlations(self):
        """Test that decoupling creates mixed correlation structure."""
        config = generate_decoupling_scenario()
        
        # Should have both positive and negative correlations
        off_diagonal = config.correlation_matrix[np.triu_indices_from(config.correlation_matrix, k=1)]
        
        positive_corrs = np.sum(off_diagonal > 0)
        negative_corrs = np.sum(off_diagonal < 0)
        
        # Should have both positive and negative correlations
        assert positive_corrs > 0
        assert negative_corrs > 0
        
    def test_decoupling_correlation_matrix_properties(self):
        """Test that decoupling correlation matrix is valid."""
        config = generate_decoupling_scenario()
        
        matrix = config.correlation_matrix
        
        # Should be symmetric
        assert np.allclose(matrix, matrix.T)
        
        # Diagonal should be 1
        assert np.allclose(np.diag(matrix), 1.0)
        
        # Should be positive semi-definite
        eigenvals = np.linalg.eigvals(matrix)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
        
    def test_decoupling_strength_effects(self):
        """Test decoupling strength parameter effects."""
        weak_decoupling = generate_decoupling_scenario(decoupling_strength=0.5)
        strong_decoupling = generate_decoupling_scenario(decoupling_strength=1.5)
        
        # Stronger decoupling should have more extreme correlations
        weak_corrs = weak_decoupling.correlation_matrix[np.triu_indices_from(weak_decoupling.correlation_matrix, k=1)]
        strong_corrs = strong_decoupling.correlation_matrix[np.triu_indices_from(strong_decoupling.correlation_matrix, k=1)]
        
        weak_abs_mean = np.mean(np.abs(weak_corrs))
        strong_abs_mean = np.mean(np.abs(strong_corrs))
        
        assert strong_abs_mean > weak_abs_mean


class TestScenarioGrid:
    """Test scenario grid generation."""
    
    def test_scenario_grid_basic(self):
        """Test basic scenario grid generation."""
        scenarios = ["crisis", "boom"]
        parameters = {
            "intensity": [0.5, 1.0, 1.5],
            "growth_factor": [0.8, 1.2]
        }
        
        grid = generate_scenario_grid(scenarios, parameters)
        
        # Should generate all combinations
        expected_keys = [
            "crisis_0.5", "crisis_1.0", "crisis_1.5",
            "boom_0.8", "boom_1.2"
        ]
        
        assert len(grid) == 5
        for key in expected_keys:
            assert key in grid
            assert isinstance(grid[key], ScenarioConfig)
            
    def test_scenario_grid_empty_parameters(self):
        """Test scenario grid with empty parameters."""
        scenarios = ["crisis"]
        parameters = {}
        
        grid = generate_scenario_grid(scenarios, parameters)
        
        # Should use default parameter value (1.0)
        assert len(grid) == 1
        assert "crisis_1.0" in grid
        
    def test_scenario_grid_unknown_scenario(self):
        """Test scenario grid with unknown scenario type."""
        scenarios = ["unknown_scenario"]
        parameters = {}
        
        grid = generate_scenario_grid(scenarios, parameters)
        
        # Should skip unknown scenarios
        assert len(grid) == 0
        
    def test_scenario_grid_all_scenario_types(self):
        """Test scenario grid with all scenario types."""
        scenarios = ["crisis", "boom", "tech_shock", "decoupling"]
        parameters = {
            "intensity": [1.0],
            "growth_factor": [1.0],
            "shock_magnitude": [1.0],
            "decoupling_strength": [1.0]
        }
        
        grid = generate_scenario_grid(scenarios, parameters)
        
        assert len(grid) == 4
        assert "crisis_1.0" in grid
        assert "boom_1.0" in grid
        assert "tech_shock_1.0" in grid
        assert "decoupling_1.0" in grid


class TestScenarioSummary:
    """Test scenario summary generation."""
    
    def test_scenario_summary_basic(self):
        """Test basic scenario summary generation."""
        config = generate_boom_scenario()
        summary = get_scenario_summary(config)
        
        # Should contain expected fields
        assert "name" in summary
        assert "description" in summary
        assert "n_sectors" in summary
        assert "avg_expected_return" in summary
        assert "avg_volatility" in summary
        assert "avg_correlation" in summary
        assert "volatility_range" in summary
        assert "return_range" in summary
        assert "correlation_regime" in summary
        assert "expected_patterns" in summary
        
        # Values should be reasonable
        assert summary["n_sectors"] == 3
        assert summary["avg_expected_return"] > 0  # Boom should have positive returns
        assert summary["avg_volatility"] > 0
        assert len(summary["volatility_range"]) == 2
        assert len(summary["return_range"]) == 2
        
    def test_scenario_summary_crisis_vs_boom(self):
        """Test scenario summary differences between crisis and boom."""
        crisis_config = generate_crisis_scenario()
        boom_config = generate_boom_scenario()
        
        crisis_summary = get_scenario_summary(crisis_config)
        boom_summary = get_scenario_summary(boom_config)
        
        # Boom should have higher average returns
        assert boom_summary["avg_expected_return"] > crisis_summary["avg_expected_return"]
        
        # Crisis should have higher correlations
        assert crisis_summary["avg_correlation"] > boom_summary["avg_correlation"]
        
        # Crisis should have higher volatility
        assert crisis_summary["avg_volatility"] > boom_summary["avg_volatility"]
        
    def test_scenario_summary_correlation_regimes(self):
        """Test that summary correctly identifies correlation regimes."""
        crisis = generate_crisis_scenario()
        boom = generate_boom_scenario()
        tech_shock = generate_tech_shock_scenario()
        decoupling = generate_decoupling_scenario()
        
        crisis_summary = get_scenario_summary(crisis)
        boom_summary = get_scenario_summary(boom)
        tech_summary = get_scenario_summary(tech_shock)
        decoupling_summary = get_scenario_summary(decoupling)
        
        assert crisis_summary["correlation_regime"] == "high_correlation"
        assert boom_summary["correlation_regime"] == "low_correlation"
        assert tech_summary["correlation_regime"] == "sector_divergence"
        assert decoupling_summary["correlation_regime"] == "mixed_correlation"


class TestScenarioIntegration:
    """Test integration between scenario templates and regime-switching model."""
    
    def test_scenario_config_compatibility(self):
        """Test that scenario configs are compatible with regime-switching functions."""
        # Import regime-switching functions to test compatibility
        from multi_agent_economics.models.market_for_finance import (
            transition_regimes, generate_regime_returns, build_regime_covariance
        )
        
        config = generate_crisis_scenario()
        
        # Test regime transition compatibility
        new_regimes = transition_regimes(config.initial_regimes, config.transition_matrices)
        assert isinstance(new_regimes, dict)
        assert len(new_regimes) == len(config.sectors)
        
        # Test return generation compatibility
        returns = generate_regime_returns(new_regimes, config.regime_parameters)
        assert isinstance(returns, dict)
        assert len(returns) == len(config.sectors)
        
        # Test covariance building compatibility
        regime_volatilities = {}
        for sector in config.sectors:
            regime_volatilities[sector] = {
                regime: params["sigma"] 
                for regime, params in config.regime_parameters[sector].items()
            }
        
        cov_matrix = build_regime_covariance(new_regimes, regime_volatilities, config.correlation_matrix, config.sectors)
        assert cov_matrix.shape == (len(config.sectors), len(config.sectors))
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric


# Helper function to run quick smoke tests
def test_all_imports():
    """Test that all scenario template functions can be imported and called."""
    # Test basic imports
    scenarios = ["crisis", "boom", "tech_shock", "decoupling"]
    
    for scenario_name in scenarios:
        if scenario_name == "crisis":
            config = generate_crisis_scenario()
        elif scenario_name == "boom":
            config = generate_boom_scenario()
        elif scenario_name == "tech_shock":
            config = generate_tech_shock_scenario()
        elif scenario_name == "decoupling":
            config = generate_decoupling_scenario()
        
        assert isinstance(config, ScenarioConfig)
        assert config.name is not None
        assert len(config.sectors) > 0


if __name__ == "__main__":
    pytest.main([__file__])