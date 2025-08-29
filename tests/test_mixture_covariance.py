"""
Test for mixture covariance calculation in portfolio optimization.

This test verifies that the compute_portfolio_moments function correctly
implements the full mixture distribution formula instead of the incorrect
E[σᵢ] * E[σⱼ] * ρᵢⱼ approximation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.models.market_for_finance import compute_portfolio_moments


def test_mixture_covariance_calculation():
    """
    Test that mixture covariance correctly implements:
    Var[R] = Σₛ P(s) * [Σˢ + (μˢ - E[R])(μˢ - E[R])ᵀ]
    """
    # Setup test data: 2 sectors, 2 regimes each
    agent_beliefs = {
        'tech': np.array([0.7, 0.3]),      # 70% bull, 30% bear
        'finance': np.array([0.6, 0.4])    # 60% bull, 40% bear
    }
    
    regime_returns = {
        'tech': {0: 0.08, 1: 0.02},        # Bull: 8%, Bear: 2%
        'finance': {0: 0.06, 1: 0.01}      # Bull: 6%, Bear: 1%
    }
    
    regime_volatilities = {
        'tech': {0: 0.15, 1: 0.25},        # Bull: 15%, Bear: 25%
        'finance': {0: 0.12, 1: 0.20}      # Bull: 12%, Bear: 20%
    }
    
    correlations = np.array([[1.0, 0.3], [0.3, 1.0]])  # 30% correlation
    
    # Compute results
    sector_order = ["tech", "finance"]  # Canonical ordering
    expected_returns, covariance_matrix = compute_portfolio_moments(
        agent_beliefs, regime_returns, regime_volatilities, correlations, sector_order
    )
    
    # Expected returns should be: E[R] = Σₖ P(k) * μᵏ
    expected_tech_return = 0.7 * 0.08 + 0.3 * 0.02  # = 0.062
    expected_finance_return = 0.6 * 0.06 + 0.4 * 0.01  # = 0.04
    
    np.testing.assert_almost_equal(expected_returns[0], expected_tech_return, decimal=6)
    np.testing.assert_almost_equal(expected_returns[1], expected_finance_return, decimal=6)
    
    # Manual calculation of mixture covariance for verification
    # 4 regime combinations: (0,0), (0,1), (1,0), (1,1)
    
    # Regime combination probabilities
    prob_00 = 0.7 * 0.6  # = 0.42
    prob_01 = 0.7 * 0.4  # = 0.28  
    prob_10 = 0.3 * 0.6  # = 0.18
    prob_11 = 0.3 * 0.4  # = 0.12
    
    # Regime-specific means and covariances
    means_00 = np.array([0.08, 0.06])
    means_01 = np.array([0.08, 0.01])
    means_10 = np.array([0.02, 0.06])
    means_11 = np.array([0.02, 0.01])
    
    vols_00 = np.array([0.15, 0.12])
    vols_01 = np.array([0.15, 0.20])
    vols_10 = np.array([0.25, 0.12])
    vols_11 = np.array([0.25, 0.20])
    
    # Regime-specific covariance matrices
    cov_00 = np.outer(vols_00, vols_00) * correlations
    cov_01 = np.outer(vols_01, vols_01) * correlations
    cov_10 = np.outer(vols_10, vols_10) * correlations
    cov_11 = np.outer(vols_11, vols_11) * correlations
    
    # Mean differences from overall expected return
    E_R = np.array([expected_tech_return, expected_finance_return])
    diff_00 = means_00 - E_R
    diff_01 = means_01 - E_R
    diff_10 = means_10 - E_R
    diff_11 = means_11 - E_R
    
    # Full mixture covariance
    expected_cov = (
        prob_00 * (cov_00 + np.outer(diff_00, diff_00)) +
        prob_01 * (cov_01 + np.outer(diff_01, diff_01)) +
        prob_10 * (cov_10 + np.outer(diff_10, diff_10)) +
        prob_11 * (cov_11 + np.outer(diff_11, diff_11))
    )
    
    # Verify computed covariance matches manual calculation
    np.testing.assert_array_almost_equal(covariance_matrix, expected_cov, decimal=10)
    
    # Verify matrix properties
    assert covariance_matrix.shape == (2, 2)
    assert np.allclose(covariance_matrix, covariance_matrix.T)  # Symmetric
    assert np.all(np.linalg.eigvals(covariance_matrix) > 0)    # Positive definite


def test_mixture_vs_naive_approximation():
    """
    Test that mixture covariance differs from naive E[σᵢ] * E[σⱼ] * ρᵢⱼ approximation.
    """
    agent_beliefs = {
        'tech': np.array([0.5, 0.5]),      # Equal probability
        'finance': np.array([0.8, 0.2])    # Unequal probability
    }
    
    regime_returns = {
        'tech': {0: 0.10, 1: -0.05},       # High contrast
        'finance': {0: 0.05, 1: -0.02}
    }
    
    regime_volatilities = {
        'tech': {0: 0.10, 1: 0.30},        # High contrast
        'finance': {0: 0.08, 1: 0.25}
    }
    
    correlations = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    # Compute mixture covariance
    sector_order = ["tech", "finance"]  # Canonical ordering
    _, mixture_cov = compute_portfolio_moments(
        agent_beliefs, regime_returns, regime_volatilities, correlations, sector_order
    )
    
    # Compute naive approximation: E[σᵢ] * E[σⱼ] * ρᵢⱼ
    expected_vol_tech = 0.5 * 0.10 + 0.5 * 0.30    # = 0.20
    expected_vol_finance = 0.8 * 0.08 + 0.2 * 0.25  # = 0.114
    
    naive_cov = np.outer([expected_vol_tech, expected_vol_finance], 
                        [expected_vol_tech, expected_vol_finance]) * correlations
    
    # Verify they are different (mixture should be larger due to regime uncertainty)
    assert not np.allclose(mixture_cov, naive_cov, atol=1e-6)
    
    # Mixture covariance should be larger (more conservative)
    assert mixture_cov[0, 0] > naive_cov[0, 0]  # Tech variance
    assert mixture_cov[1, 1] > naive_cov[1, 1]  # Finance variance


def test_single_regime_equivalence():
    """
    Test that with single regime per sector, mixture reduces to standard covariance.
    """
    agent_beliefs = {
        'tech': np.array([1.0]),        # Certainty in regime 0
        'finance': np.array([1.0])
    }
    
    regime_returns = {
        'tech': {0: 0.06},
        'finance': {0: 0.04}
    }
    
    regime_volatilities = {
        'tech': {0: 0.15},
        'finance': {0: 0.12}
    }
    
    correlations = np.array([[1.0, 0.4], [0.4, 1.0]])
    
    sector_order = ["tech", "finance"]  # Canonical ordering
    expected_returns, covariance_matrix = compute_portfolio_moments(
        agent_beliefs, regime_returns, regime_volatilities, correlations, sector_order
    )
    
    # With single regime, should equal standard covariance calculation
    expected_cov = np.outer([0.15, 0.12], [0.15, 0.12]) * correlations
    
    np.testing.assert_array_almost_equal(covariance_matrix, expected_cov, decimal=10)
    np.testing.assert_array_almost_equal(expected_returns, [0.06, 0.04], decimal=10)


if __name__ == "__main__":
    test_mixture_covariance_calculation()
    test_mixture_vs_naive_approximation()
    test_single_regime_equivalence()
    print("✓ All mixture covariance tests passed")