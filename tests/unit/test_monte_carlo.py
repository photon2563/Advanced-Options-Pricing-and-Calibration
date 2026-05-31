import numpy as np
import pytest
from quant_engine.math.monte_carlo import simulate_mjd_paths_euler, black_scholes_call, MonteCarloValidator

def test_black_scholes_call():
    price = black_scholes_call(s=100.0, k=100.0, t=1.0, r=0.05, q=0.0, sigma=0.2)
    assert np.isclose(price, 10.45058, atol=1e-4)

def test_simulate_mjd_paths_euler():
    paths = simulate_mjd_paths_euler(
        s0=100.0, r=0.05, q=0.0, sigma=0.2, lam=0.1, mu_j=-0.1, delta=0.1,
        t=1.0, n_steps=252, n_paths=1000, use_antithetic=True
    )
    assert len(paths) == 1000
    assert np.all(paths >= 0.0)

def test_monte_carlo_validator():
    validator = MonteCarloValidator(s0=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0, n_paths=10000, n_steps=50)
    mc_price, std_err = validator.validate_mjd_call(k=100.0, lam=0.1, mu_j=-0.1, delta=0.1)
    
    assert mc_price > 0.0
    assert std_err > 0.0
    # Price should be roughly around BS price + some jump premium
    assert mc_price > 10.0 and mc_price < 12.0
