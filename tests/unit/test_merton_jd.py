import numpy as np
import pytest
from quant_engine.models.merton_jd import MertonJumpDiffusion, CarrMadanFFT

def test_merton_characteristic_function():
    mjd = MertonJumpDiffusion(s0=100.0, r=0.05, q=0.0, sigma=0.2, lam=0.1, mu_j=-0.1, delta=0.1)
    cf = mjd.characteristic_function(1.0, 1.0)
    assert isinstance(cf, complex)
    # The magnitude of CF should be <= 1 for a valid distribution characteristic function
    assert np.abs(cf) <= 1.0 + 1e-5

def test_carr_madan_fft_pricing():
    mjd = MertonJumpDiffusion(s0=100.0, r=0.05, q=0.0, sigma=0.2, lam=0.1, mu_j=-0.1, delta=0.1)
    fft_pricer = CarrMadanFFT(mjd, n_power=10, eta=0.25)
    
    strikes, call_prices = fft_pricer.price_european_calls(t=1.0)
    
    # Verify arrays are properly populated
    assert len(strikes) == 2**10
    assert len(call_prices) == 2**10
    
    # Filter to reasonable strikes to avoid deep ITM/OTM numerical instability
    mask = (strikes >= 50) & (strikes <= 150)
    rel_prices = call_prices[mask]
    
    # Prices should be non-negative
    assert np.all(rel_prices >= -1e-2)
    
    # Prices should be monotonically decreasing with respect to strike
    diff = np.diff(rel_prices)
    assert np.all(diff <= 1e-2)
