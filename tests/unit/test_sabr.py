import numpy as np
import pytest
from quant_engine.models.sabr import hagan_lognormal_vol, sabr_objective_function, SABRCalibrator

def test_hagan_lognormal_vol_atm():
    f = 100.0
    k = 100.0
    t = 1.0
    alpha = 0.2
    beta = 1.0 # Equities
    rho = -0.5
    nu = 0.4
    
    vol = hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu)
    assert 0.15 < vol < 0.25 # Should be close to alpha for ATM

def test_hagan_lognormal_vol_otm():
    f = 100.0
    k = 110.0
    t = 1.0
    alpha = 0.2
    beta = 1.0
    rho = -0.5
    nu = 0.4
    
    vol = hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu)
    assert vol > 0.0

def test_sabr_calibrator():
    f = 100.0
    t = 1.0
    strikes = np.array([80, 90, 100, 110, 120])
    # Mock some realistic market smile vols
    market_vols = np.array([0.28, 0.24, 0.20, 0.18, 0.19])
    
    calibrator = SABRCalibrator(f, t, strikes, market_vols, beta=1.0)
    result = calibrator.calibrate()
    
    assert 'alpha' in result
    assert 'rho' in result
    assert 'nu' in result
    assert result['mse'] < 1e-2  # Error should be reasonably small
