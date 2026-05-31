import numpy as np
import pytest
from quant_engine.models.sabr import SABRCalibrator
from quant_engine.models.merton_jd import MertonJumpDiffusion, CarrMadanFFT

def test_end_to_end_pricing():
    # 1. Market Data
    f = 100.0
    t = 1.0
    strikes = np.array([90, 100, 110])
    market_vols = np.array([0.22, 0.20, 0.19])
    
    # 2. SABR Calibration
    calibrator = SABRCalibrator(f, t, strikes, market_vols, beta=1.0)
    sabr_params = calibrator.calibrate()
    
    assert sabr_params['mse'] < 1e-2
    
    # 3. FFT Pricing with calibrated roughly equivalent volatility
    sigma_base = sabr_params['alpha']
    mjd = MertonJumpDiffusion(s0=100.0, r=0.05, q=0.0, sigma=sigma_base, lam=0.5, mu_j=-0.05, delta=0.1)
    
    fft_pricer = CarrMadanFFT(mjd, n_power=10, eta=0.25)
    fft_strikes, fft_prices = fft_pricer.price_european_calls(t=1.0)
    
    # Extract ATM price
    idx_atm = np.abs(fft_strikes - 100.0).argmin()
    atm_price = fft_prices[idx_atm]
    
    assert atm_price > 0.0
