import os
import numpy as np
import matplotlib.pyplot as plt
from quant_engine.models.sabr import SABRCalibrator, hagan_lognormal_vol
from quant_engine.models.merton_jd import MertonJumpDiffusion, CarrMadanFFT

def main():
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    print("Generating SABR Calibration Results...")

    # SABR Calibration
    f = 100.0
    t = 1.0
    strikes = np.linspace(80, 120, 5)
    # Smile shape data
    market_vols = np.array([0.28, 0.23, 0.20, 0.19, 0.21])
    
    calibrator = SABRCalibrator(f, t, strikes, market_vols, beta=1.0)
    res = calibrator.calibrate()
    
    alpha, rho, nu = res['alpha'], res['rho'], res['nu']
    print(f"Calibrated SABR Params: alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
    
    # Plot SABR fit
    fine_strikes = np.linspace(70, 130, 100)
    fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, 1.0, rho, nu) for k in fine_strikes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fine_strikes, fitted_vols, label='Fitted SABR Smile', color='blue', linewidth=2)
    plt.scatter(strikes, market_vols, color='red', label='Market Quotes', zorder=5)
    plt.title('SABR Volatility Smile Calibration')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/sabr_calibration.png')
    plt.close()
    
    print("SABR chart saved to results/sabr_calibration.png")
    
    # Merton JD FFT Pricing
    print("Generating Merton Jump-Diffusion FFT Results...")
    mjd = MertonJumpDiffusion(s0=100.0, r=0.05, q=0.0, sigma=0.15, lam=0.5, mu_j=-0.1, delta=0.15)
    fft_pricer = CarrMadanFFT(mjd, n_power=12, eta=0.25)
    
    fft_strikes, fft_prices = fft_pricer.price_european_calls(t=1.0)
    
    # Filter strikes to a reasonable range
    mask = (fft_strikes >= 50) & (fft_strikes <= 150)
    plot_strikes = fft_strikes[mask]
    plot_prices = fft_prices[mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_strikes, plot_prices, label='FFT European Call Prices', color='green', linewidth=2)
    plt.title('Merton Jump-Diffusion Call Options via Carr-Madan FFT')
    plt.xlabel('Strike')
    plt.ylabel('Call Price')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/fft_merton_jd_prices.png')
    plt.close()
    
    print("Merton JD chart saved to results/fft_merton_jd_prices.png")

if __name__ == "__main__":
    main()
