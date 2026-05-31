import numpy as np
import pytest
from quant_engine.math.interpolation import LogCubicSpline, MonotoneConvexSpline

def test_log_cubic_spline():
    times = np.array([1.0, 2.0, 5.0, 10.0])
    dfs = np.array([0.95, 0.90, 0.80, 0.65])
    
    target_times = np.array([1.5, 3.0, 7.5])
    spline = LogCubicSpline()
    interpolated_dfs = spline.interpolate(times, dfs, target_times)
    
    assert len(interpolated_dfs) == 3
    assert np.all(interpolated_dfs > 0)
    # Check monotonicity of discount factors
    assert interpolated_dfs[0] < dfs[0] and interpolated_dfs[0] > dfs[1]

def test_monotone_convex_spline():
    times = np.array([1.0, 2.0, 5.0, 10.0])
    dfs = np.array([0.95, 0.90, 0.80, 0.65])
    
    target_times = np.array([1.5, 3.0, 7.5])
    mcs = MonotoneConvexSpline()
    interpolated_dfs = mcs.interpolate(times, dfs, target_times)
    
    assert len(interpolated_dfs) == 3
    assert np.all(interpolated_dfs > 0)
    # Monotonicity check
    assert interpolated_dfs[0] < dfs[0] and interpolated_dfs[0] > dfs[1]
