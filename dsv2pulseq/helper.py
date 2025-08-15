"""
Helper functions
"""

import numpy as np
import math
from scipy.interpolate import interp1d

def trapezoid(amplitude, rise_time, flat_time, fall_time, dt=1e-5):
    # Time segments
    t_rise = np.arange(dt/2, rise_time, dt)
    t_flat = np.arange(dt/2, flat_time, dt)
    t_fall = np.arange(dt/2, fall_time, dt)

    # Signal segments
    rise = (amplitude / rise_time) * t_rise
    plateau = np.ones_like(t_flat) * amplitude
    fall = amplitude - (amplitude / fall_time) * t_fall

    # Concatenate full waveform
    waveform = np.concatenate((rise, plateau, fall))

    return waveform

def waveform_from_seqblock(grad, system=None):
    """
    extracts gradient waveform from Pypulseq sequence block
    """

    if grad.type == 'trap':
        waveform = trapezoid(grad.amplitude, grad.rise_time, grad.flat_time, grad.fall_time, dt=system.grad_raster_time)
    else:
        waveform = grad.waveform

    waveform = np.concatenate((np.zeros(round(grad.delay / system.grad_raster_time)), waveform))

    return waveform

def round_up_to_raster(number, raster=1e-5, tol=1e-10):
    """
    Round a number up to the nearest multiple of `raster`.
    Rounds up only if the excess beyond the nearest raster exceeds `tol`.
    
    Parameters:
    - number: float, value to round
    - raster: float, step size to round up to (e.g., 1e-6)
    - tol: float, minimum excess to consider as real (not floating point noise)
    
    Returns:
    - float: rounded value
    """
    scaled = number / raster
    floored = math.floor(scaled)
    
    if scaled - floored > tol:
        floored += 1
        
    return floored * raster

def round_to_raster(value, raster):
    """
    Round `value` to the nearest multiple of `raster`.

    Parameters:
        value (float): The number to round.
        raster (float): The raster step to round to, e.g., 1e-6.

    Returns:
        float: Rounded value.
    """
    return round(value / raster) * raster


def resample_waveform(waveform, old_raster, new_raster, method='linear'):
    """
    Resample a 1D waveform from one raster time step to another.

    Parameters
    ----------
    waveform : array_like
        1D array of waveform samples at old_raster spacing.
    old_raster : float
        Original raster (time step) in seconds.
    new_raster : float
        Desired raster (time step) in seconds.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic', etc. — see scipy.interpolate.interp1d).

    Returns
    -------
    new_waveform : ndarray
        Resampled waveform at the new raster spacing.
    new_time : ndarray
        Time vector for the new waveform.
    """
    waveform = np.asarray(waveform)
    n_old = len(waveform)
    old_time = np.arange(n_old) * old_raster

    # Create interpolation function
    interp_func = interp1d(old_time, waveform, kind=method, fill_value="extrapolate")

    # Create new time points
    n_new = int(np.round(old_time[-1] / new_raster)) + 1
    new_time = np.arange(n_new) * new_raster

    # Interpolate
    new_waveform = interp_func(new_time)

    return new_waveform
