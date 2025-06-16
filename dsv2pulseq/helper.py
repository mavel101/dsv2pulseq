"""
Helper functions
"""

import numpy as np
import math

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

    waveform = np.concatenate((np.zeros(int(grad.delay / system.grad_raster_time)), waveform))

    return waveform

def round_up_to_raster(number, decimals=5, tol=1e-10):
    """
    Round number up to a specific number of decimal places.
    Rounds up only if the digit beyond the desired precision exceeds a tolerance.
    This avoids rounding up for tiny floating point errors.
    
    Parameters:
    - number: float, the value to round
    - decimals: int, number of decimal places
    - tol: float, minimum excess to consider as real (not floating point noise)
    """
    multiplier = 10 ** decimals
    scaled = number * multiplier
    rounded = math.floor(scaled)

    if scaled - rounded > tol:
        rounded += 1

    return rounded / multiplier