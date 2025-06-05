"""
Helper functions
"""

import numpy as np
from pypulseq.Sequence.sequence import Sequence

def waveform_from_seqblock(seq_block, system=None):
    """
    extracts gradient waveform from Pypulseq sequence block
    """

    if system==None:
        grad_raster_time = 10e-6
    else:
        grad_raster_time = system.grad_raster_time

    if seq_block.channel == 'x':
        axis = 0
    elif seq_block.channel == 'y':
        axis = 1
    elif seq_block.channel == 'z':
        axis = 2
    else:
        raise ValueError('No valid gradient waveform')
    dummy_seq = Sequence() # helper dummy sequence
    dummy_seq.add_block(seq_block)
    dur = round(sum(dummy_seq.block_durations.values()), int(1/grad_raster_time))
    tvec = np.arange(grad_raster_time/2, dur, grad_raster_time)
    return dummy_seq.get_gradients()[axis](tvec)

def round_up_to_raster(number, decimals=0):
    """
    round number up to a specific number of decimal places.
    """
    multiplier = 10 ** decimals
    return np.ceil(number * multiplier) / multiplier