"""
Helper functions
"""

import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_delay import make_delay

def waveform_from_seqblock(seq_block):
    """
    extracts gradient waveform from Pypulseq sequence block
    """

    if seq_block.channel == 'x':
        axis = 0
    elif seq_block.channel == 'y':
        axis = 1
    elif seq_block.channel == 'z':
        axis = 2
    else:
        raise ValueError('No valid gradient waveform')
    dummy_seq = Sequence() # helper dummy sequence
    dummy_seq.add_block(make_delay(d=1e-3)) # dummy delay to allow gradients starting at a nonzero value
    dummy_seq.add_block(seq_block)
    return dummy_seq.gradient_waveforms()[axis,100:-1] # last value is a zero that does not belong to the waveform

def round_up_to_raster(number, decimals=0):
    """
    round number up to a specific number of decimal places.
    """
    multiplier = 10 ** decimals
    return np.ceil(number * multiplier) / multiplier