#!/usr/bin/env python

import argparse
import os
from dsv2pulseq.read_dsv import read_dsv

defaults = {'out_file': 'external.seq',
            'ref_volt': 223.529007,
            'lead_time': 100,
            'hold_time': 30,
            'adc_dead_time': 10}

def main(args):

    # check input
    sfx = ['_INF', '_GRX', '_GRY', '_GRZ', '_RFD', '_RFP']
    for suffix in sfx:
        if not os.path.isfile(args.in_file_prefix + suffix + '.dsv') or os.path.isfile(args.in_file_prefix + suffix + '.DSV'):
            raise OSError(f"DSV file {args.in_file_prefix + suffix + '.dsv'} does not exist.")

    seq = read_dsv(args.in_file_prefix, args.ref_volt, plot=False)
    seq.set_lead_hold_time(args.lead_time, args.hold_time)
    seq.set_adc_dead_time(args.adc_dead_time)
    seq.make_pulseq_sequence(args.out_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file_prefix', type=str, help="Input dsv file prefix. E.g. 'gre'")
    parser.add_argument('-o', '--out_file', type=str, help='Output Pulseq file.')
    parser.add_argument('-r', '--ref_volt', type=float, help='Reference voltage of simulation [V].')
    parser.add_argument('--lead_time', type=int, help='RF lead time [us] (minimum time between start of event block and beginning of RF).')
    parser.add_argument('--hold_time', type=int, help='RF hold time [us] (minimum time from end of RF to end of event block).')
    parser.add_argument('--adc_dead_time', type=int, help='ADC dead time [us] (minimum time between start/end of ADC readout and next ADC or event block start/end).')
    
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)
