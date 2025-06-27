#!/usr/bin/env python

import argparse
import os
import sys
from dsv2pulseq.read_dsv import read_dsv

defaults = {
    'out_file': 'external.seq',
    'ref_volt': 223.529007,
    'lead_time': 100,
    'hold_time': 30,
    'adc_dead_time': 10,
    'fov': [None, None, None],
}

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Create Pulseq sequence file from dsv file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('in_file_prefix', type=str, help="Input dsv file prefix. E.g. 'gre'")
    parser.add_argument('-o', '--out_file', type=str, help='Output Pulseq file.')
    parser.add_argument('-r', '--ref_volt', type=float, help='Reference voltage of simulation [V].')
    parser.add_argument('--lead_time', type=int, help='RF lead time [us].')
    parser.add_argument('--hold_time', type=int, help='RF hold time [us].')
    parser.add_argument('--adc_dead_time', type=int, help='ADC dead time [us].')
    parser.add_argument('--fov', type=float, nargs=3, help='Field of view [mm] for x, y, z.')
    parser.add_argument('--highgain', action='store_true', help='Set receiver gain to high.')

    parser.set_defaults(**defaults)
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_arguments(argv)

    # check input
    sfx = ['_INF', '_GRX', '_GRY', '_GRZ', '_RFD', '_RFP']
    for suffix in sfx:
        fname = args.in_file_prefix + suffix + '.dsv'
        if not os.path.isfile(fname):
            raise OSError(f"DSV file {fname} does not exist.")

    seq = read_dsv(args.in_file_prefix, args.ref_volt, plot=False)
    seq.set_lead_hold_time(args.lead_time, args.hold_time)
    seq.set_adc_dead_time(args.adc_dead_time)
    seq.make_pulseq_sequence(args.out_file, fov=args.fov, highgain=args.highgain)

if __name__ == "__main__":
    sys.exit(main())
