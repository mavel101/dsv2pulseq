#!/usr/bin/env python

import argparse
import os
from dsv2pulseq.read_dsv import read_dsv

# WIP: add simple unit test

defaults = {'out_file': 'external.seq'}

def main(args):

    # check input
    sfx = ['_ADC', '_GRX', '_GRY', '_GRZ', '_RFD', '_RFP']
    for suffix in sfx:
        if not os.path.isfile(args.in_file + suffix + '.dsv') or os.path.isfile(args.in_file + suffix + '.DSV'):
            raise OSError(f"DSV file {args.in_file + suffix + '.dsv'} does not exist.")

    print("Okay.")
    # seq = read_dsv(args.in_file, plot=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file', type=str, help='Input dsv file prefix.')
    parser.add_argument('-o', '--out_file', type=str, help='Output Pulseq file. If not specified, external.seq')

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)
