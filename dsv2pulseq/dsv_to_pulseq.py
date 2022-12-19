#!/usr/bin/env python

import argparse
import os
from dsv2pulseq.read_dsv import read_dsv

# WIPs: - VB/VD version of read_dsv_inf
#       - check why script is not installed by yml
#       - update README

defaults = {'out_file': 'external.seq',
            'ref_volt': 223.529007}

def main(args):

    # check input
    sfx = ['_INF', '_GRX', '_GRY', '_GRZ', '_RFD', '_RFP']
    for suffix in sfx:
        if not os.path.isfile(args.in_file_prefix + suffix + '.dsv') or os.path.isfile(args.in_file_prefix + suffix + '.DSV'):
            raise OSError(f"DSV file {args.in_file_prefix + suffix + '.dsv'} does not exist.")

    seq = read_dsv(args.in_file_prefix, args.ref_volt, plot=False)
    seq.write_pulseq(args.out_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file_prefix', type=str, help="Input dsv file prefix. E.g. 'gre'")
    parser.add_argument('-o', '--out_file', type=str, help='Output Pulseq file. Default: external.seq')
    parser.add_argument('-r', '--ref_volt', help='Reference voltage of simulation. Default: 223.529007 V')

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)
