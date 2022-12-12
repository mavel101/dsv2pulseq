import argparse
import os
from read_dsv import read_dsv

# WIP: add init, add script for command line, add setup.py, add simple test test

defaults = {'out_file': 'external.seq'}

def main(args):

    # check input
    sfx = ['_ADC', '_GRX', '_GRY', '_GRZ', '_RFD', '_RFP']
    for suffix in sfx:
        if os.path.isfile(args.in_file + suffix + '.dsv'):
            raise OSError(f"DSV file {args.in_file + suffix + '.dsv'} does not exist.")

    seq = read_dsv("dsv_test/gre", plot=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file', type=str, help='Input MRD file')
    parser.add_argument('-o', '--out_file', type=str, help='Output Nifti file. If not specified, external.seq')

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)