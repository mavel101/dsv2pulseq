#!/usr/bin/env python

import argparse
import twixtools

defaults = {'out_file': 'merged_rawdata.dat'}

def main(args):

    file1 = twixtools.read_twix(args.in_file_1, keep_syncdata_and_acqend=True)
    file2 = twixtools.read_twix(args.in_file_2, keep_syncdata_and_acqend=True)

    if len(file1[-1]['mdb']) != len(file2[-1]['mdb']):
        raise ValueError("Files have different number of measurement data blocks (mdb).")

    for k,mdb in enumerate(file1[-1]['mdb']):
        if not mdb.is_flag_set('ACQEND') and not mdb.is_flag_set('SYNCDATA'):
            mdb = mdb.convert_to_local()
            mdb.data = file2[-1]['mdb'][k].data.copy()
            file1[-1]['mdb'][k] = mdb

    twixtools.write_twix(file1, args.out_file, version_is_ve=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file_1', type=str, help="Input Siemens raw data file acquired with original sequence.")
    parser.add_argument('in_file_2', type=str, help="Input Siemens raw data file acquired with original sequence.")
    parser.add_argument('-o', '--out_file', type=str, help='Output Siemens raw data file name. Default: merged_rawdata.dat')

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)
