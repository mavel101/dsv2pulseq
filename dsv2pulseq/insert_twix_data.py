#!/usr/bin/env python

import argparse
import twixtools
import ismrmrd

defaults = {'out_file': 'merged_rawdata.dat'}

def main(args):

    if not args.in_file_1.endswith('.dat'):
        raise ValueError("Input file 1 must be a Siemens raw data file '.dat'.")
    if not (args.in_file_2.endswith('.dat') or args.in_file_2.endswith('.mrd')):
        raise ValueError("Input file 2 (Pulseq data file) must be either a Siemens raw data file '.dat' or ISMRMRD file '.mrd'.")
    use_mrd = args.in_file_2.endswith('.mrd')

    file1 = twixtools.read_twix(args.in_file_1, keep_syncdata=True, keep_acqend=True)
    n_acq1 = len(file1[-1]['mdb'])
    if use_mrd:
        file2 = ismrmrd.Dataset(args.in_file_2)
        n_acq2 = file2.number_of_acquisitions()
        if n_acq1 - n_acq2 == 1:
            n_acq2 += 1  # ACQEND is usually not converted to MRD
    else:
        file2 = twixtools.read_twix(args.in_file_2, keep_syncdata=True, keep_acqend=True)
        n_acq2 = len(file2[-1]['mdb'])

    if n_acq1 != n_acq2:
        raise ValueError(f"Files have different number of measurement data blocks. File 1: {len(file1[-1]['mdb'])}, File 2: {n_acq2}.")

    print(f"Read and copy {n_acq1} measurement data blocks.")
    for k,mdb in enumerate(file1[-1]['mdb']):
        if not mdb.is_flag_set('ACQEND') and not mdb.is_flag_set('SYNCDATA'):
            mdb = mdb.convert_to_local()
            if use_mrd:
                mdb.data = file2.read_acquisition(k).data.copy()
            else:
                mdb.data = file2[-1]['mdb'][k].data.copy()
            file1[-1]['mdb'][k] = mdb

    if use_mrd:
        file2.close()

    print(f"Writing merged data to {args.out_file}.")
    twixtools.write_twix(file1, args.out_file, version_is_ve=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Pulseq sequence file from dsv file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file_1', type=str, help="Siemens raw data file acquired with original sequence.")
    parser.add_argument('in_file_2', type=str, help="Pulseq raw data file.")
    parser.add_argument('-o', '--out_file', type=str, help='Output Siemens raw data file name. Default: merged_rawdata.dat')

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    main(args)
