# dsv2pulseq

Create Pulseq sequence files from Siemens dsv simulation files. The converter was only tested with VE/VX dsv file layouts.
VB/VD files are not supported, as the simulator can not output RF phase data.  
RF values might be differing slightly compared to the original sequence, as the dsv files contain RF values only on a 5us raster.

**Note that using the IDEA framework requires a research agreement with Siemens. Furthermore, additional agreements apply if product sequence source code is used to create the dsv files. Therefore, the use of Pulseq files generated with dsv2pulseq must comply with these contracts. No Pulseq sequence files may be published that expose vendor product sequences or building blocks involved in these sequences without an explicit permission of the vendor.**

## Installation

The package can be install with `pip install dsv2pulseq`.

Alternatively, a Python environment with the package installed can be created with the provided yml file: `conda env create -f dsv2pulseq.yml`.

Unittests can be run with `python -m unittest discover test/`. This includes testing, whether the conversion is successful for each dsv dataset in the folder "test/test_data".

This package only depends on numpy and PyPulseq [1].

Merging data to the original Siemens raw data file for retrospective reconstruction (see "Reconstruction of Pulseq data") requires the twixtools package (https://github.com/pehses/twixtools).

## Sequence simulation

Simulate the sequence with the following settings:  
- Transversal orientation with phase-encode direction A->P and no FOV shift (which is the default)
- Simulate with RF phase output (sim /RFP+). 
- Mandatory dsv files are "_INF", "_RFD", "_RFP", "_GRX", "_GRY" and "_GRZ".

Example data "MiniFLASH.dsv" can be found in the test/test_data folder. These simulation files are from the Siemens MiniFLASH demo sequence.

## Create Pulseq output

The conversion can be started by running:  
```
dsv_to_pulseq -o OUT_FILE -r REF_VOLT IN_FILE_PREFIX
```
from the shell.

The IN_FILE_PREFIX is the prefix of the dsv files, e.g. "gre" for "gre_XXX.dsv". The OUT_FILE is the Pulseq output sequence file (default: "external.seq"). The reference voltage is the voltage the sequence was simulated with (Siemens default: 223.529007 V).

The conversion can also be done in Python by running:
```
from dsv2pulseq import read_dsv
seq = read_dsv('/path/to/dsv/dsv_prefix')
seq_pulseq = seq.make_pulseq_sequence('external.seq')
```

There is an experimental function to check the shapes of RF waveforms and gradients that plots the difference between the original and converted waveforms:
```
from dsv2pulseq import check_dsv
check_dsv('/path_to_dsv/dsv_prefix_original', 'path_to_dsv/dsv_prefix_pulseq')
```

Note that:  
- Only single transmit channel RF pulses are currently supported.
- The Pulseq sequence has the same orientation as the original sequence in the physical/scanner coordinate system, when running the Pulseq interpreter in "XYZ in TRA" mode.
- The Pulseq interpreter version 1.5.0 should be used, as the versions before contain a bug in setting the RF raster time correctly. 
- RF and gradient waveforms might be slightly different compared to the original sequence due to limited numerical accuracy of the DSV files.
- RF pulse values are on a 5us raster in the dsv files, which might be insufficient for pulses with rapidly varying phase

## Reconstruction of Pulseq data

Data acquired with the converted Pulseq sequence cannot be easily reconstructed, as the scan headers containing the meta information are missing.
The data from the Pulseq sequence (IN_FILE_2) can be inserted into a raw data file acquired with the original sequence (IN_FILE_1) using twixtools (see dependencies). If twixtools is installed, the following command will insert the data:
```
insert_twix_data IN_FILE_1 IN_FILE_2 -o OUT_FILE
```
The merged output file (OUT_FILE) can be used for rectrospective reconstruction at the scanner. Alternatively the Pulseq data file can also be in the MRD format (requires ISMRMRD python package)

## References

[1] Ravi, Keerthi, Sairam Geethanath, and John Vaughan. "PyPulseq: A Python Package for MRI Pulse Sequence Design." Journal of Open Source Software 4.42 (2019): 1725., https://github.com/imr-framework/pypulseq
