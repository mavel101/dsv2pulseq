# dsv2pulseq

Create Pulseq sequence files from Siemens dsv simulation files.

Environment can be installed with provided yml file: conda env create -f dsv2pulseq.yml

### Sequence simulation

The sequence should be simulated in transversal orientation with phase-encode direction A->P and no FOV shift (which is the default).
It has to be simulated with RF phase output (sim /RFP+). Mandatory dsv files are "_INF", "_RFD", "_RFP", "_GRX", "_GRY" and "_GRZ".

### Pulseq output

The conversion can be started by running: `dsv_to_pulseq.py -o #OUT_FILE -r #REF_VOLT #IN_FILE_PREFIX`.  

The IN_FILE_PREFIX is the prefix of the dsv files, e.g. "gre" for "gre_XXX.dsv".
The OUT_FILE is the Pulseq output sequence file (default: "external.seq"). The reference voltage is the voltage the sequence was simulated with (default: 223.529007 V)
The Pulseq sequence has the same orientation as the original sequence, when running in "XYZ in TRA" mode.
