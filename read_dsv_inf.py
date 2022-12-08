"""
Read a dsv INF file and append blocks to sequence
"""

from sequence import Block, Rf, Grad, Adc, Trig

def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def read_dsv_inf(file, seq):
    """ Read INF dsv file
    """

    with open(file, 'r') as f:
        line_ix = float('inf')
        events = []
        for k,line in enumerate(f):
            if "EventBlock" in line:
                seq.n_blocks += 1
                duration = line[line.rfind(" "):]
                seq.add_block(duration)
            if "Rel.Time" in line:
                ix = find_char(line, '|') # position of events in line
                line_ix = k+2
            if "|" not in line:
                # end of event block
                # WIP: add all objects to a block
                line_ix = float('inf')
            if k>=line_ix:
                time = line[ix[0]:ix[1]].strip()
                if line[ix[1]:ix[2]].strip():
                    rf = line[ix[1]:ix[2]].strip()
                if line[ix[2]:ix[3]].strip():
                    freqphase = line[ix[2]:ix[3]].strip()
                if line[ix[4]:ix[5]].strip():
                    adc = line[ix[4]:ix[5]].strip()
                if line[ix[5]:ix[6]].strip():
                    gp = line[ix[5]:ix[6]].strip()
                if line[ix[6]:ix[7]].strip():
                    gr = line[ix[6]:ix[7]].strip()
                if line[ix[7]:ix[8]].strip():
                    gs = line[ix[7]:ix[8]].strip()
                if line[ix[8]:ix[9]].strip():
                    trig = line[ix[8]:ix[9]].strip()

                # check if event block already contains event of same kind, then open new block