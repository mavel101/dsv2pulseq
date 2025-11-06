"""
Read a dsv INF file and append blocks to sequence
"""

import re

FLAG_BITS = {
    'ACQEND': 0,  # last scan
    'RTFEEDBACK': 1,  # Realtime feedback scan
    'HPFEEDBACK': 2,  # High perfomance feedback scan
    'ONLINE': 3,  # processing should be done online
    'OFFLINE': 4,  # processing should be done offline
    'SYNCDATA': 5,  # readout contains synchroneous data
    'noname6': 6,
    'noname7': 7,
    'LASTSCANINCONCAT': 8,  # Flag for last scan in concatination
    'noname9': 9,
    'RAWDATACORRECTION': 10,  # Correct with the rawdata corr. factor
    'LASTSCANINMEAS': 11,  # Flag for last scan in measurement
    'SCANSCALEFACTOR': 12,  # Flag for scan specific additional scale
    '2NDHADAMARPULSE': 13,  # 2nd RF exitation of HADAMAR
    'REFPHASESTABSCAN': 14,  # reference phase stabilization scan
    'PHASESTABSCAN': 15,  # phase stabilization scan
    'D3FFT': 16,  # execute 3D FFT
    'SIGNREV': 17,  # sign reversal
    'PHASEFFT': 18,  # execute phase fft
    'SWAPPED': 19,  # swapped phase/readout direction
    'POSTSHAREDLINE': 20,  # shared line
    'PHASCOR': 21,  # phase correction data
    'PATREFSCAN': 22,  # additional scan for PAT ref line/partition
    'PATREFANDIMASCAN': 23,  # PAT ref that is also used as image scan
    'REFLECT': 24,  # reflect line
    'NOISEADJSCAN': 25,  # noise adjust scan
    'SHARENOW': 26,  # lines may be shared between e.g. phases
    'LASTMEASUREDLINE': 27,  # indicates last meas line of e.g. phases
    'FIRSTSCANINSLICE': 28,  # first scan in slice; req for timestamps
    'LASTSCANINSLICE': 29,  # last scan in slice; req for timestamps
    'TREFFECTIVEBEGIN': 30,  # indicates the TReff begin (triggered)
    'TREFFECTIVEEND': 31,  # indicates the TReff end (triggered)
}

def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def read_dsv_inf(file, seq):
    """ Read INF dsv file (tested only for VE)

    Important: The INF file contains the gradients in logical coordinate system (GP,GR,GS)
    """

    delta_rf = seq.delta_rf
    delta_grad = seq.delta_grad

    counter_pattern = r"sLC\.ush(\w+)\s*:\s*(\d+)"
    flag_pattern = r"aulEvalInfoMask0\s*:\s*(0x[0-9A-Fa-f]+)"
    adc_time_pattern = r"MeasHeader\s+(\d+)"

    counters_flags = []

    with open(file, 'r') as f:
        line_ix = float('inf')
        block_idx = -1
        for k,line in enumerate(f):
            if 'EventBlock' in line:
                block_idx = seq.n_blocks
                char_ix = find_char(line, ' ')
                start_time = int(line[char_ix[-2]:char_ix[-1]])
                duration = int(line[char_ix[-1]:])
                seq.add_block(block_idx, duration, start_time)
            if 'Rel.Time' in line:
                ix = find_char(line, '|') # position of events in line
                line_ix = k+2
                ts_old = 0
                freq_phase = None
            if '|' not in line:
                # end of event block
                line_ix = float('inf')
                ts = -1
            if k>=line_ix:
                block = seq.get_block(block_idx)
                ts = int(line[ix[0]+1:ix[1]].strip())
                block.add_timestamp(ts)

                if ts != ts_old and freq_phase is not None:
                    # NCO (freq/phase) is independent event in Siemens seq, but not in Pulseq
                    # Has to be connected to either RF or ADC event
                    block.set_freqphase(freq_phase, ts_old)
                    freq_phase = None

                if line[ix[1]+1:ix[2]].strip():
                    rf_str = line[ix[1]+1:ix[2]].strip()
                    rf_dur = int(rf_str[rf_str.rfind('/')+1:].strip())
                    rf_shape_ix = slice((block.start_time+ts)//delta_rf, (block.start_time+ts+rf_dur)//delta_rf)
                    block.add_rf(rf_dur, rf_shape_ix, ts)
                if line[ix[2]+1:ix[3]].strip():
                    freqphase_str = line[ix[2]+1:ix[3]].strip()
                    freq = float(freqphase_str[freqphase_str.rfind(':')+1:freqphase_str.rfind('/')].strip())
                    phase = float(freqphase_str[freqphase_str.rfind('/')+1:].strip())
                    freq_phase = [freq, phase]
                if line[ix[4]+1:ix[5]].strip():
                    adc_str = line[ix[4]+1:ix[5]].strip()
                    sample_str = adc_str[adc_str.rfind(':')+1:adc_str.rfind('/')].strip()
                    if 'VOP_READOUT' not in adc_str and 'SMD' not in sample_str: # pTx RX events during RFs are not ADCs
                        adc_samples = float(sample_str)
                        adc_dur = float(adc_str[adc_str.rfind('/')+1:].strip())
                        block.add_adc(adc_dur, adc_samples, ts)
                if line[ix[5]+1:ix[6]].strip():
                    gp_str = line[ix[5]+1:ix[6]].strip()
                    gp_ix = find_char(gp_str, '/')
                    gp_amp = float(gp_str[gp_str.rfind(':')+1:gp_ix[0]].strip()) # amplitude in logical! coordinate system
                    gp_rut = int(gp_str[gp_ix[0]+1:gp_ix[1]].strip())
                    gp_dur = int(gp_str[gp_ix[1]+1:gp_ix[2]].strip())
                    gp_rdt = int(gp_str[gp_ix[2]+1:].strip())
                    gp_shape_ix = slice((block.start_time+ts)//delta_grad, (block.start_time+ts+gp_dur+gp_rdt)//delta_grad)
                    block.add_grad('p', gp_amp, gp_dur, gp_rut, gp_rdt, gp_shape_ix, ts)
                if line[ix[6]+1:ix[7]].strip():
                    gr_str = line[ix[6]+1:ix[7]].strip()
                    gr_ix = find_char(gr_str, '/')
                    gr_amp = float(gr_str[gr_str.rfind(':')+1:gr_ix[0]].strip())
                    gr_rut = int(gr_str[gr_ix[0]+1:gr_ix[1]].strip())
                    gr_dur = int(gr_str[gr_ix[1]+1:gr_ix[2]].strip())
                    gr_rdt = int(gr_str[gr_ix[2]+1:].strip())
                    gr_shape_ix = slice((block.start_time+ts)//delta_grad, (block.start_time+ts+gr_dur+gr_rdt)//delta_grad)
                    block.add_grad('r', gr_amp, gr_dur, gr_rut, gr_rdt, gr_shape_ix, ts)
                if line[ix[7]+1:ix[8]].strip():
                    gs_str = line[ix[7]+1:ix[8]].strip()
                    gs_ix = find_char(gs_str, '/')
                    gs_amp = float(gs_str[gs_str.rfind(':')+1:gs_ix[0]].strip())
                    gs_rut = int(gs_str[gs_ix[0]+1:gs_ix[1]].strip())
                    gs_dur = int(gs_str[gs_ix[1]+1:gs_ix[2]].strip())
                    gs_rdt = int(gs_str[gs_ix[2]+1:].strip())
                    gs_shape_ix = slice((block.start_time+ts)//delta_grad, (block.start_time+ts+gs_dur+gs_rdt)//delta_grad)
                    block.add_grad('s', gs_amp, gs_dur, gs_rut, gs_rdt, gs_shape_ix, ts)
                if line[ix[8]+1:ix[9]].strip():
                    trig_str = line[ix[8]+1:ix[9]].strip()
                    trig_type = trig_str[trig_str.rfind(':')+1:trig_str.rfind('/')].strip()
                    trig_dur = int(trig_str[trig_str.rfind('/')+1:].strip())
                    block.add_trig(trig_dur, trig_type, ts)

                ts_old = ts

            # counters/flags
            match = re.search(adc_time_pattern, line)
            if match:
                adc_time = int(match.group(1))
                counters_flags.append({'adc_time': adc_time, 'counters': {}, 'flags': []})

            match = re.search(counter_pattern, line)
            if match:
                name = match.group(1)
                value = match.group(2)
                counters_flags[-1]['counters'][name] = int(value)
            match = re.search(flag_pattern, line)
            if match:
                flags_hex = match.group(1)
                mask = int(flags_hex, 16)
                counters_flags[-1]['flags'] = [flag for flag, bit in FLAG_BITS.items() if mask & (1 << bit)]
    
    # assign counters and flags to ADCs
    for block in seq.block_list:
        for ts in block.timestamps:
            for event in block.timestamps[ts]:
                if event.type == 'adc':
                    # find matching counters/flags by adc_time
                    adc_time = block.start_time + ts
                    if len(counters_flags) > 0 and counters_flags[0]['adc_time'] == adc_time:
                        event.counters = counters_flags[0]['counters']
                        event.flags = counters_flags[0]['flags']
                        counters_flags.pop(0)
