"""
Read a dsv INF file and append blocks to sequence
"""

def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def read_dsv_inf(file, seq):
    """ Read INF dsv file (tested only for VE)

    Important: The INF file contains the gradients in logical coordinate system (GP,GR,GS)
    """

    delta_rf = seq.delta_rf
    delta_grad = seq.delta_grad

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
                    if "VOP_READOUT" not in adc_str and "SMD" not in sample_str: # pTx RX events during RFs are not ADCs
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
