import numpy as np
import os

class Block():
    """
    Event block in "Siemens style", allows more than one object per channel
    All units are in Siemens units:
    times: [us]
    gradients: [mT/m]
    RF pulses: [V]
    """

    def __init__(self, idx, duration, start_time):
        self.block_idx = idx
        self.block_duration = duration
        self.start_time = start_time
        self.timestamps = {}

    def add_timestamp(self, ts):
        if not str(ts) in self.timestamps:
            self.timestamps[str(ts)] = []

    def add_rf(self, duration, shp_ix, ts):
        self.add_timestamp(ts)
        rf = Rf(duration, shp_ix, ts)
        self.timestamps[str(ts)].append(rf)
    
    def add_grad(self, channel, amp, duration, ramp_up, ramp_dn, shp_ix, ts):
        self.add_timestamp(ts)
        grad = Grad(channel, amp, duration, ramp_up, ramp_dn, shp_ix, ts)
        self.timestamps[str(ts)].append(grad)

    def add_adc(self, duration, samples, ts):
        self.add_timestamp(ts)
        adc = Adc(duration, samples, ts)
        self.timestamps[str(ts)].append(adc)

    def add_trig(self, duration, trig_type, ts):
        self.add_timestamp(ts)
        trig = Trig(duration, trig_type, ts)
        self.timestamps[str(ts)].append(trig)

    def set_freqphase(self, freq_phase, ts):
        for event in self.timestamps[str(ts)]:
            if event.type == 'rf' or event.type == 'adc':
                event.set_freqphase(freq_phase)

class Rf():

    def __init__(self, duration, shp_ix, delay):
        self.type = 'rf'
        self.duration = duration
        self.delay = delay
        self.freq = 0
        self.phase = 0
        self.shp_ix = shp_ix

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]
    
class Grad():

    def __init__(self, channel, amp, duration, ramp_up, ramp_dn, shp_ix, delay):
        self.type = 'g' + channel 
        self.channel = channel
        self.amp = amp
        self.duration = duration # flat_top + ramp_up
        self.ramp_up = ramp_up
        self.ramp_dn = ramp_dn
        self.delay = delay
        self.shp_ix = shp_ix

class Adc():

    def __init__(self, duration, samples, delay):
        self.type = 'adc'
        self.duration = duration
        self.samples = samples
        self.delay = delay
        self.freq = 0
        self.phase = 0

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]

class Trig():

    def __init__(self, duration, trig_type, delay):
        self.type = 'trig'
        self.duration = duration
        self.delay = delay
        self.trig_type = trig_type

class Sequence():

    def __init__(self):
        self.n_blocks = 0
        self.duration = 0
        self.block_list = []

        self.rf_val = np.array([])
        self.gx_val = np.array([])
        self.gy_val = np.array([])
        self.gz_val = np.array([])

        self.rf_delta = 0 
        self.gx_delta = 0 
        self.gy_delta = 0 
        self.gz_delta = 0 

    def add_block(self, idx, duration, start_time):
        self.n_blocks += 1
        self.duration = start_time + duration
        self.block_list.append(Block(idx, duration, start_time))

    def get_block(self, idx):
        return self.block_list[idx]

    def set_shapes(self, shapes):

        rf_val = shapes[0].values * np.exp(1j*np.deg2rad(shapes[1].values))
        self.rf_val = rf_val
        self.gx_val = shapes[2].values
        self.gy_val = shapes[3].values
        self.gz_val = shapes[4].values
        self.rf_delta = int(shapes[0].definitions.horidelta)
        self.gx_delta = int(shapes[2].definitions.horidelta)
        self.gy_delta = int(shapes[3].definitions.horidelta)
        self.gz_delta = int(shapes[4].definitions.horidelta)

    def get_shape(self, event):
        """
        Gets shape for RF or gradient event
        """
        if event.type == 'rf':
            return self.rf_val[event.shp_ix]
        elif event.type == 'gx':
            return self.gx_val[event.shp_ix]
        elif event.type == 'gy':
            return self.gy_val[event.shp_ix]
        elif event.type == 'gz':
            return self.gz_val[event.shp_ix]
        else:
            return None

    def write_pulseq(self, filename, ref_volt=223.529007):
        """
        Create a Pulseq file from the sequence object.
        filename: Pulseq output filename
        """

        import pypulseq as pp
        from pypulseq.make_arbitrary_grad import make_arbitrary_grad
        import time

        print("Create Pulseq sequence file.")
        start_time = time.time()

        filename = os.path.splitext(filename)[0] + '.seq'

        # Conversion factors from dsv to (Py)Pulseq (SI) units
        cf_time = 1e-6 # [us] -> [s]
        cf_grad = 1e-3 # [mT/m] -> [T/m]
        cf_rf = np.pi * 1e3 / ref_volt # [V] -> [Hz] - ref_volt is voltage for 1ms 180 deg rectangular pulse

        rf_raster = 1e-6
        grad_raster = 1e-5
        delta = {'rf': self.rf_delta*cf_time, 'gx': self.gx_delta*cf_time, 'gy': self.gy_delta*cf_time, 'gz': self.gz_delta*cf_time}

        # set high system values to not produce any errors
        system = pp.Opts(max_grad=1e4, max_slew=1e4, rf_raster_time=rf_raster, grad_raster_time=grad_raster, grad_unit='mT/m', slew_unit='mT/m/ms')

        # track objects and duration
        # time tracking is always done in us, as these are integers, time is only converted to [s], when creating a pulseq event
        pp_block_dur = 0
        event_check = {'rf': False, 'gx': False, 'gy': False, 'gz': False, 'adc': False, 'trig': False}
        ts_offset = 0

        pp_seq = pp.Sequence()
        pp_seq.set_definition('Name', filename)
        for block in self.block_list:
            pp_events = []
            for k,ts in enumerate(block.timestamps):
                events = block.timestamps[ts]

                # if the Pulseq block already contains an event of the same kind, the block has to be splitted
                if any(event_check[event.type] for event in events):
                    pp_events_tmp = []
                    for pp_event in pp_events:
                        pp_dur = int(np.round(pp.calc_duration(pp_event) * 1e6))
                        if pp_dur > int(ts) - ts_offset:
                            if hasattr(pp_event, 'waveform') or hasattr(pp_event, 'amplitude'):
                                split_pt = (pp_dur - (int(ts) - ts_offset)) * cf_time
                                g_new1, g_new2 = pp.split_gradient_at(pp_event, split_pt, system=system)
                                pp_events_tmp.append(g_new2)
                                pp_event = g_new1
                            else:
                                raise ValueError(f"ADC, RF or trigger event in block with index {block.block_idx} can not be splitted. This version only supports splitting of gradients")
                    
                    pp_seq.add_block(*pp_events)
                    pp_events = pp_events_tmp
                    ts_offset = int(ts)

                for event in events:
                    # add event to the block
                    event_del = event.delay - ts_offset
                    if event.type == 'rf':
                        rf_sig = self.get_shape(event)
                        rf_fac = int(np.round(delta[event.type]/rf_raster))
                        rf_sig = np.interp(np.linspace(0,1,rf_fac*len(rf_sig)), np.linspace(0,1,len(rf_sig)), rf_sig)
                        rf = pp.make_arbitrary_rf(signal=rf_sig*cf_rf, flip_angle=2*np.pi, delay=event_del*cf_time, freq_offset=event.freq, phase_offset=np.deg2rad(event.phase), system=system)
                        pp_events.append(rf)
                        event_dur = event_del + event.duration
                    elif event.type == 'gx' or event.type == 'gy' or event.type == 'gz':
                        if event.ramp_dn != 0:
                            # trapezoid
                            g_flat = (event.duration - event.ramp_dn) * cf_time
                            g_area = event.amp * cf_grad * (g_flat + (event.ramp_up/2+event.ramp_dn/2) * cf_time)
                            g = pp.make_trapezoid(channel=event.channel, amplitude=event.amp*cf_grad, area=g_area, flat_time=g_flat, rise_time=event.ramp_up*cf_time, delay=event_del*cf_time, system=system)
                            g.fall_time = event.ramp_dn * cf_time # in make_trapezoid, its not possible to make an asymetric trapezoid, so we recalculate the area after setting the fall time explicitly
                            g.area = g.amplitude * (g.flat_time + g.rise_time/2 + g.fall_time/2)
                            event_dur = event_del + event.duration + event.ramp_dn
                        elif event.duration == 0 and event.ramp_up == 0:
                            # zero duration gradient
                            pass
                        else:
                            # arbitrary
                            g_wf = self.get_shape(event)
                            g_fac = int(np.round(delta[event.type]/grad_raster))
                            g_wf = np.interp(np.linspace(0,1,g_fac*len(g_wf)), np.linspace(0,1,len(g_wf)), g_wf)
                            g = make_arbitrary_grad(channel=event.channel, waveform=g_wf*cf_grad, delay=event_del*cf_time, system=system)
                            event_dur = event.duration + event_del
                        pp_events.append(g)
                    elif event.type == 'adc':
                        adc = pp.make_adc(num_samples=event.samples, duration=event.duration, delay=event_del*cf_time, freq_offset=event.freq, phase_offset=np.deg2rad(event.phase), system=system)
                        event_dur = event_del + event.duration
                        pp_events.append(adc)
                    elif event.type == 'trig':
                        if event.trig_type == 'EXTRIG0': # only external triggers supported atm
                            trig = pp.make_digital_output_pulse(channel='ext1', duration=event.duration, delay=event_del*cf_time, system=system)
                            pp_events.append(trig)
                            event_dur = event_del + event.duration

                    event_check[event.type] = True

                    # calculate new block duration
                    if event_dur > pp_block_dur:
                        pp_block_dur = event_dur

                    # add possible delay at end of Siemens block
                    if k == len(block.timestamps)-1:
                        ts_rel = int(ts) - ts_offset
                        if ts_rel > pp_block_dur:
                            block_delay = (ts_rel - pp_block_dur) * cf_time
                            delay = pp.make_delay(d=block_delay, system=system)
                            pp_events.append(delay)
                    
            pp_seq.add_block(*pp_events)

        pp_seq.write(filename)
        end_time = time.time()
        print(f"Finished creating Pulseq file in {(end_time-start_time):.2f}s.")
