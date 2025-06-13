import numpy as np
import os
import time
import pypulseq as pp
from dsv2pulseq.helper import round_up_to_raster, waveform_from_seqblock

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

    def __repr__(self):
        return f"Block {self.block_idx} with duration {self.block_duration} us at start time {self.start_time} us with timestamps {self.timestamps}."
    
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

    def __init__(self, ref_volt):
        self.n_blocks = 0
        self.duration = 0
        self.block_list = []

        self.rf_val = np.array([])
        self.gx_val = np.array([])
        self.gy_val = np.array([])
        self.gz_val = np.array([])

        # raster times
        self.delta = {'rf': 5, 'grad': 10} # raster times in dsv files [us]

        # Conversion factors from dsv to (Py)Pulseq (SI) units
        self.gamma = 42.576e6
        self.cf_time = 1e-6 # [us] -> [s]
        self.cf_grad = -1* 1e-3*self.gamma # [mT/m] -> [Hz/m], the "-1" is to be compatible to the "XYZ in TRA" mode of Pulseq

        # ref_volt is voltage for 1ms 180 deg rectangular pulse
        # -> pulse [Hz] = pulse[V]/ref_volt[V] * 500[Hz]
        self.cf_rf = 5e2 / ref_volt # [V] -> [Hz]

        # trigger types
        self.trig_types = {'EXTRIG0': 'ext1', 'OSC0': 'osc0', 'OSC1': 'osc1'}

        # coil lead & hold times
        self.rf_lead_time = 100 # [us]
        self.rf_hold_time = 30 # [us]

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
        self.delta['rf'] = int(shapes[0].definitions.horidelta)
        self.delta['grad']= int(shapes[2].definitions.horidelta)

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
        
    def set_lead_hold(self, rf_lead, rf_hold):
        """
        Set RF lead and hold times
        """
        self.rf_lead_time = rf_lead
        self.rf_hold_time = rf_hold

    def write_pulseq(self, filename):
        """
        Create a Pulseq file from the sequence object.
        Time tracking is always done in [us] (integers), 
        time is only converted to [s], when creating a pulseq event
        
        Inputs:
        --------
        filename: Pulseq output filename
        """

        print("Create Pulseq sequence file.")
        start_time = time.time()

        filename = os.path.splitext(filename)[0] + '.seq'

        system = pp.Opts(
            max_grad=1e4,
            max_slew=1e4,
            rf_raster_time=self.delta['rf'] * self.cf_time,
            grad_raster_time=self.delta['grad'] * self.cf_time,
            grad_unit='mT/m',
            slew_unit='mT/m/ms',
            rf_ringdown_time=0,
            rf_dead_time=0
        )

        pp_seq = pp.Sequence(system=system)
        pp_seq.set_definition('Name', os.path.basename(filename))


        event_check = {'rf': 0, 'gx': 0, 'gy': 0, 'gz': 0, 'adc': 0, 'trig': 0}
        pp_events = []
        ts_offset = 0
        last_ts = 0

        for ix, block in enumerate(self.block_list):
            block_offset = self.block_list[ix - 1].block_duration if ix > 0 else 0
            ts_offset -= block_offset # offset time if Siemens block is splitted
            for ts in block.timestamps:
                last_ts = ts
                events = block.timestamps[ts]
                concat_g = {'gx': False, 'gy': False, 'gz': False}
                pulseq_time = int(ts) - ts_offset # timestamp in the current Pulseq block [us]

                # Check if the block has to be split (Pulseq only allows one event per channel per block)
                # triggers cannot be split, so we check if there is a trigger in the block
                # gradients are either splitted or concatenated, depending on the presence of ADC or RF events
                if any(event_check[event.type] for event in events):
                    split = any((int(np.round(pp.calc_duration(pp_event) / self.cf_time)) > pulseq_time) for pp_event in pp_events)
                    check_trig = any(event.type == 'trig' for event in events)
                    if split and not check_trig:
                        check_adc = any(event.type == 'adc' for event in events) and event_check['adc'] > 0
                        check_rf = any(event.type == 'rf' for event in events) and event_check['rf'] > 0
                        if check_adc or check_rf:
                            pp_events_tmp = []
                            event_check = dict.fromkeys(event_check, 0)
                            for i, pp_event in enumerate(pp_events):
                                pp_dur = int(np.round(pp.calc_duration(pp_event) / self.cf_time))
                                if check_adc:
                                    split_pt = int(pulseq_time / self.delta['grad'])
                                elif check_rf:
                                    split_pt = int((pulseq_time - self.rf_lead_time - self.rf_hold_time) / self.delta['grad'])
                                else:
                                    raise ValueError("Unable to determine split point: neither ADC nor RF split is applicable.")
                                if split_pt < 0:
                                    raise ValueError("Invalid split point.")
                                if pp_dur > pulseq_time:
                                    if hasattr(pp_event, 'waveform') or hasattr(pp_event, 'amplitude'):
                                        g_wf = waveform_from_seqblock(pp_event, system=system)
                                        g_new = pp.make_arbitrary_grad(channel=pp_event.channel, waveform=g_wf[:split_pt], delay=pp_event.delay, system=system)
                                        pp_events_tmp.append(pp.make_arbitrary_grad(channel=pp_event.channel, waveform=g_wf[split_pt:], delay=0, system=system))
                                        pp_events[i] = g_new
                                        event_check['g' + pp_event.channel] = 1
                                    else:
                                        raise ValueError(f"ADC, RF or trigger event in block with index {block.block_idx} cannot be split. Only gradients are supported.")
                            pp_seq.add_block(*pp_events)
                            pp_events = pp_events_tmp.copy()
                            if check_adc:
                                ts_offset = int(int(ts) / self.delta['grad']) * self.delta['grad']
                            elif check_rf:
                                ts_offset = int(ts) - self.rf_lead_time - self.rf_hold_time
                        else:
                            for event in events:
                                if event_check[event.type] > 0:
                                    event_del = event.delay - ts_offset
                                    if event_del < 0:
                                        raise ValueError("Negative event delay encountered during concatenation.")
                                    pp_event = pp_events[event_check[event.type] - 1]
                                    if event.type == 'trig':
                                        raise ValueError(f"Trigger event in block with index {block.block_idx} cannot be concatenated.")
                                    elif event.type[0] == 'g':
                                        g_conc = self.__make_pp_grad(event, event_del, system)
                                        if g_conc is not None:
                                            g_wf = [waveform_from_seqblock(pp_event, system=system), waveform_from_seqblock(g_conc, system=system)]
                                            len_wf = max([len(wf) for wf in g_wf])
                                            g_wf = [np.concatenate([wf, np.zeros(len_wf - len(wf))]) for wf in g_wf]
                                            g_wf = g_wf[0] + g_wf[1]
                                            pp_events[event_check[event.type] - 1] = pp.make_arbitrary_grad(channel=event.channel, waveform=g_wf, system=system)
                                        concat_g[event.type] = True
                                    else:
                                        raise ValueError("Unrecognized event type during concatenation.")
                    else:
                        delay = round_up_to_raster((int(ts) - ts_offset) * self.cf_time, 5)
                        if self.__check_delay(pp_events, delay):
                            pp_events.append(pp.make_delay(delay))
                        pp_seq.add_block(*pp_events)
                        pp_events = []
                        ts_offset = int(ts)
                        event_check = dict.fromkeys(event_check, 0)

                # add events to the block
                for event in events:
                    event_del = event.delay - ts_offset
                    if event_del < 0:
                        raise ValueError("Negative event delay encountered.")
                    if event.type == 'rf':
                        if event_del < self.rf_lead_time:
                            raise ValueError(f"RF lead time violation in block with index {block.block_idx}")
                        rf = self.__make_pp_rf(event, event_del, system)
                        pp_events.append(rf)
                        event_check[event.type] = len(pp_events)
                        delay = pp.calc_duration(rf) + self.rf_hold_time * self.cf_time
                        if self.__check_delay(pp_events, delay):
                            pp_events.append(pp.make_delay(delay))
                    elif event.type in ['gx', 'gy', 'gz']:
                        if not concat_g[event.type]:
                            g = self.__make_pp_grad(event, event_del, system)
                            if g is not None:
                                pp_events.append(g)
                                event_check[event.type] = len(pp_events)
                    elif event.type == 'adc':
                        adc_dur = round_up_to_raster(event.duration * self.cf_time, 7) # ADCs can be on nanosecond raster
                        adc_del = round_up_to_raster(event_del * self.cf_time, 6)
                        adc = pp.make_adc(num_samples=event.samples, duration=adc_dur, delay=adc_del, freq_offset=event.freq, phase_offset=np.deg2rad(event.phase), system=system)
                        pp_events.append(adc)
                        event_check[event.type] = len(pp_events)
                    elif event.type == 'trig':
                        trig_dur = round_up_to_raster(event.duration * self.cf_time, 6)
                        trig_del = round_up_to_raster(event_del * self.cf_time, 6)
                        if event.trig_type in self.trig_types:
                            trig = pp.make_digital_output_pulse(channel=self.trig_types[event.trig_type], duration=trig_dur, delay=trig_del, system=system)
                            pp_events.append(trig)
                            event_check[event.type] = len(pp_events)
                        else:
                            delay = trig_dur + trig_del
                            if self.__check_delay(pp_events, delay):
                                pp_events.append(pp.make_delay(d=delay))
                            print(f'Unknown trigger type {event.trig_type} in block {block.block_idx} at time stamp {ts}. Replaced with delay.')

        # add last block
        if pp_events:
            delay = round_up_to_raster((int(last_ts) - ts_offset) * self.cf_time, 5)
            if self.__check_delay(pp_events, delay):
                pp_events.append(pp.make_delay(delay))  # account for possible delay in last block
            pp_seq.add_block(*pp_events)

        # check timing of the sequence
        # set raster time to 1us for timing check, as otherwise ADC delays will throw errors
        pp_seq.system.rf_raster_time = 1e-6
        ok, error_report = pp_seq.check_timing()
        pp_seq.system.rf_raster_time = self.delta['rf'] * self.cf_time
        if not ok:
            raise ValueError(f"PyPulseq timing check failed: {error_report}")
        
        # write sequence
        pp_seq.write(filename, check_timing=False)
        end_time = time.time()
        print(f"Finished creating Pulseq file in {(end_time - start_time):.2f}s.")

    def __make_pp_rf(self, rf_event, event_del, system):
            """
            Make a Pulseq RF event
            """

            rf_sig = self.get_shape(rf_event)
            rf_del = round_up_to_raster(event_del*self.cf_time, 6)
            rf = pp.make_arbitrary_rf(signal=rf_sig, flip_angle=1, delay=rf_del, freq_offset=rf_event.freq, phase_offset=np.deg2rad(rf_event.phase), return_gz=False, system=system)
            rf.signal = rf_sig * self.cf_rf # reset the signal as it gets scaled in make_arbitrary_rf
            return rf

    def __make_pp_grad(self, grad_event, event_del, system):
        """
        Make a Pulseq gradient event
        """

        if grad_event.ramp_dn != 0:
            # trapezoid
            g_flat = round_up_to_raster((grad_event.duration - grad_event.ramp_up) * self.cf_time, 5)
            g_ramp_up = round_up_to_raster(grad_event.ramp_up*self.cf_time, 5)
            g_ramp_dn = round_up_to_raster(grad_event.ramp_dn*self.cf_time, 5)
            g_del = round_up_to_raster(event_del*self.cf_time, 5)
            return pp.make_trapezoid(channel=grad_event.channel, amplitude=grad_event.amp*self.cf_grad, flat_time=g_flat, rise_time=g_ramp_up, fall_time=g_ramp_dn, delay=g_del, system=system)
        elif grad_event.duration == 0 and grad_event.ramp_up == 0:
            # zero duration gradient
            return None
        else:
            # arbitrary
            g_wf = self.get_shape(grad_event)
            return pp.make_arbitrary_grad(channel=grad_event.channel, waveform=g_wf*self.cf_grad, delay=event_del*self.cf_time, system=system)

    def __check_delay(self, pp_events, delay):
        """
        Check if delay is the longest delay in the current event block
        """
        delays = [item.delay for item in pp_events if item.type == "delay"] + [0]
        if delay > max(delays):
            return True
        else:
            return False