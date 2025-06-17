import numpy as np
import os
import time
import copy
from warnings import warn
import logging
import pypulseq as pp
from dsv2pulseq.helper import round_up_to_raster, waveform_from_seqblock


def init_event_dict():
    return {'rf': None, 'gx': None, 'gy': None, 'gz': None, 'adc': None, 'trig': None, 'delay': None}

def extract_events(event_dict):
    return [event for event in event_dict.values() if event is not None]

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

    def add_rf(self, duration, shape_ix, ts):
        self.add_timestamp(ts)
        rf = Rf(duration, shape_ix)
        self.timestamps[str(ts)].append(rf)
    
    def add_grad(self, channel, amp, duration, ramp_up, ramp_dn, shape_ix, ts):
        self.add_timestamp(ts)
        grad = Grad(channel, amp, duration, ramp_up, ramp_dn, shape_ix)
        self.timestamps[str(ts)].append(grad)

    def add_adc(self, duration, samples, ts):
        self.add_timestamp(ts)
        adc = Adc(duration, samples)
        self.timestamps[str(ts)].append(adc)

    def add_trig(self, duration, trig_type, ts):
        self.add_timestamp(ts)
        trig = Trig(duration, trig_type)
        self.timestamps[str(ts)].append(trig)

    def set_freqphase(self, freq_phase, ts):
        for event in self.timestamps[str(ts)]:
            if event.type == 'rf' or event.type == 'adc':
                event.set_freqphase(freq_phase)

class Rf():

    def __init__(self, duration, shape_ix):
        self.type = 'rf'
        self.duration = duration
        self.freq = 0
        self.phase = 0
        self.shape_ix = shape_ix

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]
    
class Grad():

    def __init__(self, channel, amp, duration, ramp_up, ramp_dn, shape_ix):
        self.type = 'g' + channel 
        self.channel = channel
        self.amp = amp
        self.duration = duration # flat_top + ramp_up
        self.ramp_up = ramp_up
        self.ramp_dn = ramp_dn
        self.shape_ix = shape_ix

class Adc():

    def __init__(self, duration, samples):
        self.type = 'adc'
        self.duration = duration
        self.samples = samples
        self.freq = 0
        self.phase = 0

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]

class Trig():

    def __init__(self, duration, trig_type):
        self.type = 'trig'
        self.duration = duration
        self.trig_type = trig_type

class Sequence():

    def __init__(self, ref_volt):
        self.n_blocks = 0
        self.duration = 0
        self.block_list = []

        self.rf = np.array([])
        self.gx = np.array([])
        self.gy = np.array([])
        self.gz = np.array([])

        # raster times
        self.delta_rf = 5 # RF raster times in dsv file [us]
        self.delta_grad = 10 # Gradient raster times in dsv file [us]

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

        # ADC dead time
        self.adc_dead_time = 10 # [us]

    def add_block(self, idx, duration, start_time):
        self.n_blocks += 1
        self.duration = start_time + duration
        self.block_list.append(Block(idx, duration, start_time))

    def get_block(self, idx):
        return self.block_list[idx]

    def set_shapes(self, shapes):

        rf = shapes[0].values * np.exp(1j*np.deg2rad(shapes[1].values))
        self.rf = rf
        self.gx = shapes[2].values
        self.gy = shapes[3].values
        self.gz = shapes[4].values
        self.delta_rf = int(shapes[0].definitions.horidelta)
        self.delta_grad= int(shapes[2].definitions.horidelta)

    def get_shape(self, event):
        """
        Gets shape for RF or gradient event
        """
        if event.type == 'rf':
            return self.rf[event.shape_ix]
        elif event.type == 'gx':
            return self.gx[event.shape_ix]
        elif event.type == 'gy':
            return self.gy[event.shape_ix]
        elif event.type == 'gz':
            return self.gz[event.shape_ix]
        else:
            return None
        
    def set_lead_hold_time(self, rf_lead_time, rf_hold_time):
        """
        Set RF lead and hold times
        """
        self.rf_lead_time = rf_lead_time
        self.rf_hold_time = rf_hold_time

    def set_adc_dead_time(self, adc_dead_time):
        """
        Set ADC dead time
        """
        self.adc_dead_time = adc_dead_time

    def make_pulseq_sequence(self, filename=None):
        """
        Create a Pulseq file from the sequence object.
        Time tracking is always done in [us] (integers), 
        time is only converted to [s], when creating a pulseq event
        
        Inputs:
        --------
        filename: If provided, Pulseq sequence will be saved to this file.
        """

        logging.info("Create Pulseq sequence file.")
        start_time = time.time()

        filename = os.path.splitext(filename)[0] + '.seq'

        system = pp.Opts(
            max_grad=1e4,
            max_slew=1e4,
            rf_raster_time=self.delta_rf * self.cf_time,
            grad_raster_time=self.delta_grad * self.cf_time,
            grad_unit='mT/m',
            slew_unit='mT/m/ms',
            rf_ringdown_time=self.rf_hold_time * self.cf_time,
            rf_dead_time=self.rf_lead_time * self.cf_time,
            adc_dead_time=self.adc_dead_time * self.cf_time
        )

        pp_seq = pp.Sequence(system=system)
        pp_seq.set_definition('Name', os.path.basename(filename))

        pulseq_events = init_event_dict()
        ts_offset = 0

        if not self.block_list:
            warn("No blocks in sequence. Please check the input DSV files. Exiting.")
            return

        # Shift RF and ADC timestamps to account for lead and dead times
        block_list = self.shift_timestamps()

        for ix, block in enumerate(block_list):
            block_offset = block_list[ix - 1].block_duration if ix > 0 else 0
            ts_offset -= block_offset # offset time if Siemens block is splitted
            for ts in block.timestamps:
                events = block.timestamps[ts]
                concat_g = {'gx': False, 'gy': False, 'gz': False}
                pulseq_time = int(ts) - ts_offset # timestamp in the current Pulseq block [us]

                # Check if the block has to be split (Pulseq only allows one event per channel per block)
                # Gradients are either splitted or concatenated, depending on the presence of ADC, RF or Trigger events
                if any(pulseq_events[event.type] is not None for event in events):
                    split = any((round(pp.calc_duration(pp_event) / self.cf_time) > pulseq_time) for pp_event in pulseq_events.values() if pp_event is not None)
                    if split: # block needs to be split or concatenated
                        check_adc = any(event.type == 'adc' for event in events) and pulseq_events['adc'] is not None
                        check_rf = any(event.type == 'rf' for event in events) and pulseq_events['rf'] is not None
                        check_trig = any(event.type == 'trig' for event in events) and pulseq_events['trig'] is not None
                        split_pt = round(pulseq_time / self.delta_grad)
                        if check_adc or check_rf or check_trig: # split block
                            pulseq_events_tmp = init_event_dict()
                            for key, pp_event in pulseq_events.items():
                                if pp_event is None:
                                    continue
                                pp_dur = round(pp.calc_duration(pp_event) / self.cf_time)
                                if pp_dur > pulseq_time:
                                    if hasattr(pp_event, 'waveform') or hasattr(pp_event, 'amplitude'):
                                        g_pre, g_post = self.__split_gradients(pp_event, split_pt, system)
                                        pulseq_events_tmp[key] = g_post
                                        pulseq_events[key] = g_pre
                                    else:
                                        raise ValueError(f"Event with type {pp_event.type} in block with index {block.block_idx} cannot be split. Only gradients can be split.")
                            pp_seq.add_block(*extract_events(pulseq_events))
                            pulseq_events = pulseq_events_tmp.copy()
                            ts_offset = round(int(ts) / self.delta_grad) * self.delta_grad
                        else: # concatenate gradients
                            for event in events:
                                if pulseq_events[event.type] is not None:
                                    event_del = int(ts) - ts_offset
                                    if event_del < 0:
                                        raise ValueError("Negative event delay encountered during concatenation.")
                                    pp_event = pulseq_events[event.type]
                                    if event.type == 'trig':
                                        raise ValueError(f"Trigger event in block with index {block.block_idx} cannot be concatenated.")
                                    elif event.type[0] == 'g':
                                        g_conc = self.__make_pp_grad(event, event_del, system)
                                        if g_conc is not None:
                                            g_wf = [waveform_from_seqblock(pp_event, system=system), waveform_from_seqblock(g_conc, system=system)]
                                            len_wf = max([len(wf) for wf in g_wf])
                                            g_wf = [np.concatenate([wf, np.zeros(len_wf - len(wf))]) for wf in g_wf]
                                            g_wf = g_wf[0] + g_wf[1]
                                            pulseq_events[event.type] = pp.make_arbitrary_grad(channel=event.channel, waveform=g_wf, system=system)
                                        concat_g[event.type] = True
                                    else:
                                        raise ValueError("Unrecognized event type during concatenation.")
                    else: # no split needed, start new block
                        block_dur = pp.calc_duration(*extract_events(pulseq_events))
                        block_dur_rounded = round_up_to_raster(pp.calc_duration(*extract_events(pulseq_events)), 5)
                        if abs(block_dur - block_dur_rounded) > 1e-7:
                            pulseq_events['delay'] = pp.make_delay(block_dur_rounded)
                        pp_seq.add_block(*extract_events(pulseq_events))

                        # Add remaining delay to the new block
                        delay = round_up_to_raster((int(ts) - ts_offset) * self.cf_time, 5)
                        delay_remain = delay - block_dur_rounded
                        pulseq_events = init_event_dict()
                        pulseq_events['delay'] = pp.make_delay(delay_remain)
                        ts_offset += round(block_dur_rounded  / self.cf_time)

                # add events to the block
                for event in events:
                    event_del = int(ts) - ts_offset
                    if event_del < 0:
                        raise ValueError("Negative event delay encountered.")
                    if event.type == 'rf':
                        event_del += self.rf_lead_time
                        rf = self.__make_pp_rf(event, event_del, system)
                        pulseq_events['rf'] = rf
                    elif event.type in ['gx', 'gy', 'gz']:
                        if not concat_g[event.type]:
                            g = self.__make_pp_grad(event, event_del, system)
                            if g is not None:
                                pulseq_events[event.type] = g
                    elif event.type == 'adc':
                        event_del += self.adc_dead_time
                        adc_dur = round_up_to_raster(event.duration * self.cf_time, 7) # ADCs can be on nanosecond raster
                        adc_del = round_up_to_raster(event_del * self.cf_time, 6)
                        adc = pp.make_adc(num_samples=event.samples, duration=adc_dur, delay=adc_del, freq_offset=event.freq, phase_offset=np.deg2rad(event.phase), system=system)
                        pulseq_events['adc'] = adc
                    elif event.type == 'trig':
                        trig_dur = round_up_to_raster(event.duration * self.cf_time, 6)
                        trig_del = round_up_to_raster(event_del * self.cf_time, 6)
                        if event.trig_type in self.trig_types:
                            trig = pp.make_digital_output_pulse(channel=self.trig_types[event.trig_type], duration=trig_dur, delay=trig_del, system=system)
                            pulseq_events['trig'] = trig
                        else:
                            delay = trig_dur + trig_del
                            if self.__check_delay(pulseq_events, delay):
                                pulseq_events['trig'] = pp.make_delay(d=delay)
                            logging.info(f'Unknown trigger type {event.trig_type} in block {block.block_idx} at time stamp {ts}. Replaced with delay.')

        # add last block
        delay = round_up_to_raster((int(ts) - ts_offset) * self.cf_time, 5)
        if self.__check_delay(pulseq_events, delay):
            pulseq_events['delay'] = pp.make_delay(delay)  # account for possible delay in last block
        if any(pulseq_events.values()):
            pp_seq.add_block(*extract_events(pulseq_events))

        # check timing of the sequence
        # set raster time to 1us for timing check, as otherwise ADC delays will throw errors
        pp_seq.system.rf_raster_time = 1e-6
        ok, error_report = pp_seq.check_timing()
        pp_seq.system.rf_raster_time = self.delta_rf * self.cf_time
        if not ok:
            warn(f"PyPulseq timing check failed with {len(error_report)} errors.")
        
        # write sequence
        pp_seq.write(filename, check_timing=False)
        end_time = time.time()
        logging.info(f"Finished creating Pulseq file in {(end_time - start_time):.2f}s.")

        return pp_seq

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

    def __split_gradients(self, grad_event, split_pt, system):
        """
        Split gradients at the split point.
        """
        def create_grad(wf, channel, system):
            grad = pp.make_arbitrary_grad(channel=channel, waveform=wf, delay=0, system=system)
            return grad

        # Pypulseq throws an error if the waveform has only one point.
        def fix_single_point_grad(wf, channel, system):
            grad = pp.make_arbitrary_grad(
                channel=channel,
                waveform=np.concatenate([wf, np.zeros(1, dtype=float)]),
                delay=0,
                system=system
            )
            grad.waveform = wf
            grad.tt = (np.arange(len(wf)) + 0.5) * system.grad_raster_time
            grad.shape_dur = len(wf) * system.grad_raster_time
            grad.first = wf[0]
            grad.last = wf[0]
            grad.area = np.sum(wf * system.grad_raster_time)
            return grad

        g_wf = waveform_from_seqblock(grad_event, system=system) # already contains delay
        g_pre_wf = g_wf[:split_pt]
        g_post_wf = g_wf[split_pt:]

        if len(g_pre_wf) == 1:
            g_pre = fix_single_point_grad(g_pre_wf, grad_event.channel, system)
        else:
            g_pre = create_grad(g_pre_wf, grad_event.channel, system)

        if len(g_post_wf) == 1:
            g_post = fix_single_point_grad(g_post_wf, grad_event.channel, system)
        else:
            g_post = create_grad(g_post_wf, grad_event.channel, system)

        return g_pre, g_post

    def __check_delay(self, pulseq_events, delay):
        """
        Check if delay is the longest delay in the current event block
        """
        pulseq_events = extract_events(pulseq_events)
        delays = [item.delay for item in pulseq_events if (item.type == "delay" and item is not None)] + [0]
        if delay > max(delays):
            return True
        else:
            return False
        
    def shift_timestamps(self):
        """
        Shift RF events by the RF lead time and ADC events by the ADC dead time
        to make sure they are not violated, when creating the Pulseq sequence.

        Returns:
        --------
        A copy of self.block_list with 'rf' events shifted earlier by self.rf_lead_time.
        The original self.block_list remains unchanged.
        """
        
        # Deep copy to avoid modifying original blocks or event lists
        block_list_shifted = copy.deepcopy(self.block_list)

        for block in block_list_shifted:
            shifted_timestamps = {}

            for ts_str in block.timestamps:
                ts = int(ts_str)
                events = block.timestamps[ts_str]
                if not events and ts not in shifted_timestamps:
                    shifted_timestamps[ts] = []
                for event in events:
                    if event.type == 'rf':
                        new_ts = ts - self.rf_lead_time
                        if new_ts not in shifted_timestamps:
                            shifted_timestamps[new_ts] = []
                        shifted_timestamps[new_ts].append(event)
                    elif event.type == 'adc':
                        new_ts = ts - self.adc_dead_time
                        if new_ts not in shifted_timestamps:
                            shifted_timestamps[new_ts] = []
                        shifted_timestamps[new_ts].append(event)
                    else:
                        if ts not in shifted_timestamps:
                            shifted_timestamps[ts] = []
                        shifted_timestamps[ts].append(event)

            # Sort timestamps and convert keys back to strings
            block.timestamps = dict(
                sorted(((str(ts), evts) for ts, evts in shifted_timestamps.items()), key=lambda x: int(x[0]))
            )

        return block_list_shifted