import numpy as np

class Block():
    """
    Event block in "Siemens style", allows more than one object per channel
    All units are in Siemens units:
    times: [us]
    gradients: [mT/m]
    RF pulses: [V]
    """

    # WIP: Make dictionary of events with time stamp as the key
    def __init__(self, idx, duration):
        self.block_duration = duration
        self.block_idx = idx
        self.rf_events = []
        self.grx_events = []
        self.gry_events = []
        self.grz_events = []
        self.adc_events = []
        self.trig_events = []
        self.delays = []

    def add_rf(self, duration, delay):
        rf = Rf(duration, delay)
        self.rf_events.append(rf)
    
    def add_grad(self, channel, amp, duration, ramp_up, ramp_dn, delay):
        grad = Grad(channel, amp, duration, ramp_up, ramp_dn, delay)
        if channel == 'x':
            self.grx_events.append(grad)
        elif channel == 'y':
            self.gry_events.append(grad)
        elif channel == 'z':
            self.grz_events.append(grad)
        else:
            raise ValueError("Invalid gradient channel.")

    def add_adc(self, duration, samples, delay):
        adc = Adc(duration, samples, delay)
        self.adc_events.append(adc)

    def add_trig(self, duration, delay):
        trig = Trig(duration, delay)
        self.trig_events.append(trig)

    def add_delay(self, duration):
        delay = Delay(duration)
        self.delays.append(delay)

    def set_freqphase(self, freq_phase, delay):
        for rf in self.rf_events:
            if rf.delay == delay:
                rf.set_freqphase(freq_phase)
        for adc in self.adc_events:
            if adc.delay == delay:
                adc.set_freqphase(freq_phase)

class Rf():

    def __init__(self, duration, delay):
        self.duration = duration
        self.delay = delay
        self.freq = 0
        self.phase = 0
        self.shape = np.array([])

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]

    def set_shape(self, shape):
        self.shape = np.array(shape)
    
class Grad():

    def __init__(self, channel, amp, duration, ramp_up, ramp_dn, delay):
        self.channel = channel
        self.amp = amp
        self.duration = duration # flat_top + ramp_up
        self.ramp_up = ramp_up
        self.ramp_dn = ramp_dn
        self.delay = delay
        self.shape = np.array([])

    def set_shape(self, shape):
        self.shape = np.array(shape)

class Adc():

    def __init__(self, duration, samples, delay):
        self.duration = duration
        self.samples = samples
        self.delay = delay
        self.freq = 0
        self.phase = 0

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]

class Trig():

    def __init__(self, duration, delay):
        self.duration = duration
        self.delay = delay

class Delay():

    def __init__(self, duration):
        self.duration = duration

class Sequence():

    def __init__(self):
        self.n_blocks = 0
        self.duration = 0
        self.block_list = []
        self.rf_shp = np.array([])
        self.gx_shp = np.array([])
        self.gy_shp = np.array([])
        self.gz_shp = np.array([])

    def add_block(self, idx, duration):
        self.n_blocks += 1
        self.duration += duration
        self.block_list.append(Block(idx, duration))

    def get_block(self, idx):
        return self.block_list[idx]

    def set_shapes(self, shapes):
        self.rf_shp = shapes[0]
        self.gx_shp = shapes[1]
        self.gy_shp = shapes[2]
        self.gz_shp = shapes[3]

    def write_pulseq(self):
        import pypulseq as pp

        pp_seq = pp.Sequence()
        for block in self.block_list:
            pass
        # WIP: check if event block already contains event of same kind, then open new block
        # WIP: split gradients if necessary
        # WIP: detect trapezoidal/arbitrary gradients via ramp_down time (arbitray has ramp down zero)
        # WIP: convert units

