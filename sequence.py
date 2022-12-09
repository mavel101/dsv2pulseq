import numpy as np

class Block():
    """
    Event block in "Siemens style", allows more than one object per channel
    All units are in Siemens units:
    times: [us]
    gradients: [mT/m]
    RF pulses: [V]
    """

    def __init__(self, idx, duration):
        self.block_duration = duration
        self.block_idx = idx
        self.timestamps = {}

    def add_timestamp(self, ts):
        if not str(ts) in self.timestamps:
            self.timestamps[str(ts)] = []

    def add_rf(self, duration, shape, ts):
        self.add_timestamp(ts)
        rf = Rf(duration, shape, ts)
        self.timestamps[str(ts)].append(rf)
    
    def add_grad(self, channel, amp, duration, ramp_up, ramp_dn, shape, ts):
        self.add_timestamp(ts)
        grad = Grad(channel, amp, duration, ramp_up, ramp_dn, shape, ts)
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

    def __init__(self, duration, shape, delay):
        self.type = 'rf'
        self.duration = duration
        self.delay = delay
        self.freq = 0
        self.phase = 0
        self.shape = np.array(shape)

    def set_freqphase(self, freq_phase):
        self.freq = freq_phase[0]
        self.phase = freq_phase[1]
    
class Grad():

    def __init__(self, channel, amp, duration, ramp_up, ramp_dn, shape, delay):
        self.type = 'g' + channel 
        self.channel = channel
        self.amp = amp
        self.duration = duration # flat_top + ramp_up
        self.ramp_up = ramp_up
        self.ramp_dn = ramp_dn
        self.delay = delay
        self.shape = np.array(shape)

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

    def add_block(self, idx, duration):
        self.n_blocks += 1
        self.duration += duration
        self.block_list.append(Block(idx, duration))

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

    def check_shapes(self, shape_ix):
        print(len(self.rf_val), shape_ix[0])
        print(len(self.gx_val), shape_ix[1])
        print(len(self.gy_val), shape_ix[2])
        print(len(self.gz_val), shape_ix[3])


    def write_pulseq(self):
        import pypulseq as pp

        pp_seq = pp.Sequence()
        for block in self.block_list:
            pass
        # WIP: check if event block already contains event of same kind, then open new block
        # WIP: split gradients if necessary
        # WIP: detect delays by comparing the timestamp with the longest previous event
        # WIP: detect trapezoidal/arbitrary gradients via ramp_down time (arbitray has ramp down zero)
        # WIP: convert units

