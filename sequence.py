import numpy as np

class Block():

    def __init__(self, duration):
        self.block_duration = duration
        self.events = []

class Rf():

    def __init__(self, duration, delay):
        self.duration = duration
        self.delay = delay
        self.freq = 0
        self.phase = 0
        self.shape = np.array([])

    def set_freqphase(self, freq, phase):
        self.freq = freq
        self.phase = phase

    def set_shape(self, shape):
        self.shape = np.array(shape)
    
class Grad():

    def __init__(self, duration, delay):
        self.duration = duration
        self.delay = delay
        self.shape = np.array([])

    def set_shape(self, shape):
        self.shape = np.array(shape)

class Adc():

    def __init__(self, duration, delay, samples):
        self.duration = duration
        self.delay = delay
        self.samples = samples
        self.freq = 0
        self.phase = 0

    def set_freqphase(self, freq, phase):
        self.freq = freq
        self.phase = phase

class Trig():

    def __init__(self, duration, delay):
        self.duration = duration
        self.delay = delay

class Sequence():

    def __init__(self):
        self.n_blocks = 0
        self.block_list = []

    def add_block(self, duration):
        self.block_list.append(Block(duration))

    def write_pulseq(self):
        import pypulseq as pp
        pass
