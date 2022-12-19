"""
This code is copied from the dsvreader from Markus Boland.
"""

# coding=utf-8
import numpy as np
import codecs
from collections import namedtuple


class DSVFile:
    """
    This class loads and uncompresses a dsv file.
    """

    def __decode_dsv_values(self):
        """
        Decodes a compressed dsv array
        :return: Decompressed dsv array

        The values are signed 32-bit integers. The real value
        will be obtained after division by VERTFACTOR.
        If the same sample value is repeated several times,
        the value is only listed twice, and the third value
        specifies the number of repetitions.
        Only the first sample is stored as an absolute value.
        All other values are deltas to the preceeding value.
        (This way it is possible to compress linear slopes.)
        """
        ndsv = np.zeros((int(self.definitions.samples)), dtype=np.int)
        dsv = np.array(self.values, dtype=np.int)
        count = 0
        ncount = 1
        ndsv[ncount] = dsv[count]
        prev = dsv[0]
        while count < dsv.shape[0]-1:
            count += 1
            if not (dsv[count] == prev):  # no repetition
                ndsv[ncount] = ndsv[ncount-1] + dsv[count]
                prev = dsv[count]
                ncount += 1
            else:  # number repeated
                # get repetitions
                nrep = dsv[count+1]
                ndsv[ncount] = ndsv[ncount-1] + dsv[count]
                ncount += 1
                for i in range(nrep):
                    ndsv[ncount] = ndsv[ncount-1] + dsv[count]
                    ncount += 1
                # store last value
                prev = dsv[count]

                # inc counters
                count += 1

        # apply vertical factor
        ndsv = ndsv / self.definitions.vertfactor
        return ndsv

    @staticmethod
    def __isfloat(value):
        """
        Returns if string contains a valid float
        :param value: String
        :return: Float?
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def __init__(self, filename, encoding='cp1252'):
        """
        Loads the data from filename
        :param filename: Path to dsv file
        :param encoding: Encoding of dsv file
        :return:
        The values are automatically uncompressed and stored in values field. Other parameters
        are stored in field given by the structure of the dsv file. The values are converted
        to float if possible and remain unicode if not (should be right for all params).
        """

        # load file
        dsv_file = codecs.open(filename, 'r', encoding)
        try:
            filecnt = dsv_file.readlines()
        finally:
            dsv_file.close()

        # trim lines
        filecnt = list(map(str.strip, filecnt))
        filecnt = [x for x in filecnt if len(x) > 0]
        filecnt = [x for x in filecnt if not x.startswith(';')]
        filecnt = list(map(str.lower, filecnt))

        file_data = dict()
        dictkeys = {'definitions', 'filetype', 'frame'}
        listkeys = {'values'}
        curkey = ''
        for line in filecnt:
            # Section line
            if line.startswith('[') and line.endswith(']'):
                line = line[1:-1]
                if line in dictkeys:
                    file_data[line] = dict()
                    curkey = line
                elif line in listkeys:
                    setattr(self, line, list())
                    curkey = line
                else:
                    print('Unknown key {0} ... skipping.'.format(line))
            # Key value pair
            elif line.find('=') > 0:
                key, val = line.split('=')
                if self.__isfloat(val):
                    val = float(val)
                file_data[curkey][key] = val
            elif curkey in listkeys:
                getattr(self, curkey).append(int(line))

        # set params
        for _d in dictkeys:
            setattr(self, _d, namedtuple(_d, list(file_data[_d].keys()))(**file_data[_d]))

        # uncompress values
        self.values = self.__decode_dsv_values()
        self.time = np.arange(self.values.shape[0]) * self.definitions.horidelta
        pass

    def plot(self, smin=0, skip=1000, smax=1000000):
        """
        Plots the stored dsv data
        :param smin: Start index to plot
        :param skip: Index delta
        :param smax: Stop index
        :return:
        """
        import matplotlib.pyplot as plt

        smin *= int(10./self.definitions.horidelta)
        smax *= int(10./self.definitions.horidelta)
        skip *= int(10./self.definitions.horidelta)

        plt.plot(self.time[smin:smax:skip], self.values[smin:smax:skip])
        plt.xlabel(self.definitions.horiunitname)
        plt.ylabel(self.definitions.vertunitname)
        pass
