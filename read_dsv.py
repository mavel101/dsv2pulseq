import numpy as np

from sequence import Sequence
from read_dsv_samples import DSVFile
from read_dsv_inf import read_dsv_inf

# WIP: add init, add script for command line, add setup.py

def plot_seq(dsv):
    """
    Plots the complete sequence
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20,20))
    plt.subplot(511)
    dsv[0].plot()
    plt.subplot(512)
    dsv[1].plot()
    plt.subplot(513)
    dsv[2].plot()
    plt.subplot(514)
    dsv[3].plot()
    plt.subplot(515)
    dsv[4].plot()

def read_dsv(file_prefix, plot=False):
    """ 
    Reads dsv files and returns complete sequence
    """
    seq = Sequence()

    read_dsv_inf(file_prefix+"_INF.dsv", seq)
    rfd = DSVFile(file_prefix+"_RFD.dsv")
    rfp = DSVFile(file_prefix+"_RFP.dsv")
    grx = DSVFile(file_prefix+"_GRX.dsv")
    gry = DSVFile(file_prefix+"_GRY.dsv")
    grz = DSVFile(file_prefix+"_GRZ.dsv")

    if plot:
        plot_seq([rfd,rfp,grx,gry,grz])

    # WIP: All unit set correctly?
    # WIP: Rotation of gradients?
    values = [rfd.values, rfp.values, grx.values, gry.values, grz.values]
    set_shapes(seq, values)

    return seq


def set_shapes(seq, values):
    """
    Set shapes of RF and gradient events
    """

    rf_val = values[0] * np.exp(1j*np.deg2rad(values[1]))
    grx_val = values[2]
    gry_val = values[3]
    grz_val = values[4]

    pass

def convert_units():
    """
    Convert units to SI (Pulseq units)
    """
    pass

# seq = read_dsv("dsv_test/gre", plot=True)
