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

    # read event shapes
    rfd = DSVFile(file_prefix+"_RFD.dsv")
    rfp = DSVFile(file_prefix+"_RFP.dsv")
    grx = DSVFile(file_prefix+"_GRX.dsv")
    gry = DSVFile(file_prefix+"_GRY.dsv")
    grz = DSVFile(file_prefix+"_GRZ.dsv")

    shapes = [rfd, rfp, grx, gry, grz]
    seq.set_shapes(shapes)

    # Read block structure
    read_dsv_inf(file_prefix+"_INF.dsv", seq)

    if plot:
        plot_seq([rfd,rfp,grx,gry,grz])

    return seq


seq = read_dsv("dsv_test/gre", plot=True)
