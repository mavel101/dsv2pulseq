import time

from dsv2pulseq.sequence import Sequence
from dsv2pulseq.read_dsv_samples import DSVFile
from dsv2pulseq.read_dsv_inf import read_dsv_inf

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

def read_dsv(file_prefix, ref_volt=223.529007, plot=False):
    """ 
    Reads dsv files and returns complete sequence
    """
    seq = Sequence(ref_volt)

    # read event shapes
    print("Read RF and gradient shapes.")
    start_dsv = time.time()
    rfd = DSVFile(file_prefix+"_RFD.dsv")
    rfp = DSVFile(file_prefix+"_RFP.dsv")
    grx = DSVFile(file_prefix+"_GRX.dsv")
    gry = DSVFile(file_prefix+"_GRY.dsv")
    grz = DSVFile(file_prefix+"_GRZ.dsv")
    end_dsv = time.time()
    print(f"Finished reading dsv files in {(end_dsv-start_dsv):.2f}s.")

    shapes = [rfd, rfp, grx, gry, grz]
    seq.set_shapes(shapes)

    # Read block structure
    print("Read block structure.")
    start_inf = time.time()
    read_dsv_inf(file_prefix+"_INF.dsv", seq, is_VE=True)
    end_inf = time.time()
    print(f"Finished reading block structure in {(end_inf-start_inf):.2f}s.")

    if plot:
        plot_seq([rfd,rfp,grx,gry,grz])

    return seq

def check_dsv(file_prefix1, file_prefix2):
    """ 
    Checks, if dsv files from converted sequence are the same as in the original sequence

    file_prefix1: dsv file prefix of the original sequence
    file_prefix2: dsv file prefix of the Pulseq sequence
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rfd1 = DSVFile(file_prefix1+"_RFD.dsv")
    grx1 = DSVFile(file_prefix1+"_GRX.dsv")

    # Pulseq sequence values are shifted by 20us, as there is an additional global
    # freq/phase event in the beginning
    shift_rf = -1* int(20 / rfd1.definitions.horidelta)
    shift_grad = -1* int(20 / grx1.definitions.horidelta)

    rfd1 = rfd1.values
    rfp1 = DSVFile(file_prefix1+"_RFP.dsv").values
    grx1 = grx1.values
    gry1 = DSVFile(file_prefix1+"_GRY.dsv").values
    grz1 = DSVFile(file_prefix1+"_GRZ.dsv").values

    rfd2 = DSVFile(file_prefix2+"_RFD.dsv").values
    rfp2 = DSVFile(file_prefix2+"_RFP.dsv").values
    grx2 = DSVFile(file_prefix2+"_GRX.dsv").values
    gry2 = DSVFile(file_prefix2+"_GRY.dsv").values
    grz2 = DSVFile(file_prefix2+"_GRZ.dsv").values

    rfd2 = np.roll(rfd2, shift_rf)[:len(rfd1)]
    rfp2 = np.roll(rfp2, shift_rf)[:len(rfp1)]
    grx2 = np.roll(grx2, shift_grad)[:len(grx1)]
    gry2 = np.roll(gry2, shift_grad)[:len(gry1)]
    grz2 = np.roll(grz2, shift_grad)[:len(grz1)]

    plt.figure()
    plt.subplot(511)
    plt.plot(rfd1-rfd2)
    plt.subplot(512)
    plt.plot(rfp1-rfp2)
    plt.subplot(513)
    plt.plot(grx1-grx2)
    plt.subplot(514)
    plt.plot(gry1-gry2)
    plt.subplot(515)
    plt.plot(grz1-grz2)
    