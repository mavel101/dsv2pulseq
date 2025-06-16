import time
import numpy as np
import logging
from dsv2pulseq.sequence import Sequence
from dsv2pulseq.read_dsv_samples import DSVFile
from dsv2pulseq.read_dsv_inf import read_dsv_inf

logging.basicConfig(level=logging.INFO)

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

    Parameters:
        file_prefix: dsv file prefix, e.g. 'gre'
        ref_volt: reference voltage of the simulation in V
        plot: if True, plots the sequence shapes
    Returns:
        seq: Sequence object containing the shapes and blocks
    """
    seq = Sequence(ref_volt)

    # read event shapes
    logging.info(f"Read RF and gradient shapes of {file_prefix} sequence.")
    start_dsv = time.time()
    rfd = DSVFile(file_prefix+"_RFD.dsv")
    rfp = DSVFile(file_prefix+"_RFP.dsv")
    grx = DSVFile(file_prefix+"_GRX.dsv")
    gry = DSVFile(file_prefix+"_GRY.dsv")
    grz = DSVFile(file_prefix+"_GRZ.dsv")
    end_dsv = time.time()
    logging.info(f"Finished reading dsv files in {(end_dsv-start_dsv):.2f}s.")

    shapes = [rfd, rfp, grx, gry, grz]
    seq.set_shapes(shapes)

    # Read block structure
    logging.info("Read block structure.")
    start_inf = time.time()
    read_dsv_inf(file_prefix+"_INF.dsv", seq)
    end_inf = time.time()
    logging.info(f"Finished reading block structure in {(end_inf-start_inf):.2f}s.")

    if plot:
        plot_seq([rfd,rfp,grx,gry,grz])

    return seq

def check_dsv(file_prefix1, file_prefix2, time_shift=20):
    """ 
    Checks, if dsv files from converted sequence are the same as in the original sequence

    Pulseq sequence shape values are usually shifted by 20us, as there is an additional global
    freq/phase event in the beginning of the sequence.
    
    Parameters:
        file_prefix1: dsv file prefix of the original sequence
        file_prefix2: dsv file prefix of the Pulseq sequence
        time_shift: time shift in us, which is applied to the shapes of file_prefix2

    Returns:
        seq1: Sequence object of the original sequence
        seq2: Sequence object of the Pulseq sequence
    """
    import matplotlib.pyplot as plt

    seq1 = read_dsv(file_prefix1)
    seq2 = read_dsv(file_prefix2)

    len_rf = min(len(seq1.rf), len(seq2.rf))
    len_gx = min(len(seq1.gx), len(seq2.gx))
    len_gy = min(len(seq1.gy), len(seq2.gy))
    len_gz = min(len(seq1.gz), len(seq2.gz))

    rfd1 = abs(seq1.rf)
    rfp1 = np.angle(seq1.rf)
    grx1 = seq1.gx
    gry1 = seq1.gy
    grz1 = seq1.gz

    rfd2 = abs(seq2.rf)
    rfp2 = np.angle(seq2.rf)
    grx2 = seq2.gx
    gry2 = seq2.gy
    grz2 = seq2.gz

    shift_rf = -1* int(time_shift / seq1.delta_rf)
    shift_grad = -1* int(time_shift / seq1.delta_grad)

    rfd2 = np.roll(rfd2, shift_rf)
    rfp2 = np.roll(rfp2, shift_rf)
    grx2 = np.roll(grx2, shift_grad)
    gry2 = np.roll(gry2, shift_grad)
    grz2 = np.roll(grz2, shift_grad)

    # Plot sequence 1
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Sequence 1: {file_prefix1}", fontsize=16)
    plt.subplot(511)
    plt.plot(rfd1)
    plt.title("RFD")
    plt.subplot(512)
    plt.plot(rfp1)
    plt.title("RFP")
    plt.subplot(513)
    plt.plot(grx1)
    plt.title("GX")
    plt.subplot(514)
    plt.plot(gry1)
    plt.title("GY")
    plt.subplot(515)
    plt.plot(grz1)
    plt.title("GZ")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot sequence 2
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Sequence 2: {file_prefix2}", fontsize=16)
    plt.subplot(511)
    plt.plot(rfd2)
    plt.title("RFD")
    plt.subplot(512)
    plt.plot(rfp2)
    plt.title("RFP")
    plt.subplot(513)
    plt.plot(grx2)
    plt.title("GX")
    plt.subplot(514)
    plt.plot(gry2)
    plt.title("GY")
    plt.subplot(515)
    plt.plot(grz2)
    plt.title("GZ")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot difference
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Difference: {file_prefix1} - {file_prefix2}", fontsize=16)
    plt.subplot(511)
    plt.plot(rfd1[:len_rf] - rfd2[:len_rf])
    plt.title("RFD")
    plt.subplot(512)
    plt.plot(rfp1[:len_rf] - rfp2[:len_rf])
    plt.title("RFP")
    plt.subplot(513)
    plt.plot(grx1[:len_gx] - grx2[:len_gx])
    plt.title("GX")
    plt.subplot(514)
    plt.plot(gry1[:len_gy] - gry2[:len_gy])
    plt.title("GY")
    plt.subplot(515)
    plt.plot(grz1[:len_gz] - grz2[:len_gz])
    plt.title("GZ")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

    return seq1, seq2
    