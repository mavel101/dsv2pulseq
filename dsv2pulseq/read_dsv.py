import time
import numpy as np
import logging
import os
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

    Pulseq sequence shape usually have two 20us long additional global
    freq/phase event in the beginning and end of the sequence. This can be
    compensated by the time_shift parameter.

    If provided, also the ADC shapes are compared.

    Parameters:
        file_prefix1: dsv file prefix of the original sequence
        file_prefix2: dsv file prefix of the Pulseq sequence
        time_shift: time shift in us for the Pulseq sequence

    Returns:
        seq1: Sequence object of the original sequence
        seq2: Sequence object of the Pulseq sequence
    """
    import matplotlib.pyplot as plt

    seq1 = read_dsv(file_prefix1)
    seq2 = read_dsv(file_prefix2)

    if os.path.exists(file_prefix1 + "_ADC.dsv") and os.path.exists(file_prefix2 + "_ADC.dsv"):
        adc1_dsv = DSVFile(file_prefix1 + "_ADC.dsv")
        adc2_dsv = DSVFile(file_prefix2 + "_ADC.dsv")
        adc_delta = adc1_dsv.definitions.horidelta
        seq1.adc = adc1_dsv.values
        seq2.adc = adc2_dsv.values
    else:
        seq1.adc = None
        seq2.adc = None

    if os.path.exists(file_prefix1 + "_NC1.dsv") and os.path.exists(file_prefix2 + "_NC1.dsv"):
        nc1_dsv = DSVFile(file_prefix1 + "_NC1.dsv")
        nc2_dsv = DSVFile(file_prefix2 + "_NC1.dsv")
        nc_delta = nc1_dsv.definitions.horidelta
        seq1.nc1 = nc1_dsv.values
        seq2.nc1 = nc2_dsv.values
    else:
        seq1.nc1 = None
        seq2.nc1 = None

    rfd1 = abs(seq1.rf)
    rfp1 = np.angle(seq1.rf)
    grx1 = seq1.gx
    gry1 = seq1.gy
    grz1 = seq1.gz
    adc1 = seq1.adc
    nco1 = seq1.nc1

    rfd2 = abs(seq2.rf)
    rfp2 = np.angle(seq2.rf)
    grx2 = seq2.gx
    gry2 = seq2.gy
    grz2 = seq2.gz
    adc2 = seq2.adc
    nco2 = seq2.nc1

    if time_shift > 0:
        shift_rf = int(time_shift / seq1.delta_rf)
        shift_grad = int(time_shift / seq1.delta_grad)
        rfd2 = rfd2[shift_rf:-shift_rf]
        rfp2 = rfp2[shift_rf:-shift_rf]
        grx2 = grx2[shift_grad:-shift_grad]
        gry2 = gry2[shift_grad:-shift_grad]
        grz2 = grz2[shift_grad:-shift_grad]
    if adc1 is not None and adc2 is not None:
        subplots = 6
        if time_shift > 0:
            shift_adc = int(time_shift / adc_delta)
            adc2 = adc2[shift_adc:-shift_adc]
    else:
        subplots = 5
    if nco1 is not None and nco2 is not None:
        subplots += 1
        if time_shift > 0:
            shift_nco = int(time_shift / nc_delta)
            nco2 = nco2[shift_nco:-shift_nco]
    else:
        nco1 = None
        nco2 = None

    len_rf = min(len(rfd1), len(rfd2))
    len_gx = min(len(grx1), len(grx2))
    len_gy = min(len(gry1), len(gry2))
    len_gz = min(len(grz1), len(grz2))
    plot_adc = plot_nco = False
    if adc1 is not None and adc2 is not None:
        len_adc = min(len(adc1), len(adc2))
        plot_adc = True
    if nco1 is not None and nco2 is not None:
        len_nco = min(len(nco1), len(nco2))
        plot_nco = True

    # Plot sequence 1
    n_subplots = subplots  # 5 or 6
    plt.figure(figsize=(10, 2 * n_subplots))
    plt.suptitle(f"Sequence 1: {file_prefix1}", fontsize=16)
    plt.subplot(n_subplots, 1, 1)
    plt.plot(rfd1)
    plt.title("RFD")
    plt.subplot(n_subplots, 1, 2)
    plt.plot(rfp1)
    plt.title("RFP")
    plt.subplot(n_subplots, 1, 3)
    plt.plot(grx1)
    plt.title("GX")
    plt.subplot(n_subplots, 1, 4)
    plt.plot(gry1)
    plt.title("GY")
    plt.subplot(n_subplots, 1, 5)
    plt.plot(grz1)
    plt.title("GZ")
    if plot_adc:
        plt.subplot(n_subplots, 1, 6)
        plt.plot(adc1)
        plt.title("ADC")
    if plot_nco:
        plt.subplot(n_subplots, 1, n_subplots)
        plt.plot(nco1)
        plt.title("NC1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot sequence 2
    plt.figure(figsize=(10, 2 * n_subplots))
    plt.suptitle(f"Sequence 2: {file_prefix2}", fontsize=16)
    plt.subplot(n_subplots, 1, 1)
    plt.plot(rfd2)
    plt.title("RFD")
    plt.subplot(n_subplots, 1, 2)
    plt.plot(rfp2)
    plt.title("RFP")
    plt.subplot(n_subplots, 1, 3)
    plt.plot(grx2)
    plt.title("GX")
    plt.subplot(n_subplots, 1, 4)
    plt.plot(gry2)
    plt.title("GY")
    plt.subplot(n_subplots, 1, 5)
    plt.plot(grz2)
    plt.title("GZ")
    if plot_adc:
        plt.subplot(n_subplots, 1, 6)
        plt.plot(adc2)
        plt.title("ADC")
    if plot_nco:
        plt.subplot(n_subplots, 1, n_subplots)
        plt.plot(nco2)
        plt.title("NC1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot difference
    plt.figure(figsize=(10, 2 * n_subplots))
    plt.suptitle(f"Difference: {file_prefix1} - {file_prefix2}", fontsize=16)
    plt.subplot(n_subplots, 1, 1)
    plt.plot(rfd1[:len_rf] - rfd2[:len_rf])
    plt.title("RFD")
    plt.subplot(n_subplots, 1, 2)
    plt.plot(rfp1[:len_rf] - rfp2[:len_rf])
    plt.title("RFP")
    plt.subplot(n_subplots, 1, 3)
    plt.plot(grx1[:len_gx] - grx2[:len_gx])
    plt.title("GX")
    plt.subplot(n_subplots, 1, 4)
    plt.plot(gry1[:len_gy] - gry2[:len_gy])
    plt.title("GY")
    plt.subplot(n_subplots, 1, 5)
    plt.plot(grz1[:len_gz] - grz2[:len_gz])
    plt.title("GZ")
    if plot_adc:
        plt.subplot(n_subplots, 1, 6)
        plt.plot(adc1[:len_adc] - adc2[:len_adc])
        plt.title("ADC")
    if plot_nco:
        plt.subplot(n_subplots, 1, n_subplots)
        plt.plot(nco1[:len_nco] - nco2[:len_nco])
        plt.title("NC1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

    return seq1, seq2
    