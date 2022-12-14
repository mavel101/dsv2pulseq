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

def read_dsv(file_prefix, ref_volt, plot=False):
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
