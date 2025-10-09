#%%

import os
from matplotlib import pyplot as plt
import numpy as np
import MRzeroCore as mr0
import twixtools
import torch

seq_path = '../test/test_data/'
seq_file = 'dzne_ep3d.seq'

brain_phantom_res = [64, 64, 1]
include_sens = False # load sensitivity maps from sensmaps.h5, generated with JEMRIS
plot_phantom = True
plot_slc = 0

insert_raw = False # insert simulation data into raw data file
scale_fac = 1e-4 # data scale factor for insertion
raw_path = ''
raw_file = ''
out_path = ''
out_file = ''

#%%

print('load phantom')
obj_p = mr0.VoxelGridPhantom.load('phantom_data/subject05_3T.npz')
obj_p.B0[:] = 0

# Load sensitivity maps
# 32 Biot-Savart coils from JEMRIS with extent 256, array radius 256, 3D
if include_sens:
    from bart import bart

    refdata = np.load('refdata_coils.npy')
    sens_caldir = bart(1, 'caldir 32', refdata).T
    sens = bart(1, f'resize -c 1 {18} 2 {18} 3 {48}', sens_caldir)

    # import h5py
    # sensmaps = h5py.File('sensmaps.h5', 'r')
    # maps = sensmaps['maps']
    # coil_keys = sorted(maps['magnitude'].keys()) 
    # maps_mag = np.stack([maps['magnitude'][k][()] for k in coil_keys])
    # maps_phase = np.stack([maps['phase'][k][()] for k in coil_keys])
    # sens = maps_mag * np.exp(1j * maps_phase)

    obj_p.coil_sens = torch.tensor(sens, dtype=torch.complex64)

# interpolate and plot
obj_p = obj_p.interpolate(*brain_phantom_res)
if plot_phantom: obj_p.plot(plot_slice=plot_slc)

# build
phantom = obj_p.build().cuda()

#%%
print('load sequence')
seq = os.path.join(seq_path, seq_file)
seq0 = mr0.Sequence.import_file(seq).cuda()

#%%
print('simulate sequence')
graph = mr0.compute_graph(seq0, phantom, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, phantom, print_progress=True)

#%%
signal_numpy = signal.cpu().numpy()
signal_numpy_scaled = signal_numpy * scale_fac

plt.figure()
plt.title('Signal magnitude')
plt.plot(np.abs(signal_numpy_scaled))

plt.figure()
plt.title('Signal real and imaginary parts')
plt.plot(signal_numpy_scaled.real, label='real')
plt.plot(signal_numpy_scaled.imag, label='imag')

# %%

if insert_raw:
    print("insert data into raw data file")
    raw = os.path.join(raw_path, raw_file)
    raw_data = twixtools.read_twix(raw, keep_syncdata=True, keep_acqend=True)

    nc = raw_data[-1]['mdb'][0].data.shape[0]
    if nc != signal_numpy_scaled.shape[1]:
        print(f"Number of coils {nc} in raw data file does not match simulated coils {signal_numpy_scaled.shape[1]}.")

    sample_ix = 0
    for k, mdb in enumerate(raw_data[-1]['mdb']):
        if not mdb.is_flag_set('ACQEND') and not mdb.is_flag_set('SYNCDATA') and not mdb.is_flag_set('NOISEADJSCAN'):
            mdb = mdb.convert_to_local()
            mdb_samples = mdb.data.shape[1]
            mdb.data = signal_numpy_scaled[sample_ix:sample_ix + mdb_samples,:].T
            raw_data[-1]['mdb'][k] = mdb
            sample_ix += mdb_samples

    if sample_ix != signal_numpy_scaled.shape[0]:
        print(f"MDB data length {sample_ix} does not match simulated signal length {signal_numpy_scaled.shape[0]}.")

    print(f"Writing modified raw data to {out_file}.")
    out = os.path.join(out_path, out_file)
    twixtools.write_twix(raw_data, out, version_is_ve=True)
