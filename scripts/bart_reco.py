#%%

import os
from matplotlib import pyplot as plt
import numpy as np
from bart import bart
import twixtools
import nibabel as nib

raw_path = ''
raw_file = ''
nifti_ref = ''

#%% read data

cc_cha = 8 # number of compressed coils

raw = os.path.join(raw_path, raw_file)
raw_data = twixtools.read_twix(raw, keep_syncdata=True, keep_acqend=True)[-1]
raw_mapped = twixtools.map_twix(raw_data)

# %% reshape data

noise = raw_mapped['noise']
noise_data = np.squeeze(noise[:])
noise_data_reshaped = np.moveaxis(noise_data, -1, 0)[:,np.newaxis, np.newaxis]

refscan = raw_mapped['refscan']
refscan.flags['remove_os'] = True
refdata = np.squeeze(refscan[:])
refdata_reshaped = np.moveaxis(refdata, -1, 0)

imgscan = raw_mapped['image']
imgscan.flags['remove_os'] = True
imgscan.flags['zf_missing_lines'] = True
imgdata = np.squeeze(imgscan[:])
imgdata_reshaped = np.moveaxis(imgdata, -1, 0)

#%% noise prewhitening 

refdata_white = bart(1, 'whiten', refdata_reshaped, noise_data_reshaped)
imgdata_white = bart(1, 'whiten', imgdata_reshaped, noise_data_reshaped)

#%% coil compression

ccmat = bart(1, f'cc -S -M', refdata_white)
refdata_cc = bart(1, f'ccapply -S -p {cc_cha}', refdata_white, ccmat)
imgdata_cc = bart(1, f'ccapply -S -p {cc_cha}', imgdata_white, ccmat)

#%% sensitivity maps

ecalone = bart(1, 'ecalib -1 -g -I', refdata_cc)
ecaltwo_str = f'ecaltwo -g -m 1 {imgdata_cc.shape[0]} {imgdata_cc.shape[1]} {imgdata_cc.shape[2]}'
sens = bart(1, ecaltwo_str, ecalone)

# refdata_resized = bart(1, f'resize -c 0 {imgdata_cc.shape[0]} 1 {imgdata_cc.shape[1]} 2 {imgdata_cc.shape[2]}', refdata_cc)
# sens = bart(1, 'caldir 32', refdata_resized)

#%%

pics_str = 'pics -S -g -e -l1 -r 0.1 -i 200'
img = bart(1, pics_str, imgdata_cc, sens)

# %%
plot_slc = 110
plt.figure()
plt.imshow(np.abs(img[plot_slc, :, :]), cmap='gray')

# %%
refimg = nib.load(nifti_ref)
img_save = np.transpose(img, [1,2,0])[::-1,:,::-1]
nifti = nib.Nifti1Image(np.abs(img_save), refimg.affine, refimg.header)
outfile = os.path.join(raw_path, os.path.splitext(raw_file)[0] + '_recoBART.nii.gz')
nib.save(nifti, outfile)
# %%
