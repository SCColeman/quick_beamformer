#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coreg functional and anatomical.

@author: sebastiancoleman
"""

import mne
import os.path as op
import os
from glob import glob
from matplotlib import pyplot as plt
from nilearn import image
import pandas as pd
import nibabel as nib

#%% paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'

#%% get list of subjects

dir_list = sorted(glob(op.join(data_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]

#%% mark fiducials on FreeSurfer surface

s = 83
subject = subjects[s]
fname = glob(op.join(data_path, subject, '*-raw.fif'))[0]
raw = mne.io.Raw(fname, preload=False, verbose=False)
mne.gui.coregistration(subjects_dir=subjects_dir, subject=subject, 
                               inst=fname, fullscreen=False, width=1600, height=1200)

#%% run this section to verify fid position if not visible

img = nib.load(op.join(subjects_dir, subject, 'mri', 'T1.mgz'))
img.orthoview()

#%% Only run this section if the subject has correctly marked fiducials

# make derivatives folder for subject at this stage
if not op.exists(op.join(deriv_path, subject)):
    os.makedirs(op.join(deriv_path, subject))

# coreg using HPI and anatomical landmarks
coreg = mne.coreg.Coregistration(raw.info, subject, subjects_dir)
coreg.fit_fiducials()

# save out transform
trans = coreg.trans
trans_fname = op.join(deriv_path, subject, subject + '-trans.fif')
mne.write_trans(trans_fname, trans, overwrite=True)

# alignment fig
fig = mne.viz.create_3d_figure(size=(600,600), bgcolor='white', show=True)
mne.viz.plot_alignment(raw.info, coreg.trans, subject, subjects_dir, fig=fig, coord_frame='mri')
fig.plotter.view_xz(negative=True)
fig.plotter.zoom_camera(1.7)
screenshot1 = fig.plotter.screenshot()
fig.plotter.view_yz()
fig.plotter.zoom_camera(1.5)
screenshot2 = fig.plotter.screenshot()
fig.plotter.close()
fig, ax = plt.subplots(1,2, figsize=(7,4))
ax[0].imshow(screenshot1)
ax[0].axis('off')
ax[1].imshow(screenshot2)
ax[1].axis('off')
plt.tight_layout()
fig.savefig(op.join(deriv_path, subject, subject + '_coreg.png'))
plt.close()


