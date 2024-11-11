# -*- coding: utf-8 -*-
"""
Finish FreeSurfer steps.

@author: sebastiancoleman.
"""

import mne
import os.path as op
from mne.bem import make_watershed_bem, make_scalp_surfaces
from glob import glob

#%% paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'

dir_list = sorted(glob(op.join(subjects_dir, '*')))
subjects = [op.basename(d) for d in dir_list]

#%% finish FS steps

for subject in subjects:
    make_scalp_surfaces(subject, subjects_dir, overwrite=True)
    make_watershed_bem(subject, subjects_dir, overwrite=True)