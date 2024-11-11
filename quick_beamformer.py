#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast and simple MEG source reconstruction pipeline. Convert MEG data from 
sensor space to atlas space.

Note this script treats the data as if it were resting state, i.e., no epoching
prior to source reconstruction. This is highly recommended to allow for 
artifact-free filtering later on.

Script assumes you already have a subjects_dir with the required FreeSurfer
outputs, and you already have a saved transform (-trans.fif). These can be 
created by running FreeSurfers recon-all, followed by finish_freesurfer.py, 
followed by coreg.py.

@author: Sebastian C. Coleman, sebastian.coleman@sickkids.ca
"""

import mne
from mne_connectivity import symmetric_orth
import os.path as op
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from nilearn import plotting, datasets, image
import nibabel as nib

#%% functions

def make_atlas_nifti(atlas_img, values):

    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()

    # empty image
    atlas_new = np.zeros(np.shape(atlas_data))

    # place values in each parcel region
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]

    # make image from new atlas data
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    return img_interp

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'

#%% load atlas (AAL90 - remove cerebellum)

atlas = datasets.fetch_atlas_aal()
atlas_img = image.load_img(atlas.maps)
atlas_data = atlas_img.get_fdata()
indices = atlas.indices
labels = atlas.labels
atlas_data[atlas_data >= int(indices[90])] = 0
aal90 = image.new_img_like(atlas_img, atlas_data)
aal90_labels = labels[:90]
coords = plotting.find_parcellation_cut_coords(aal90)

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]

#%% quick beamformer

for s, subject in enumerate(subjects):
    
    ### PREPROCESSING ###
    
    # load raw unprocessed data
    fname = glob(op.join(data_path, subject, '*-raw.fif'))[0]
    raw = mne.io.Raw(fname, preload=True)
    
    # basic preprocessing
    raw.apply_gradient_compensation(3)
    raw.pick('mag')
    raw.resample(250)
    raw.filter(3,45)  # this range mostly eliminates the need for ICA
    
    # remove bad channels based on z-scored variance
    chan_var = np.var(raw.get_data(), axis=1)
    outliers = zscore(chan_var) > 3
    bad_chan = [raw.ch_names[ch] for ch in range(len(raw.ch_names)) if outliers[ch]]
    raw.drop_channels(bad_chan)
    
    # annotate bad segments (treat as resting state)
    data = raw.get_data()
    segment_len = int(1 * raw.info['sfreq'])
    variances = np.array([np.mean(np.var(data[:, i:i+segment_len], axis=1)) for i in np.arange(0, data.shape[1]-segment_len+1, segment_len)])
    variances = zscore(variances)
    outliers = variances > 3
    annotations = raw.annotations
    for i, ind in enumerate(np.arange(0, data.shape[1]-segment_len, segment_len)):
        if outliers[i]:
            onset = raw.times[ind]
            duration = segment_len * (1/raw.info['sfreq'])
            description = 'BAD_segment'
            annotations += mne.Annotations(onset, duration, description, orig_time=annotations.orig_time)
    raw.set_annotations(annotations, verbose=False)
    
    # plot preprocessed sensor-level PSD
    fig = raw.compute_psd(fmin=3, fmax=45, reject_by_annotation=True).plot()
    fig.savefig(op.join(deriv_path, subject, subject + '_sensor_psd.png'))
    plt.close()
    
    # plot preprocessed sensor-level power maps
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    bands = {'Theta (4-8 Hz)': (4, 8), 'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30)}
    raw.compute_psd(fmin=3, fmax=45, reject_by_annotation=True).plot_topomap(bands, axes=ax, normalize=True)
    plt.tight_layout()
    fig.savefig(op.join(deriv_path, subject, subject + '_sensor_power.png'))
    plt.close()
    
    # save preprocessed data
    raw.save(op.join(deriv_path, subject, subject + '_preproc-raw.fif'), overwrite=True)
    
    ### FORWARD MODEL ###
    
    # single-shell conduction model
    conductivity = (0.3,)
    model = mne.make_bem_model(
            subject=subject, ico=4,
            conductivity=conductivity,
            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    # get mri-->MNI transform and apply inverse to atlas
    mri_mni_t = mne.read_talxfm(subject, subjects_dir=subjects_dir)['trans']
    mni_mri_t = np.linalg.inv(mri_mni_t)
    centroids_mri = mne.transforms.apply_trans(mni_mri_t, coords / 1000) # in m
    
    # create AAL source space
    rr = centroids_mri # positions
    nn = np.zeros((rr.shape[0], 3)) # normals
    nn[:,-1] = 1.
    src = mne.setup_volume_source_space(
        subject,
        pos={'rr': rr, 'nn': nn},
        subjects_dir=subjects_dir,
        verbose=True,
    )
    
    # forward solution
    trans_fname = op.join(deriv_path, subject, subject + '-trans.fif')
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans_fname,
        src=src,
        bem=bem,
        meg=True,
        eeg=False
        )
    mne.write_forward_solution(op.join(deriv_path, subject, subject + '-fwd.fif'), fwd, overwrite=True)
    
    ### SOURCE RECON ###
    
    # calculate covariance of all data and plot
    cov = mne.compute_raw_covariance(raw, reject_by_annotation=True)
    cov_data = cov.data
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cov_data)
    ax.set_xlabel('Sensor')
    ax.set_ylabel('Sensor')
    ax.set_title('Data Covariance')
    plt.tight_layout()
    fig.savefig(op.join(deriv_path, subject, subject + '_data_cov.png'))
    plt.close()

    # construct beamformer
    filters = mne.beamformer.make_lcmv(
            raw.info,
            fwd,
            cov,
            reg=0.05,
            noise_cov=None,
            pick_ori='max-power',
            rank=None,
            reduce_rank=True,
            verbose=False,
            )

    # apply beamformer
    stc =  mne.beamformer.apply_lcmv_raw(raw, filters, verbose=False)
    source_data = zscore(stc.data, 1)

    # make source raw
    info = mne.create_info(aal90_labels, raw.info['sfreq'], 'misc', verbose=False)
    source_raw = mne.io.RawArray(source_data, info, verbose=False)
    source_raw.set_meas_date(raw.info['meas_date'])
    source_raw.set_annotations(raw.annotations, verbose=False)
    source_raw.save(op.join(deriv_path, subject, subject + '_source-raw.fif'), overwrite=True)

    # orthogonalise
    source_data_orth = zscore(symmetric_orth(source_data), 1)
    source_raw_orth = mne.io.RawArray(source_data_orth, source_raw.info)
    source_raw_orth.set_meas_date(raw.info['meas_date'])
    source_raw_orth.set_annotations(raw.annotations, verbose=False)
    source_raw_orth.save(op.join(deriv_path, subject, subject + '_source_orth-raw.fif'), overwrite=True)
    
    # compute source level PSD
    psd = source_raw.compute_psd(method='welch', n_fft=500, fmin=3, fmax=30, picks='all')
    freqs = psd.freqs
    power = psd.data
    power = (power.T / np.sum(power,1)).T
    
    # plot PSD
    fig, ax = plt.subplots(figsize=(6,3))
    values = np.mean(power,0)
    err = np.std(power,0) / np.sqrt(power.shape[0])
    ax.plot(freqs, values, color='black')
    ax.fill_between(freqs, values-err, values+err, color='black', alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Source-Level PSD')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(op.join(deriv_path, subject, subject + '_source_psd.png'))
    plt.close()
    
    # plot source-level power maps
    fmin = (4, 8, 13)
    fmax = (8, 13, 30)
    band_names = ['Theta', 'Alpha', 'Beta']
    for freq in range(len(fmin)):
        freq_range = (freqs > fmin[freq]) * (freqs < fmax[freq])
        power_map = np.mean(power[:,freq_range], -1)
        threshold = np.percentile(power_map, 80)
        power_img = make_atlas_nifti(aal90, power_map)
        fig = plotting.plot_stat_map(power_img, threshold=threshold, cmap='Reds', title=band_names[freq])
        fig.savefig(op.join(deriv_path, subject, subject + '_source_' + band_names[freq] + '.png'))
        plt.close()