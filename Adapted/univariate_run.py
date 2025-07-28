# Original scirpt was wirtten by Alexander Enge [https://github.com/SkeideLab/SLANG-analysis/tree/93890d2cda9aac1ded61d41ae1ed5bd27f4d4bc0/scripts]
# Adapted by Garam Jeong to allow flexibility of running the code partially without running through from first level to the second level at once.
# Script adapted for COCOA project to study longitudinal development of children learning language.

# Run part1 GLM for each subject without paralell

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from bids.layout import BIDSLayout, BIDSLayoutIndexer
from nilearn.interfaces.bids import save_glm_to_bids

import new_part1_glm as pt1
import new_part2_ACF as pt2
import part3_betas as pt3

### Set the global variables

# Input parameters: File paths
BIDS_DIR = Path('/data/pt_02825/MPCDF/BIDS')
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate'

# Input parameters: Inclusion/exclusiong criteria
FD_THRESHOLD = 0.7
DF_QUERY = 'perc_outliers <= 0.25 & n_sessions >= 2' #to sort out which subjects to include 

# Input parameters: First-level GLM
TASK = 'Literacy'
SPACE = 'MNI152NLin6Asym'
BLOCKWISE = False
SMOOTHING_FWHM = 5.0
HRF_MODEL = 'glover + derivative + dispersion'
SAVE_RESIDUALS = True

# Define contrast specs in a clean, readable format
CONTRASTS_SPEC = {
    'unimodal_audios': {'unimodal_audios': 1},
    'unimodal_images': {'unimodal_images': 1},
    'congruent': {'congruent': 1},
    'incongruent': {'incongruent': 1},
    
    'congruent-audio': {'congruent': 1, 'unimodal_audios': -1},
    'congruent-visual': {'congruent': 1, 'unimodal_images': -1},
    'congruent-incongruent': {'congruent': 1, 'incongruent': -1},
    'incongruent-congruent': {'congruent': -1, 'incongruent': 1},
    'congruent-audio-visual': {'congruent': 1, 'unimodal_audios': -0.5, 'unimodal_images': -0.5}
}
# Input parameters: Cluster correction
CLUSTSIM_SIDEDNESS = '2-sided'
CLUSTSIM_NN = 'NN1'  # Must be 'NN1' for Nilearn's `get_clusters_table`
CLUSTSIM_VOXEL_THRESHOLD = 0.001
CLUSTSIM_CLUSTER_THRESHOLD = 0.05
CLUSTSIM_ITER = 10000

# Input parameters: Group-level linear mixed models
FORMULA = 'beta ~ time + (time | subject)'

###Starting main bit here

# Load BIDS structure
layout = BIDSLayout(BIDS_DIR, derivatives=FMRIPREP_DIR,
                    database_path=PYBIDS_DIR, reset_database=True)

print('Layout loaded.')

# Get subject and session list
subjects_sessions = pt1.get_subjects_sessions(layout, TASK, SPACE)
print('\n Got a list of (Sub, Ses).')

# First level GLM, each subject and session without parallel job

all_glm              = []
all_mask_imgs        = []
all_mask_paths       = []
all_precs_non_steady = []
all_precs_outliers   = []
all_residuals_file   = []
all_sub_ses_list     = []

for subject, session in subjects_sessions: 
    # print which (sub,ses) is procssed to check the order of pairs
    all_sub_ses_list.append((subject,session))
    
    try:
        glms, mask_imgs, mask_paths,precs_non_steady, precs_outlier, residuals_files = pt1.run_glm(
            layout, BIDS_DIR, FMRIPREP_DIR, PYBIDS_DIR, TASK, SPACE, BLOCKWISE,
            FD_THRESHOLD, HRF_MODEL, SMOOTHING_FWHM, UNIVARIATE_DIR,
            SAVE_RESIDUALS, subject=subject, session=session)
        
        all_glm.append(glms)
        all_mask_imgs.append(mask_imgs)
        all_mask_paths.append(mask_paths)
        all_precs_non_steady.append(precs_non_steady)
        all_precs_outliers.append(precs_outlier)
        all_residuals_file.append(residuals_files)
        
        print(f"\n First level model of subject {subject} session {session} is done.\n")

    except Exception as e:
        print(f"Error processing subject {subject}, session {session}: {e}")
        continue

# Load metadata for mapping GLMs
meta_df, good_ixs = pt1.load_meta_df(layout, TASK, all_precs_non_steady,
                                  all_precs_outliers, DF_QUERY) #  save meta_df as csv in derivatives folder
meta_df = meta_df.set_index(['subject','session']) # index: (sub,ses)
meta_df = meta_df.loc[good_ixs]

### Part 2 Compute ACF parameters for multiple comparison correction
# ACF parameters are computed per each run of (sub, ses) 

# Try Call the saved acf and cluster size 
ls_ixs = []
for (sub,ses) in all_sub_ses_list:
    if (sub,ses) in good_ixs:
        ls_ixs.append(all_sub_ses_list.index((sub,ses)))

residuals_files = [all_residuals_file[ix] for ix in ls_ixs]
mask_paths      = [all_mask_paths[ix] for ix in ls_ixs]
lst_res_mask    = [(str(res), mask)
                    for res_list, mask_list in zip(residuals_files, mask_paths)
                    for res, mask in zip(res_list, mask_list)]

# save the list of residuals and mask files to tsv to call later if it is required.
acf_df = pd.DataFrame()
acf_df['residuals_files'] = residuals_files
acf_df['mask_paths'] = mask_paths
acf_df_path = UNIVARIATE_DIR / 'acf_list.csv'
acf_df.to_csv(acf_df_path, index=False)

acfs = [pt2.compute_acf(residuals_file, mask_path)
        for (residuals_file, mask_path) in lst_res_mask]

# mean of afcs from each run of (sub,ses)
acf = np.nanmean(acfs, axis=0) # ignore nan value
print(f'ACF parameters (a, b, c): {acf} \n')

# second method create group mask without using nilearn but numpy by stacking all mask images
good_mask_paths = []
for (res, mask) in lst_res_mask:
     good_mask_paths.append(mask)
     
output_path = UNIVARIATE_DIR / f'task-{TASK}_space-{SPACE}_desc-brain_mask.nii.gz'

# create and save the mask image
pt1.make_group_mask(good_mask_paths, output_path, threshold_ratio=0.5)

mask_file   = output_path
mask_img    = nib.load(mask_file)
mask        = mask_img.get_fdata().astype(np.int32)
voxel_ixs   = np.transpose(mask.nonzero())

# Compute FWE-corrected cluster threshold (This is a number, therefore don't save it)
cluster_threshold = \
    pt2.compute_cluster_threshold(acf, mask_file, CLUSTSIM_SIDEDNESS,
                              CLUSTSIM_NN, CLUSTSIM_VOXEL_THRESHOLD,
                              CLUSTSIM_CLUSTER_THRESHOLD, CLUSTSIM_ITER)
print(f'Cluster threshold: {cluster_threshold:.1f} voxels \n')

### Part 3 Second level model - mixed model
# Run mixed model analysis of each voxel across subject and session
    
for contrast_id in CONTRASTS_SPEC:
    print(f'Processing contrast: {contrast_id}')

    # load saved effect size map of good (sub,ses)
    good_glms = []
    bad_glms  = []
    for (subject, session) in all_sub_ses_list:
        if (subject, session) in good_ixs:
            fpath = UNIVARIATE_DIR / f"sub-{subject}" / f"ses-{session}" / "func" / \
                    f"sub-{subject}_ses-{session}_task-{TASK}_FirstLevel_{contrast_id}_effect_size_map.nii.gz"
            if fpath.exists():
                good_glms.append(fpath)
            else:
                print(f"Missing first level effect size for subject {subject}, session {session}")
                bad_glms.append((subject,session))
    
    betas = [nib.load(img).get_fdata() for img in good_glms] #img is a list of effect size object 
    betas = np.array(betas).squeeze()  
    print(f"\n beta images are loaded and squeezed for {betas.shape[0]} (subject, session)")
    
    # Fit voxelwise mixed models in Julia
    print(f"\n Fitting mixed models for contrast: {contrast_id}")
    betas_per_voxel = [betas[:, x, y, z] for x, y, z in voxel_ixs]
    model_df = meta_df.copy()
    model_df = model_df[~model_df.index.isin(bad_glms)]    # remove sub,ses if contrast-effect_map doesn't exist
    model_df = model_df.reset_index()[['subject', 'session', 'perc_non_steady',
                                   'perc_outliers', 'time']]    # reset sub,ses as column names
    model_dfs = [model_df.assign(beta=vals) for vals in betas_per_voxel]

    res = pt3.fit_mixed_models(FORMULA, model_dfs)
    bs, zs = zip(*res)
    b0, b1 = np.array(bs).T
    z0, z1 = np.array(zs).T

    # Save results
    for array, suffix in zip([b0, b1, z0, z1], ['b0', 'b1', 'z0', 'z1']):
        img, file = pt3.save_array_to_nifti(array, mask, voxel_ixs,
                                        UNIVARIATE_DIR, TASK, SPACE,
                                        contrast_id, suffix)
        print(f" Saved: {file}")

        if suffix.startswith('z'):
            pt3.save_clusters(img, CLUSTSIM_VOXEL_THRESHOLD, cluster_threshold,
                          UNIVARIATE_DIR, TASK, SPACE, contrast_id, suffix)

