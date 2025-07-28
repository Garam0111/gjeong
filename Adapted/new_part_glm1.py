# Script was adjusted by Garam Jeong and Johanna Finemann
# Original script is from Alexsander Enge, univariate.py [https://github.com/SkeideLab/SLANG-analysis/tree/93890d2cda9aac1ded61d41ae1ed5bd27f4d4bc0/scripts]

import os

from pathlib import Path
import numpy as np
import pandas as pd

from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)
from nilearn.image import binarize_img, load_img, math_img, mean_img
from nilearn.reporting import make_glm_report
from nilearn.interfaces.bids import save_glm_to_bids

# Input parameters: File paths
BIDS_DIR = Path('/data/pt_02825/MPCDF/BIDS')
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate'
GLM_DIR = DERIVATIVES_DIR / 'univariate'/'FirstLevel'

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


# functions

# =============================================================================
# load event files of a given (sub,ses)
# =============================================================================
def load_events(layout, subject, session, task):
    """Loads events files for a given subject, session, and task.
    Return: list; a list of data frames from events.tsv of each run"""
    
    events_files = layout.get(subject=subject, session=session, task=task,
                              suffix='events', extension='tsv')
    print(f"Subject={subject}, Session={session}, Task={task}")
    print(f"Events files found: {events_files}")

    if not events_files:
        raise ValueError(f"No events files found for subject={subject}, session={session}, task={task}.")
    
    # Combine all event files into a single DataFrame
    events_dfs = []
    for events_file in events_files:
        df = pd.read_csv(events_file, sep='\t')
        # Add metadata for which run this file corresponds to
        r = events_file.entities.get("run", "unknown")
        df["run"] = r
        events_dfs.append(df)

    return events_dfs

# =============================================================================
# load confounds of a given (sub,ses) - dim of axis 1 is not matched
# =============================================================================
def get_confounds(layout, subject, session, task, fd_threshold=0.5):
    """
    Loads and combines confounds for a given subject, session, and task of all runs.
    Return: list, list, list, int, int;
            a list of confounds, a list of percentage of non steady outlier,
            a list of percent of outlier from each run, min of acomcor len
            max num non_steady_state outlier, mas num of fd_outlier
    """
    # Fetch all confounds files for the given subject, session, and task
    confounds_files = layout.get(subject=subject, session=session, task=task,
                                 desc='confounds', suffix='timeseries',
                                 extension='tsv')
    print(f"Confounds files found for subject={subject}, session={session}, task={task}: {len(confounds_files)} runs.")
    
    # prepare list of outputs for all run for a (subject, session)
    confounds_lst       = []
    perc_non_steady_lst = []
    perc_outliers_lst   = []

    non_steady_lst = []
    outlier_lst    = []
    compcor_lst     = []
    
    for r in range(len(confounds_files)):
        confounds       = pd.read_csv(confounds_files[r], sep='\t')
    
        hmp_cols        = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    
        compcor_cols = [col for col in confounds if col.startswith('a_comp_cor_')]
        compcor_cols = compcor_cols[:6]
        compcor_lst.append(len(compcor_cols))
    
        cosine_cols     = [col for col in confounds if col.startswith('cosine')]
    
        non_steady_cols = [col for col in confounds
                           if col.startswith('non_steady_state_outlier')]
        n_non_steady    = len(non_steady_cols)
        n_volumes       = len(confounds)
        perc_non_steady = n_non_steady / n_volumes
        
        perc_non_steady_lst.append(perc_non_steady)
        non_steady_lst.append(len(non_steady_cols))
        
        print(f'Found {n_non_steady} non-steady-state volumes ' +
              f'({perc_non_steady * 100:.1f}%) for subject {subject}, session {session}, run {r+1}"')
    
        # Add outlier regressors based on FD threshold
        fd = confounds['framewise_displacement']
        outlier_ixs    = np.where(fd > fd_threshold)[0]
        outliers      = np.zeros((len(fd), len(outlier_ixs)))
        outliers[outlier_ixs, np.arange(len(outlier_ixs))] = 1
        outlier_cols  = [f'fd_outlier{i}' for i in range(len(outlier_ixs))]
        outliers      = pd.DataFrame(outliers, columns=outlier_cols)
        confounds     = pd.concat([confounds, outliers], axis=1)
        n_outliers    = len(outlier_ixs)
        perc_outliers = n_outliers / n_volumes 
        
        perc_outliers_lst.append(perc_outliers)
        outlier_lst.append(len(outlier_cols))
        
        print(f"Outlier columns added: {len(outlier_cols)}")
        print(f'\n Found {n_outliers} outlier volumes ' +
              f'({perc_outliers * 100:.1f}%) for subject {subject}, session {session}, run {r+1}')
    
        # Collect all required columns
        cols_to_use = hmp_cols + compcor_cols + cosine_cols + non_steady_cols + outlier_cols
        missing_cols = [col for col in cols_to_use if col not in confounds.columns]
        if missing_cols:
            print(f"\n Warning: Missing columns in confounds file for run {r+1}: {missing_cols}")
        cols_to_use = [col for col in cols_to_use if col in confounds.columns]

        confounds_lst.append(confounds[cols_to_use])
    
    compcor_m    = min(compcor_lst)
    non_steady_M = max(non_steady_lst)
    outlier_M    = max(outlier_lst)
    

    return confounds_lst, perc_non_steady_lst, perc_outliers_lst, compcor_m, non_steady_M, outlier_M

# =============================================================================
# load a list of dim matched confound of a given (sub,ses)
# =============================================================================
def load_confound(confounds_lst, perc_non_steady, perc_outlier, compcor_m, non_steady_M, outlier_M):
    """" Regularize the size of confound matrix
        Retrun: list, float, float; list of confounds, mean percent of non_steady for the session, mean percent of outliers for the session
    """
    new_confounds = []
    for r in range(len(confounds_lst)):
        confounds      = confounds_lst[r]
        hmp_cols       = [col for col in confounds if col in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
        non_steady_col = [col for col in confounds if col.startswith('non_steady')]
        fd_outlier_col = [col for col in confounds if col.startswith('fd_outlier')]
        compcor_col    = [col for col in confounds if col.startswith('a_comp_')]
        cosine_col     = [col for col in confounds if col.startswith('cosine')]
        print(f"\n moving: {len(hmp_cols)},\n non_steady: {len(non_steady_col)}, \n outlier : {len(fd_outlier_col)}, \n acomp : {len(compcor_col)}, \n cosine : {len(cosine_col)}")
        
        # take the minimum length of acompcor to regularize the dim of design matrices of (sub,ses)
        if len(compcor_col) > compcor_m:
            keep_cols = compcor_col[:compcor_m]
            drop_cols = [col for col in compcor_col if col not in keep_cols]
            confounds = confounds.drop(columns=drop_cols)
        
        # add non_steady_state_outlier 
        if len(non_steady_col) < non_steady_M:
            n_missing = non_steady_M - len(non_steady_col)
            t = len(confounds)
            padding_cols = [f'non_steady_state_outlier{i+len(non_steady_col)}' for i in range(n_missing)]
            padding_df = pd.DataFrame(np.zeros((t, n_missing)), columns=padding_cols)
            
            # Find the maximum index of the non_steady_state_outlier columns
            if any(col.startswith('non_steady') for col in confounds.columns): 
                insert_at = confounds.columns.get_loc(non_steady_col[-1]) + 1
                part1     = confounds.iloc[:, :insert_at]
                part2     = confounds.iloc[:, insert_at:]
                confounds = pd.concat([part1, padding_df, part2], axis=1)
        
            else: # no non_steady_state_outlier
                insert_at = confounds.columns.get_loc(cosine_col[-1]) + 1
                part1     = confounds.iloc[:, :insert_at]
                part2     = confounds.iloc[:, insert_at:]
                confounds = pd.concat([part1, padding_df, part2], axis=1)
                
            all_ns_cols = [col for col in confounds.columns if col.startswith('non_steady_state_outlier')]
            new_names = [f'non_steady_state_outlier{i}' for i in range(non_steady_M)]
            confounds.rename(columns=dict(zip(all_ns_cols, new_names)), inplace=True)    

        # add fd_outlier
        if len(fd_outlier_col) < outlier_M:
            n_missing = outlier_M - len(fd_outlier_col)
            t = len(confounds)
            padding_cols = [f'fd_outlier{i+len(fd_outlier_col)}' for i in range(n_missing)]
            padding_df = pd.DataFrame(np.zeros((t, n_missing)), columns=padding_cols)

            insert_at = (confounds.columns.get_loc(fd_outlier_col[-1]) + 1
                         if fd_outlier_col else len(confounds.columns))
            confounds = pd.concat([confounds.iloc[:, :insert_at],
                                   padding_df,
                                   confounds.iloc[:, insert_at:]], axis=1)
            # Rename all fd_outlier columns to consistent names
            all_fd_cols = [col for col in confounds.columns if col.startswith('fd_outlier')]
            new_names = [f'fd_outlier{i}' for i in range(outlier_M)]
            confounds.rename(columns=dict(zip(all_fd_cols, new_names)), inplace=True)
            
        new_confounds.append(confounds)
    # mean prec_non_steady of a session
    m_perc_non_steady = np.mean(perc_non_steady)
    m_perc_outlier    = np.mean(perc_outlier)
        
    return new_confounds, m_perc_non_steady, m_perc_outlier

# =============================================================================
# load a list of brain image mask files for a given (sub,ses)
# =============================================================================
def load_mask_img(layout, subject, session, task, space):
    """Loads brain masks across runs.
    Return: list, list; a list mask images from each run, a list of absolute location of mask files"""
    
    mask_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='brain', suffix='mask',
                            extension='nii.gz')
    print(f"Subject={subject}, Session={session}, Task={task}, Space={space}")
    print(f"Mask files found: {mask_files}")
    
    if not mask_files:
        raise ValueError(f"No brain masks found for subject={subject}, session={session}, task={task}, space={space}.")

    # Load images and extract absolute paths
    mask_imgs  = [load_img(f.path) for f in mask_files]
    mask_paths = [os.path.abspath(f.path) for f in mask_files]
    
    return mask_imgs, mask_paths
    
# =============================================================================
# load a list of functional img fiels for a given (sub,ses)
# =============================================================================
def load_func_img(layout, subject, session, task, space):
    """Loads functional images across runs.
    Return: list; a list functional images from each run"""
    
    func_files = layout.get(subject=subject, session=session, task=task,
                            space=space, suffix='bold', extension='nii.gz')
    print(f"Subject={subject}, Session={session}, Task={task}, Space={space}")
    print(f"func files found: {func_files}")
    
    if not func_files:
        raise ValueError(f"No func img found for subject={subject}, session={session}, task={task}, space={space}.")
    
    func_imgs = [load_img(func_file) for func_file in func_files]
    
    return func_imgs

# =============================================================================
# Sort subjects and sessions
# =============================================================================
def get_subjects_sessions(layout, task, space, subject=None):
    """Gets a list of all subject-session pairs with preprocessed data."""
    if not subject:
        subjects = layout.get_subjects(task=task, space=space, desc='preproc',
                                   suffix='bold', extension='nii.gz')
        remove = {'ewy3'}
        subjects = list(filter(lambda x: x not in remove, subjects))
    else:
        subjects = subject.split(',')
    all_sessions = [layout.get_sessions(subject=subject, task=task,
                                        space=space, desc='preproc',
                                        suffix='bold', extension='nii.gz')
                    for subject in subjects]

    subjects_sessions = [(subject, session)
                         for subject, sessions in zip(subjects, all_sessions)
                         for session in sessions]

    return sorted(subjects_sessions)

# =============================================================================
# Build contrast array 
# =============================================================================
def make_contrast_vector(design_matrix_columns, condition_weights):
    """Build a contrast vector relevant to design matrix
       Return: np array; contrast values assigned array"""
       
    contrast = np.zeros(len(design_matrix_columns))
    
    for condition, weight in condition_weights.items():
        if condition not in design_matrix_columns:
            raise ValueError(f"'{condition}' not found in design matrix columns.")
            
        idx = design_matrix_columns.index(condition)
        contrast[idx] = weight
    return contrast

# =============================================================================
# First level glm of multiple runs of a give (sub,ses)
# =============================================================================
# Run first level glm within a session (3 or 4 runs)
def run_glm(layout, bids_dir, fmriprep_dir, pybids_dir, task, space, blockwise,
            fd_threshold, hrf_model, smoothing_fwhm, output_dir,
            save_residuals, subject, session):
    """Runs a first-level GLM for a given subject and session.
        Return: list, list, list, float, float, list;
                list of FirstLevelModel beta img (nilearn), list of mask imgs, list of mask imgs paths,
                mean perc_non_steady, mean perc_fd_outlier, list of residual files """
    
    # load inputs for nilearn.FirstLevelModel for multiple runs
    func_imgs             = load_func_img(layout, subject, session, task, space)
    mask_imgs, mask_paths = load_mask_img(layout, subject, session, task, space)
    events                = load_events(layout, subject, session, task)
    nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M = \
        get_confounds(layout, subject, session, task, fd_threshold)
    confounds, perc_non_steady, perc_outliers             = \
        load_confound(nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M)

    # FirstLevelModel
    print(f"Running GLM for subject {subject}, session {session}...")


    # Transform scans to time (Start from 1 sec, TR duration)    
    n_scans     = func_imgs[0].shape[-1]
    tr          = layout.get_tr()
    frame_times = tr * (np.arange(n_scans) + 0.5) 
   
    print(f"\n    Shape of frame_times: {frame_times.shape}")
    
    # create design matrices from event files for each run
    design_matrix = []
    for r in range(len(confounds)):
        d_matrix = make_first_level_design_matrix(frame_times,
                                                  events=events[r],
                                                  hrf_model=hrf_model,
                                                  drift_model=None,
                                                  high_pass=None,
                                                  add_regs=confounds[r],)
        design_matrix.append(d_matrix)
        
        
    # FirstLevelModel (signals mean to zero along time axis)
    glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm,
                                mask_img=mask_imgs[0], minimize_memory=False)

        
    # Fit the model
    glm_result = glm.fit(run_imgs=func_imgs, design_matrices=design_matrix)
        
    # Save residual files
    func_dir = output_dir / f'sub-{subject}' / f'ses-{session}' / 'func'
    func_dir.mkdir(parents=True, exist_ok=True)

    residuals = glm.residuals
    
    for r in range(len(residuals)):
        residuals_filename = f'sub-{subject}_ses-{session}_task-{task}_run-{r+1}space-{space}_desc-residuals.nii.gz'
        residuals_file = func_dir / residuals_filename
        residuals[r].to_filename(residuals_file)
        
    # Compute contrast, save effect, effect_variance and z score for second level analysis
    all_effect_size = []
    for index, (contrast_id, condition_weights) in enumerate(CONTRASTS_SPEC.items()):
        print(f"  Contrast {index + 1:02d} of {len(CONTRASTS_SPEC)}: {contrast_id}")
        
        # Create contrast vector assuming that all design matrices having the same dimension
        contrast_vector = make_contrast_vector(list(design_matrix[0].columns), condition_weights)
    
        # Compute contrasts
        effect_size = glm.compute_contrast(contrast_vector, output_type="effect_size")
        z_score     = glm.compute_contrast(contrast_vector, output_type="z_score")
    
        all_effect_size.append(effect_size)
    
        # Save outputs
        ef_image_path = func_dir / f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel_{contrast_id}_effect_size_map.nii.gz"
        z_image_path = func_dir / f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel_{contrast_id}_z_map.nii.gz"
        effect_size.to_filename(ef_image_path)
        z_score.to_filename(z_image_path)

                
    # Save report of multiple run firstlevel model (i.e. fixed effects from each session across valid runs)
    
    design_columns = list(design_matrix[0].columns)
    contrast_vectors = {
        name: make_contrast_vector(design_columns, weights)
        for name, weights in CONTRASTS_SPEC.items()
    }

    save_glm_to_bids(glm, contrasts=contrast_vectors,
            out_dir= output_dir / "derivatives" / f"sub-{subject}" / f"ses-{session}" ,
            prefix=f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel",)
        
    # Create report of a session
    mean_func_img = mean_img(func_imgs)
    report        = make_glm_report(model=glm, contrasts=contrast_vectors, bg_img = mean_func_img)
    report.save_as_html(func_dir / "report.html")            
    return all_effect_size, mask_imgs, mask_paths, perc_outliers, perc_non_steady, residuals 

# =============================================================================
# load meta data frame
# =============================================================================
def load_meta_df(layout, task, percs_outliers, percs_non_steady, df_query, subject = None):
    """Load the DataFrame with the subject/session metadata for the mixed model."""
    """ subject should be None or a list of subjects """

    # Fetch run-level collections and convert to DataFrame
    basic_df = layout.get_collections(task=task, level='session', types='scans', merge=True).to_df() 
    basic_df = basic_df.sort_values(['subject', 'session', 'acq_time']).reset_index(drop=True)
        
    # Aggregate to session-level by selecting the earliest acquisition time per session
    df = (
        basic_df.groupby(['subject', 'session'], as_index=False)
        .agg({'acq_time': 'min'})  # Retain only the earliest acquisition time
    )

    if subject:
        df = df[df['subject'].isin(subject)]
   
    # Debugging
    print(f"Length of basic_df: {len(basic_df)}")
    print(f"Length of df after aggregation: {len(df)}")
    

    # Add metadata columns
    df['n_sessions'] = df.groupby('subject')['session'].transform('count')
    df['perc_non_steady'] = percs_non_steady
    df['perc_outliers'] = percs_outliers
    df['acq_time'] = pd.to_datetime(df['acq_time'])
    print(f"Length of percs_non_steady: {len(percs_non_steady)}")
    print(f"Length of percs_outliers: {len(percs_outliers)}")

    # Add time-related columns
    df['time_diff'] = df['acq_time'] - df['acq_time'].min()
    df['time'] = df['time_diff'].dt.days / 30.437  # Convert to months
    #df['time2'] = df['time'] ** 2

    # Filter by query
    good_df               = df.query(df_query) # first filter -  fd outlier
    good_df['n_sessions'] = df.groupby('subject')['session'].transform('count')
    good_df               = df.query('n_sessions >= 2') # session more than 2
    good_df               = df.set_index(['subject', 'session']) # set index as (sub,ses)
    
    # get a list of good (sub,ses)
    good_ixs = good_df.index.tolist() 

    # reset the index of df
    #df = df.set_index(['subject', 'session'])
    
    metadata_file = Path(layout.root) / f'derivatives/metadata_task-{task}.tsv'
    df.to_csv(metadata_file, sep='\t', index=False, float_format='%.5f')
    print(f"Metadata saved to {metadata_file}")

    return df, good_ixs

# =============================================================================
# Miscellanious fucntions to save images 
# =============================================================================
def combine_save_mask_imgs(mask_imgs, output_dir, task, space,
                           perc_threshold=1):
    """
    input: ...
    output: binarized image data, nii file object, nii file name
    
    Combines brain masks across subjects and sessions and saves the result.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    mask_img = combine_mask_imgs(mask_imgs, perc_threshold)

    mask_file, mask_name = save_img(mask_img, output_dir, task, space,
                         desc='brain', suffix='mask')

    return mask_img, mask_file, mask_name


def combine_mask_imgs(mask_imgs, perc_threshold=1):
    """
    Input: nii image or a list of images
           If a list of 4D images is given, the mean of each 4D image is 
           computed separately, and the resulting mean is computed after.
    Combines brain masks across subjects and sessions.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    return binarize_img(mean_img(mask_imgs), threshold=perc_threshold)


def save_img(img, output_dir, task, space, desc, suffix,
             subject=None, session=None):
    """Saves a NIfTI image to a file in the output directory."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.nii.gz'

    if session:
        filename = f'ses-{session}_{filename}'

    if subject:
        filename = f'sub-{subject}_{filename}'

    file = output_dir / filename
    img.to_filename(file)

    return file, filename
