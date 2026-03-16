"""
Representation Similarity Anlysis (RSA)

Use the results of first level anlaysis of each session and run RSA at each session to use them for longitudinal analysis using LMM
First Level Analysis: Beta value of each alphabet condition & geometry label - (sub,ses,run) is the unit of pattern analysis

- Calculate representation dissimilarity matrix (RDM) of 6 alphabets per (sub,ses,ROI), ROI labels includes also unions of ROIs
- Save the result and use it as input for LMM (dissimilarity ~ session + (1|subject)+(session|subject)) 

"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from part3_betas import transform_key
from itertools import combinations
from nibabel.freesurfer.io import read_label

#------------------------------------------------------------------------------
## Global variables
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FREESURFER_DIR = DERIVATIVES_DIR / 'freesurfer'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate_surf3'

# directory to save results
res_dir = UNIVARIATE_DIR / 'RSA'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# DIR of ROI masks
mask_dir = DERIVATIVES_DIR / 'masks' / 'surf_masks'

# contrasts
#CONTRASTS_SPEC = ['congruent','incongruent']
CONTRASTS_SPEC = ['audios','visual']

# geometry types (COCOA: anatomical ROIs from area, curvature and volume)
geo_type = ['area', 'curv', 'volume']

# (sub,ses) list
meta_df  = pd.read_csv(UNIVARIATE_DIR/ 'metadata_task-Literacy.tsv', sep = '\t', dtype={'session':str})

# when you load session, the string is '1' not '01'. For consistancy, change the session str to '01' shape
for i in range(len(meta_df)):
    session = meta_df.loc[i,'session']
    if len(session) != 2:
        meta_df.loc[i,'session'] = '0' + session 
        
meta_df  = meta_df.set_index(['subject','session'])
ixs      = meta_df.index.tolist()
good_ixs = [ix for ix in ixs if meta_df.loc[ix]['good_ixs'] == 'Y']  # ix = (sujbect, session)

## Miscellaneous Function 

#==============================================================================
# Compare the list of first level glm effet files and good_ixs
# cross_run RSA case, len of glm_l files and glm_r files shouldn be >= 12 (6 trial types * 2 runs)
# Once you fulfill the requirement, comment out the following lines.
def sanity_check(res_dir, CONTRASTS_SPEC):
    """
    

    Parameters
    ----------
    res_dir : Posix path object
        directory where the first glm saved.

    Returns
    -------
    missing_all : dict
        (Sub,Ses) having first level glm effect files.

    """
    missing_all = {}
    for contrast_id in CONTRASTS_SPEC:
        missing   = []
        sub_lst   = []
        ses_lst   = []
        glm_l_lst = []
        glm_r_lst = []
        for (sub, ses) in good_ixs:    
            print(f'sub {sub} ses {ses} is being processing.')
            # grab the beta maps of (Sub,Ses)
            glm_dir = res_dir / f'sub-{sub}' 
            glm_files_l = list(glm_dir.glob(f'**/*{sub}*{ses}*hemi-L*contrast-{contrast_id}*effect_statmap.gii'))
            glm_files_l = sorted(glm_files_l)
            glm_files_r = list(glm_dir.glob(f'**/*{sub}*{ses}*hemi-R*contrast-{contrast_id}*effect_statmap.gii'))
            glm_files_r = sorted(glm_files_r)
            len_l = len(glm_files_l)
            len_r = len(glm_files_r)
            sub_lst.append(sub)
            ses_lst.append(ses)
            glm_l_lst.append(len_l)
            glm_r_lst.append(len_r)
            
            if len_l <1:
                missing.append((sub,ses))
                print(f"No file found for sub: {sub}, ses: {ses}")
            
        df = pd.DataFrame({'Subject':sub_lst, 'Session':ses_lst, 'GLM_L':glm_l_lst, 'GLM_R':glm_r_lst})
        df.to_csv(UNIVARIATE_DIR / f'good_ixs_glm_lst_cont-{contrast_id}.csv', index=False)   
        print(f"Saved: {res_dir / f'good_ixs_glm_lst_cont-{contrast_id}.csv'}")
        missing_all[contrast_id] = missing
    return missing_all
#==============================================================================


## Script for analysis
#==============================================================================
# Creat a csv files of correlation matrix for (sub,ses,contrast)
#==============================================================================

# Cross run RSA: Pearson correlations in selected ROIs between different runs within a session 
def run_RSA(res_dir, CONTRASTS_SPEC, geo_type, good_ixs):
    
    for cont in CONTRASTS_SPEC:
        contrast_id = transform_key(cont)
        
        for geo in geo_type:
            
            # loop over sub-session
            for (sub, ses) in good_ixs:    
                subject = sub
                session = ses
                print(f'sub {sub} ses {ses} is being processing.')
                
                # grab the beta maps of (Sub,Ses)
                glm_dir     = res_dir/ f'sub-{subject}' 
                glm_files_l = []
                glm_files_r = []
                
                for letter in ['B','F','K','M','P','T']:
                    glm_f_l = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_f_r = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_files_l.extend(glm_f_l)
                    glm_files_r.extend(glm_f_r)
                    
                glm_files_l = sorted(glm_files_l)
                glm_files_r = sorted(glm_files_r)
                
                print(f'glm_l files: [{len(glm_files_l)}], glm_r_files: [{len(glm_files_r)}]')
                
                # create labels for class and cv fold
                trial_labels = [str(f)[str(f).find(contrast_id) + len(contrast_id)] for f in glm_files_l]    #for func_imgas, e.g., 'a','b'...
                run_labels   = [str(f)[str(f).find('run-') + 4] for f in glm_files_l]    #label to mark which run it is, e.g., 0, 0 ,1,1..
                
                # grab the label file pahts
                mask_f_l = list(mask_dir .glob( f'**/lh*{geo}*label*'))
                mask_f_r = list(mask_dir .glob( f'**/rh*{geo}*label*'))
                print(f'mask {geo} hemi-L [{len(mask_f_l)}], hemi-R [{len(mask_f_r)}]')
                
                # create the full vertice size numpy array
                mask_l_arr = nib.load(glm_files_l[0]).darrays[0].data.shape
                mask_r_arr = nib.load(glm_files_r[0]).darrays[0].data.shape

                # Create dictionaries to store individual masks for each hemisphere.
                mask_l_dic    = {}
                mask_r_dic    = {}
                all_roi_names = set()
                
                # Process left hemisphere masks
                for lab_l in mask_f_l:
                    roi_name             = Path(lab_l).stem.replace('lh.', '')
                    verts                = read_label(lab_l)
                    mask_arr             = np.zeros(mask_l_arr, dtype=bool)
                    mask_arr[verts]      = True
                    mask_l_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Process right hemisphere masks
                for lab_r in mask_f_r:
                    roi_name             = Path(lab_r).stem.replace('rh.', '')
                    verts                = read_label(lab_r)
                    mask_arr             = np.zeros(mask_r_arr, dtype=bool)
                    mask_arr[verts]      = True
                    mask_r_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Create the final dictionary of masks to loop through, handling missing hemispheres.
                mask_imgs    = {}
                union_mask_l = np.zeros(mask_l_arr, dtype=bool)
                union_mask_r = np.zeros(mask_r_arr, dtype=bool)
                
                for roi_name in all_roi_names:
                    mask_l              = mask_l_dic.get(roi_name, np.zeros(mask_l_arr, dtype=bool))
                    mask_r              = mask_r_dic.get(roi_name, np.zeros(mask_r_arr, dtype=bool))
                    mask_imgs[roi_name] = (mask_l, mask_r)
                    union_mask_l       |= mask_l
                    union_mask_r       |= mask_r
                
                # Add the 'union' mask to the dictionary
                mask_imgs['union'] = (union_mask_l, union_mask_r)
                print(f"Prepared masks for: {list(mask_imgs.keys())}")
                print("-" * 50)
                
                # -----------------------------------------------------------------------------
                # LOOP OVER EACH MASK AND CALCULATE CORRELATION COEFFICIENTS
                
                all_roi_summary_results = []

                for roi_label, (mask_l, mask_r) in mask_imgs.items():
                    print(f"Starting decoding for ROI: {roi_label}")
                    
                    # Extract masked beta patterns
                    glms, conds, runs = [], [], []
                    for img_l, img_r, cond, run in zip(glm_files_l, glm_files_r, trial_labels, run_labels):
                        d_l = nib.load(img_l).darrays[0].data
                        d_r = nib.load(img_r).darrays[0].data
                
                        d_l_masked = d_l[mask_l]
                        d_r_masked = d_r[mask_r]
                
                        combined_data = np.concatenate([d_l_masked, d_r_masked])
                        glms.append(combined_data)
                        conds.append(cond)
                        runs.append(run)
                    
                    # prepare data
                    X     = np.vstack(glms)
                    conds = np.array(conds)
                    runs  = np.array(runs)
                    
                    # prepare dataframe 
                    df_entries = []
                    
                    # Cross-run condition similarities -  pearson correlation
                    for i, j in combinations(range(len(X)), 2):
                        if runs[i] == runs[j]:
                            continue  # only cross-run comparisons
                        cond = sorted((conds[i],conds[j]))
                        r = np.corrcoef(X[i], X[j])[0, 1]
                        z = (1/2)*np.log((1+r)/(1-r))   # Fisher transformation
                        df_entries.append({
                            "subject": subject,
                            "session": session,
                            "roi": roi_label,
                            "cond": [conds[i],conds[j]],
                            "cond_sorted": (cond[0], cond[1]), 
                            "run_i": runs[i],
                            "run_j": runs[j],
                            "raw_corr": r,
                            "fisher_corr" : z
                        })  
                    all_roi_summary_results.append(pd.DataFrame(df_entries))
                # -----------------------------------------------------------------------------
                # save (sub,ses) csv       
                rsa_summary_df = pd.concat(all_roi_summary_results, ignore_index=True)
                
                # file directory and name to save results
                out_dir = res_dir /'RSA'/f'{geo}'
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                file_name1     = out_dir/ f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id}_RSM.csv'
                
                # calculate standard error and mean of a pair within a session 
                # df_rsa is your long-format run-pair dataframe
                group_cols = ['subject','session','roi','cond_sorted']
                
                agg = rsa_summary_df.groupby(group_cols)['fisher_corr'].agg(
                    mean_z = 'mean',
                    sd_z   = 'std',
                    n_pairs  = 'count'
                ).reset_index()
                file_name2 = out_dir/ f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id}_RSM_agg.csv'
                # compute standard error
                agg['se_z'] = agg['sd_z'] / np.sqrt(agg['n_pairs'])
                
                rsa_summary_df.to_csv(file_name1, index=False)
                print(f'Saved summary RSM: {file_name1}')
                
                agg.to_csv(file_name2, index=False)
                print(f'Saved aggregated summary RSM: {file_name2}')
                print(f"\nCalculation of RSM for all ROIs and the union is complete for sub {subject}, ses {session}, {geo}, {contrast_id}.")
                print("-" * 50)

    return print('\n    Saved Corss-run Pearson correlations.')

# Session wise Pearson correlation 
# 1st level GLM calculated across runs: No cross-run correlation
# Letter wise GLM
def session_RSA(res_dir, CONTRASTS_SPEC, geo_type, good_ixs):
    
    for cont in CONTRASTS_SPEC:
        contrast_id = transform_key(cont)
        
        for geo in geo_type:
            
            # loop over sub-session
            for (sub, ses) in good_ixs:    
                subject = sub
                session = ses
                print(f'sub {sub} ses {ses} contrast {contrast_id} is being processing.')
                
                # grab the beta maps of (Sub,Ses)
                glm_dir     = res_dir/ f'sub-{subject}' 
                glm_files_l = []
                glm_files_r = []
                
                for letter in ['B','F','K','M','P','T']:
                    glm_f_l = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_f_r = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_files_l.extend(glm_f_l)
                    glm_files_r.extend(glm_f_r)
                    
                glm_files_l = sorted(glm_files_l)
                glm_files_r = sorted(glm_files_r)
                
                print(f'glm_l files: [{len(glm_files_l)}], glm_r_files: [{len(glm_files_r)}]')
                
                # create labels for class and cv fold
                trial_labels = [str(f)[str(f).find(contrast_id) + len(contrast_id)] for f in glm_files_l]    #for func_imgas, e.g., 'a','b'...
                
                # grab the label file pahts
                mask_f_l = list(mask_dir .glob( f'**/lh*{geo}*label*'))
                mask_f_r = list(mask_dir .glob( f'**/rh*{geo}*label*'))
                print(f'mask {geo} hemi-L [{len(mask_f_l)}], hemi-R [{len(mask_f_r)}]')
                
                # create the full vertice size numpy array
                mask_l_arr = nib.load(glm_files_l[0]).darrays[0].data.shape
                mask_r_arr = nib.load(glm_files_r[0]).darrays[0].data.shape

                # Create dictionaries to store individual masks for each hemisphere.
                mask_l_dic    = {}
                mask_r_dic    = {}
                all_roi_names = set()
                
                # Process left hemisphere masks
                for lab_l in mask_f_l:
                    roi_name             = Path(lab_l).stem.replace('lh.', '')
                    verts                = read_label(lab_l)
                    mask_arr             = np.zeros(mask_l_arr, dtype=bool)
                    mask_arr[verts]      = True
                    mask_l_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Process right hemisphere masks
                for lab_r in mask_f_r:
                    roi_name             = Path(lab_r).stem.replace('rh.', '')
                    verts                = read_label(lab_r)
                    mask_arr             = np.zeros(mask_r_arr, dtype=bool)
                    mask_arr[verts]      = True
                    mask_r_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Create the final dictionary of masks to loop through, handling missing hemispheres.
                mask_imgs    = {}
                union_mask_l = np.zeros(mask_l_arr, dtype=bool)
                union_mask_r = np.zeros(mask_r_arr, dtype=bool)
                
                for roi_name in all_roi_names:
                    mask_l              = mask_l_dic.get(roi_name, np.zeros(mask_l_arr, dtype=bool))
                    mask_r              = mask_r_dic.get(roi_name, np.zeros(mask_r_arr, dtype=bool))
                    mask_imgs[roi_name] = (mask_l, mask_r)
                    union_mask_l       |= mask_l
                    union_mask_r       |= mask_r
                
                # Add the 'union' mask to the dictionary
                mask_imgs['union'] = (union_mask_l, union_mask_r)
                print(f"Prepared masks for: {list(mask_imgs.keys())}")
                print("-" * 50)
                
                # -----------------------------------------------------------------------------
                # LOOP OVER EACH MASK AND CALCULATE CORRELATION COEFFICIENTS
                
                all_roi_summary_results = []

                for roi_label, (mask_l, mask_r) in mask_imgs.items():
                    print(f"Starting decoding for ROI: {roi_label}")
                    
                    # Extract masked beta patterns
                    glms, conds = [], []
                    for img_l, img_r, cond in zip(glm_files_l, glm_files_r, trial_labels):
                        d_l = nib.load(img_l).darrays[0].data
                        d_r = nib.load(img_r).darrays[0].data
                
                        d_l_masked = d_l[mask_l]
                        d_r_masked = d_r[mask_r]
                
                        combined_data = np.concatenate([d_l_masked, d_r_masked])
                        glms.append(combined_data)
                        conds.append(cond)
                                        
                    # prepar data
                    X     = np.vstack(glms)
                    conds = np.array(conds)                 
                    
                    df_entries = []
                    # Session wise condition similarities - Pearson correlation
                    for i, j in combinations(range(len(X)), 2):
                        r = np.corrcoef(X[i], X[j])[0, 1]
                        z = (1/2)*np.log((1+r)/(1-r))   # Fisher transformation
                        df_entries.append({
                            "subject": subject,
                            "session": session,
                            "roi": roi_label,
                            "cond": [conds[i],conds[j]],
                            "raw_corr": r,
                            "fisher_corr" : z
                        })  
                    all_roi_summary_results.append(pd.DataFrame(df_entries))
                # -----------------------------------------------------------------------------
                # save (sub,ses) csv       
                rsa_summary_df = pd.concat(all_roi_summary_results, ignore_index=True)
                
                # file directory and name to save results
                out_dir = res_dir / 'RSA' / f'{geo}'
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                file_name1     = out_dir/ f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id}_RSM.csv'
            
                rsa_summary_df.to_csv(file_name1, index=False)
                print(f'Saved summary RSM: {file_name1}')
                print(f"\nCalculation of RSM for all ROIs and the union is complete for sub {subject}, ses {session}, {geo}, {contrast_id}.")
                print("-" * 50)

    return print('\n    Saved Session-wise Pearson correlation.')

# Perason correlation between conditions across two contarst ID
def cross_contrast_RSA(res_dir, CONTRASTS_PAIR, geo_type, good_ixs):
    """
    Compute Pearson correlations between conditions across two contrast IDs.

    CONTRASTS_PAIR : list or tuple of two contrast specs (as in CONTRASTS_SPEC)
    geo_type       : list of anatomical ROI grouping types 
    good_ixs       : list of (subject, session) tuples
    """

    cont1, cont2  = CONTRASTS_PAIR
    contrast_id_1 = transform_key(cont1)
    contrast_id_2 = transform_key(cont2)

    for geo in geo_type:
        for (sub, ses) in good_ixs:
            subject = sub
            session = ses
            print(f'\n cross-contrast RSA: sub {sub} ses {ses}, geo {geo}')
            print(f'Comparing {contrast_id_1} and {contrast_id_2}')

            glm_dir = res_dir / f'sub-{subject}'
            glm_files = {contrast_id_1: {'L': [], 'R': []},
                         contrast_id_2: {'L': [], 'R': []}}

            # gather all files for both contrasts
            for contrast_id in [contrast_id_1, contrast_id_2]:
                for letter in ['B', 'F', 'K', 'M', 'P', 'T']:
                    glm_files[contrast_id]['L'] += list(glm_dir.glob(
                        f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_files[contrast_id]['R'] += list(glm_dir.glob(
                        f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                glm_files[contrast_id]['L'] = sorted(glm_files[contrast_id]['L'])
                glm_files[contrast_id]['R'] = sorted(glm_files[contrast_id]['R'])

            print(f'glm files L: {len(glm_files[contrast_id_1]["L"])}, R: {len(glm_files[contrast_id_1]["R"])} (contrast1)')
            print(f'glm files L: {len(glm_files[contrast_id_2]["L"])}, R: {len(glm_files[contrast_id_2]["R"])} (contrast2)')

            # create labels per contrast
            trial_labels = {}
            for contrast_id in [contrast_id_1, contrast_id_2]:
                trial_labels[contrast_id] = [
                    str(f)[str(f).find(contrast_id) + len(contrast_id)] 
                    for f in glm_files[contrast_id]['L']
                ]

            # load ROI masks (same as your function)
            mask_f_l = list(mask_dir.glob(f'**/lh*{geo}*label*'))
            mask_f_r = list(mask_dir.glob(f'**/rh*{geo}*label*'))

            mask_l_arr = nib.load(glm_files[contrast_id_1]['L'][0]).darrays[0].data.shape
            mask_r_arr = nib.load(glm_files[contrast_id_1]['R'][0]).darrays[0].data.shape

            mask_l_dic, mask_r_dic = {}, {}
            all_roi_names = set()

            for lab_l in mask_f_l:
                roi_name = Path(lab_l).stem.replace('lh.', '')
                verts = read_label(lab_l)
                mask_arr = np.zeros(mask_l_arr, dtype=bool)
                mask_arr[verts] = True
                mask_l_dic[roi_name] = mask_arr
                all_roi_names.add(roi_name)

            for lab_r in mask_f_r:
                roi_name = Path(lab_r).stem.replace('rh.', '')
                verts = read_label(lab_r)
                mask_arr = np.zeros(mask_r_arr, dtype=bool)
                mask_arr[verts] = True
                mask_r_dic[roi_name] = mask_arr
                all_roi_names.add(roi_name)

            mask_imgs = {}
            for roi_name in all_roi_names:
                mask_l = mask_l_dic.get(roi_name, np.zeros(mask_l_arr, dtype=bool))
                mask_r = mask_r_dic.get(roi_name, np.zeros(mask_r_arr, dtype=bool))
                mask_imgs[roi_name] = (mask_l, mask_r)
            mask_imgs['union'] = (
                np.any([m[0] for m in mask_imgs.values()], axis=0),
                np.any([m[1] for m in mask_imgs.values()], axis=0)
            )

            all_roi_results = []

            # -----------------------------------------------------------
            for roi_label, (mask_l, mask_r) in mask_imgs.items():
                print(f'ROI: {roi_label}')

                # collect data for each contrast
                contrast_data = {}
                for contrast_id in [contrast_id_1, contrast_id_2]:
                    glms = []
                    for img_l, img_r in zip(glm_files[contrast_id]['L'], glm_files[contrast_id]['R']):
                        d_l = nib.load(img_l).darrays[0].data[mask_l]
                        d_r = nib.load(img_r).darrays[0].data[mask_r]
                        glms.append(np.concatenate([d_l, d_r]))
                    contrast_data[contrast_id] = np.vstack(glms)

                conds_1 = np.array(trial_labels[contrast_id_1])
                conds_2 = np.array(trial_labels[contrast_id_2])
                X1, X2 = contrast_data[contrast_id_1], contrast_data[contrast_id_2]

                df_entries = []
                # pairwise correlation between all (cond_i from contrast1) and (cond_j from contrast2)
                for i in range(len(X1)):
                    for j in range(len(X2)):
                        r = np.corrcoef(X1[i], X2[j])[0, 1]
                        z = 0.5 * np.log((1 + r) / (1 - r))
                        df_entries.append({
                            "subject": subject,
                            "session": session,
                            "roi": roi_label,
                            "cond1": conds_1[i],
                            "cond2": conds_2[j],
                            "contrast1": contrast_id_1,
                            "contrast2": contrast_id_2,
                            "raw_corr": r,
                            "fisher_corr": z
                        })

                all_roi_results.append(pd.DataFrame(df_entries))

            rsa_cross_df = pd.concat(all_roi_results, ignore_index=True)

            out_dir = res_dir /'RSA'/ f'{geo}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id_1}x{contrast_id_2}_CrossRSM.csv'
            rsa_cross_df.to_csv(out_file, index=False)

            print(f'    Saved cross-contrast RSA to: {out_file}')
            print('-' * 70)

    print("\nAll subjects/sessions processed for cross-contrast RSA.")


#==============================================================================
# Creat a long format csv files of correlation for all (sub,ses,contrast)
#==============================================================================
def long_csv(res_dir, geo_type):
    
    for geo in geo_type:
        res_dir = res_dir / 'RSA'
        out_dir = res_dir/ f"{geo}"
        
        if not os.path.exists(res_dir):
            print(f'No such {out_dir} exists.')  
            
        all_df = []
        
        for con in CONTRASTS_SPEC:
            res_list = list(out_dir.glob(f'**/*{geo}_{con}_RSM_agg.csv'))
            res_list = sorted(res_list)  
            
            for f in res_list:
                df             = pd.read_csv(f)
                df['contrast'] = [con]*len(df)
                all_df.append(df)       
                
            if len(all_df) == 0:
                print('No c relevant results exist.') 
                
        all_df_csv  = pd.concat(all_df, ignore_index = True)
        #all_df_name = out_dir/ 'all_right_prob.csv'
        all_df_name = out_dir/'all_sub_ses_longformat.csv'
        all_df_csv.to_csv(all_df_name, index=False)   
    return print(f"Suammry for {geo} is saved : {all_df_name} ")
