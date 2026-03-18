"""
MVPA (Multi Voxel Pattern Analysis)

It uses the results of first level anlaysis of COCOA to run MVPA at each session
First Level Analysis: Beta value of each alphabet condition & geometry label(ROI) - (sub,ses,run) is the unit of pattern analysis per (condition, ROI)

- SVC (supportive vector classification): Leave one out cross validation, RBF/Linear Kernel, 6 class classification (6 alphabets), ROI masker (each ROI and union of ROIs)
- save classification probability per (sub,ses,ROI) 
- the saved result tested using Linear mixed model (LMM) (probability ~ session*condition + (1| subject) + (session | subject)) to test longditudinal changes of classification probability.
  with R

ref: Multi-voxel pattern analysis for developmental cognitive neuroscientists (Guassi Moreira, 2025, Developmental Cognitive Nuerosicence)
"""

# MPVA
import os
import ast
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from multiprocessing import Process, Queue

import concurrent.futures
from nibabel.freesurfer.io import read_label

from nilearn.surface import load_surf_data, SurfaceImage
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.decoding import Decoder

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

#------------------------------------------------------------------------------
## Global variables
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FREESURFER_DIR = DERIVATIVES_DIR / 'freesurfer'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate_surf_smooth2'

# DIR of ROI masks
mask_dir = DERIVATIVES_DIR / 'masks' / 'surf_masks'

# Univaraite functional analysis contrasts
CONTRASTS_SPEC = ['congruent','incongruent','congruentMinusIncongruent']

# geometry types (COCOA: anatomical ROIs from LMM analysis showing significant effects)
geo_type = ['area','curv','volume']

# Regularization paramenter of SVC 
# For small sample size (under 10), fix it as a samll value
c = 0.1

# kernel of SVC: for FMRI data, linear kernel is widely used considering interpretability
kernel = ['rbf', 'linear'] # scikit learn default kernel for SVC is 'rbf'

# used alphabets
letter_list = ['B','F','K','M','P','T']

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



## Help Functions
"""
#==============================================================================
# Compare the list of first level glm effet files and good_ixs
# len of glm_l files and glm_r files shouldn be >= 12 (6 trial types * 2 runs)
# Once you fulfill the requirement, comment the following lines out.

sub_lst   = []
ses_lst   = []
glm_l_lst = []
glm_r_lst = []

contrast_id ='congruentMinusIncongruent'

for (sub, ses) in good_ixs:    
    subject = sub
    session = ses
    print(f'sub {sub} ses {ses} is being processing.')
    
    # grab the beta maps of (Sub,Ses)
    glm_dir = UNIVARIATE_DIR/ f'sub-{subject}' 
    glm_files_l = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id}*effect_statmap.gii'))
    glm_files_l = sorted(glm_files_l)
    glm_files_r = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id}*effect_statmap.gii'))
    glm_files_r = sorted(glm_files_r)
    
    len_l = len(glm_files_l)
    len_r = len(glm_files_r)
    
    sub_lst.append(sub)
    ses_lst.append(ses)
    glm_l_lst.append(len_l)
    glm_r_lst.append(len_r)
    
df = pd.DataFrame({'Subject':sub_lst, 'Session':ses_lst, 'GLM_L':glm_l_lst, 'GLM_R':glm_r_lst})
df.to_csv(UNIVARIATE_DIR / f'good_ixs_glm_lst_cont-{contrast_id}.csv', index=False)   
#==============================================================================
"""

# Time out function

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout=60):
    """
    Run func(*args, **kwargs) in a subprocess with timeout.
    Returns result or raises TimeoutException.
    """
    if kwargs is None:
        kwargs = {}
    q = Queue()

    def wrapper(q, func, args, kwargs):
        try:
            res = func(*args, **kwargs)
            q.put(res)
        except Exception as e:
            q.put(e)

    p = Process(target=wrapper, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutException("Timeout reached")

    if q.empty():
        raise TimeoutException("No result returned")

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result

## Analysis Functions
#==============================================================================
def pre_MVPA(geo_type, contrasts_list, good_ixs, letter_lst, C, data_path):
    
    """
    ==============================================================================
     Creat a csv files of decoding probabilities for (sub,ses,contrast,c)
     c is the regularization parameter for SVC from Scikit learn 
     COCOA used fixed c = 0.1 
     Multiclass classification: decode alaphabet classes for a contrast
    ==============================================================================

    Parameters
    ----------
    geo_type : list
        list of measured anatomical metric e.g. ['area','curv'].
    contrasts_list : list
        list of contrasts e.g. ['congruent','incongruent','congruent-incongruent'].
    good_ixs : list
        list of (subject,session).
    letter_lst : list
        list of different alphaets e.g. ['B','F','K','M','P',T'].
    C : float
        Regularization parameter of SVM.
    data_path : str
        Path to load the first level analysis results and to save outputs.

    Returns
    -------
    None.

    """
    for geo in geo_type:
        all_prob = []
        
        for contrast_id in contrasts_list:
           
            # load the result of regularization test (when you have more than 10 runs per session, use MVPA_NestedCV_V2 and test C)
            #df_c = pd.read_csv(UNIVARIATE_DIR/"MVPA_NestedCV_v2"/f"nested_cv_results_contrast-{contrast_id}_geo-{geo}.csv")
            #df_c = df_c[df_c["Selected_C_Values"].notna()]
            
            # loop over sub-session
            for (sub, ses) in good_ixs:    
                subject = sub
                session = ses
                print(f'sub {sub} ses {ses} is being processing.')
                    
                # grab the beta maps of (Sub,Ses)
                glm_dir     = UNIVARIATE_DIR/ f'sub-{subject}' 
                glm_files_l = []
                glm_files_r = []
                
                for letter in letter_lst:
                    glm_f_l = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_f_r = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_files_l.extend(glm_f_l)
                    glm_files_r.extend(glm_f_r)
                    
                glm_files_l = sorted(glm_files_l)
                glm_files_r = sorted(glm_files_r)
                
                print(f'glm_l files: [{len(glm_files_l)}], glm_r_files: [{len(glm_files_r)}]')
                
                # create labels for class and cv fold
                trial_labels = [str(f)[str(f).find(contrast_id) + len(contrast_id)] for f in glm_files_l]    #for func_imgas, e.g., 'a','b'...
                run_labels   = [str(f)[str(f).find('run-') + 4] for f in glm_files_l]    #label to mark which run it is for cv, e.g., 0, 0 ,1,1..
                
                # grab the label file pahts
                mask_f_l = list(mask_dir .glob( f'**/lh*{geo}*label*'))
                mask_f_r = list(mask_dir .glob( f'**/rh*{geo}*label*'))
                print(f'mask {geo} hemi-L [{len(mask_f_l)}], hemi-R [{len(mask_f_r)}]')
                
                # create the full vertice size numpy array
                mask_l_arr = nib.load(glm_files_l[0]).darrays[0].data.shape
                mask_r_arr = nib.load(glm_files_r[0]).darrays[0].data.shape

                # Create dictionaries to store individual masks for each hemisphere.
                mask_l_dic = {}
                mask_r_dic = {}
                all_roi_names = set()
                
                # Process left hemisphere masks
                for lab_l in mask_f_l:
                    roi_name = Path(lab_l).stem.replace('lh.', '')
                    verts = read_label(lab_l)
                    mask_arr = np.zeros(mask_l_arr, dtype=bool)
                    mask_arr[verts] = True
                    mask_l_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Process right hemisphere masks
                for lab_r in mask_f_r:
                    roi_name = Path(lab_r).stem.replace('rh.', '')
                    verts = read_label(lab_r)
                    mask_arr = np.zeros(mask_r_arr, dtype=bool)
                    mask_arr[verts] = True
                    mask_r_dic[roi_name] = mask_arr
                    all_roi_names.add(roi_name)
                
                # Create the final dictionary of masks to loop through, handling missing hemispheres.
                mask_imgs = {}
                union_mask_l = np.zeros(mask_l_arr, dtype=bool)
                union_mask_r = np.zeros(mask_r_arr, dtype=bool)
                
                for roi_name in all_roi_names:
                    mask_l = mask_l_dic.get(roi_name, np.zeros(mask_l_arr, dtype=bool))
                    mask_r = mask_r_dic.get(roi_name, np.zeros(mask_r_arr, dtype=bool))
                    mask_imgs[roi_name] = (mask_l, mask_r)
                    union_mask_l |= mask_l
                    union_mask_r |= mask_r
                
                # Add the 'union' mask to the dictionary
                mask_imgs['union'] = (union_mask_l, union_mask_r)
                print(f"Prepared masks for: {list(mask_imgs.keys())}")
                print("-" * 50)
                
                # -----------------------------------------------------------------------------
                # DECODING: LOOP OVER EACH MASK AND PERFORM DECODING
                
                all_roi_summary_results = []
                all_roi_probability_results = []
                
                for roi_label, (mask_l, mask_r) in mask_imgs.items():
                    print(f"Starting decoding for ROI: {roi_label}")
                    
                    # Prepare NumPy array by applying the current ROI's mask
                    glms = []
                    for (img_l, img_r) in zip(glm_files_l, glm_files_r):
                        d_l = nib.load(img_l).darrays[0].data
                        d_r = nib.load(img_r).darrays[0].data        
                        # Apply the current mask
                        d_l_masked = d_l[mask_l]
                        d_r_masked = d_r[mask_r]
                        
                        # Concatenate the left and right hemisphere data
                        combined_data = np.concatenate([d_l_masked, d_r_masked])
                        glms.append(combined_data)
                        
                    # Data sets, label, cv labels
                    X = np.vstack(glms)
                    y = np.array(trial_labels)
                    groups = np.array(run_labels)
                    
                    # Check if there are enough features left in the ROI after masking
                    if X.shape[1] == 0:
                        print(f"Warning: ROI '{roi_label}' has no vertices after masking. Skipping.")
                        continue
                                  
                    # Mean of selected C from gridsearc_cv (when you have over 10 runs per session)
                    # try:
                    #     selected_C = df_c[(df_c['Subject']== subject) & (df_c['Session']==int(session[1]))&df_c['ROI'== roi_label]]["Selected_C_Values"]
                    #     c          = np.mean(ast.literal_eval(selected_C[0]))
                    #     c          = round(c,2)
                    # except:
                    #     c = 0.5
                    
                    # Create the classification pipeline
                    classifier_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svc', SVC(C=c, kernel = kernel[1], probability=True, random_state=42))
                    ])
                    
                    cv = LeaveOneGroupOut()
                                
                    try:
                        probabilities = run_with_timeout(
                            cross_val_predict,
                            args=(classifier_pipeline, X, y),
                            kwargs={
                                "groups": groups,
                                "cv": cv,
                                "n_jobs": 1,   # no nested parallelism
                                "method": "predict_proba"
                            },
                            timeout=600  # seconds per ROI
                        )
                    except TimeoutException:
                        print(f"ROI {roi_label} timed out → assigning chance level.")
                        n_classes = len(np.unique(y))
                        probabilities = np.ones((len(y), n_classes)) / n_classes
                    except Exception as e:
                        print(f"ROI {roi_label} failed: {e} → assigning chance level.")
                        n_classes = len(np.unique(y))
                        probabilities = np.ones((len(y), n_classes)) / n_classes
                                        
                    # -----------------------------------------------------------------------------
                    # AGGREGATE PROBABILITIES AND CALCULATE ACCURACY
                    
                    # Create a DataFrame to easily group and aggregate the data.
                    results_df = pd.DataFrame(probabilities, columns=sorted(np.unique(y)))
                    results_df['true_class'] = y
                    results_df['run'] = groups
                    results_df['subject'] = subject
                    results_df['session'] = session
                    results_df['roi'] = roi_label
                    
                    # Store all probability results
                    all_roi_probability_results.append(results_df)
                
                    # Calculate the mean probability for each class across runs
                    mean_probabilities = results_df.groupby(['subject', 'session', 'run', 'true_class', 'roi']).mean()
                    
                    # Calculate the mean correct prediction probability for each class.
                    within_class_probabilities = []
                    classes = sorted(np.unique(y))
                    for true_class in classes:
                        mean_prob = mean_probabilities.xs(true_class, level='true_class')[true_class].mean()
                        within_class_probabilities.append(mean_prob)
                    
                    # Create a summary DataFrame for this ROI and store it.
                    summary_df = pd.DataFrame({
                        'Subject': [subject] * len(classes),
                        'Session': [session] * len(classes),
                        'Class': classes,
                        'Mean Correct Probability': within_class_probabilities,
                        'ROI': [roi_label] * len(classes)
                    })
                    
                    all_roi_summary_results.append(summary_df)
                    
                    print(f"Finished decoding for ROI: {roi_label}")
                    print("-" * 50)
        
        # -----------------------------------------------------------------------------
        # COMBINE AND SAVE ALL RESULTS TO A CSV FILE
        
                final_summary_df         = pd.concat(all_roi_summary_results, ignore_index=True)
                final_all_probability_df = pd.concat(all_roi_probability_results, ignore_index=True)
                
                fdir = UNIVARIATE_DIR / 'MVPA' / f'{geo}'/ f'{kernel[1]}' 
                if not os.path.exists(fdir):
                    os.makedirs(fdir)
                
                file_name1 = fdir/f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id}_c-{round(c,2)}_kernel-{kernel[1]}_all.csv'
                file_name2 = fdir/f'sub-{subject}_ses-{session}_anat-{geo}_{contrast_id}_c-{round(c,2)}_kernel_{kernel[1]}_right_prob.csv'
                
                final_all_probability_df.to_csv(file_name1, index=False)
                print(f'Saved all probability data to {file_name1}')
                
                final_summary_df.to_csv(file_name2, index=False)
                print(f'Saved summary accuracy data to {file_name2}')
                
                print(f"\nDecoding for all ROIs and the union is complete for sub {subject}, ses {session}, {geo}, {contrast_id}, regularization {c}.")
                print("-" * 50)
        
            # -----------------------------------------------------------------------------        
            # Creat long-data of entire (sub,ses,ROI, mean_correct_prob, class)
            f_dir   = UNIVARIATE_DIR / "MVPA" / f"{geo}" / f"{kernel[1]}"
            df_files = sorted(f_dir.glob(f"**/*anat-{geo}_{contrast_id}_c-{c}_kernel_{kernel[1]}_right_prob.csv"))
            
            all_cont_dfs = []  
            
            for f in df_files:
                df             = pd.read_csv(f)
                df["contrast"] = contrast_id  # add contrast column
                all_cont_dfs.append(df)
            
            # concatenate contrast dataframes
            df_cont_all = pd.concat(all_cont_dfs, ignore_index=True)
            df_cont_all = df_cont_all.rename(columns={
                "Mean Correct Probability": "Mean.Correct.Probability"
            })
            
            all_prob.append(df_cont_all)
            
        df_all   = pd.concat(all_prob,ignore_index=True)
        out_file = f_dir / "all_right_prob.csv"
        df_all.to_csv(out_file, index=False)
        
        print(f"Saved combined file with {len(df_all)} rows to {out_file}")
        
    return print(f'\nMVPA DONE and SAVED in {UNIVARIATE_DIR}/MVPA in (sub,ses) level.')

#==============================================================================
def pre_MVPA2(geo_type, contrasts_list, good_ixs, letter_lst, C, input_path, output_path):
    
    """
    ==============================================================================
     Creat a csv files of decoding probabilities for (sub,ses,c)
     c is the regularization parameter for SVC from Scikit learn 
     COCOA used fixed c = 0.1 
     letter blinded, binary classification of contlasts (congruent vs incongruent)
    ==============================================================================

    Parameters
    ----------
    geo_type : list
        list of measured anatomical metric e.g. ['area','curv'].
    contrasts_list : list
        list of contrasts to be classified e.g. ['congruent','incongruent].
    good_ixs : list
        list of (subject,session).
    letter_lst : list
        list of different alphaets e.g. ['B','F','K','M','P',T'].
    C : float
        Regularization parameter of SVM.
    input_path : Path object or str
        Path to load the first level analysis results.
    output_path : Path object or str
        Path to save outputs.

    Returns
    -------
    None.

    """
    for geo in geo_type:

        # loop over sub-session
        for (sub, ses) in good_ixs:    
            subject = sub
            session = ses
            print(f'sub {sub} ses {ses} is being processing.')
                
            # grab the beta maps of (Sub,Ses)
            glm_dir     = UNIVARIATE_DIR/ f'sub-{subject}' 
            glm_files_l = []
            glm_files_r = []
        
            for letter in letter_lst:
                for contrast_id in contrasts_list:
                    glm_f_l = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-L*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_f_r = list(glm_dir.glob(f'**/*{subject}*{session}*hemi-R*contrast-{contrast_id+letter}_stat-effect_statmap.gii'))
                    glm_files_l.extend(glm_f_l)
                    glm_files_r.extend(glm_f_r)
                
            glm_files_l = sorted(glm_files_l)
            glm_files_r = sorted(glm_files_r)
            
            print(f'glm_l files: [{len(glm_files_l)}], glm_r_files: [{len(glm_files_r)}]')
            
            # create labels for class and cv fold
            trial_labels = [str(f)[str(f).find('contrast-')+9:str(f).find('_stat-effect_statmap.gii')-1] for f in glm_files_l]    # for func_images, e.g., 'congruent', 'incongruent',..    #for func_imgas, e.g., 'a','b'...
            run_labels   = [str(f)[str(f).find('run-') + 4] for f in glm_files_l]    #label to mark which run it is for cv, e.g., 0, 0 ,1,1..
            
            # grab the label file pahts
            mask_f_l = list(mask_dir .glob( f'**/lh*{geo}*label*'))
            mask_f_r = list(mask_dir .glob( f'**/rh*{geo}*label*'))
            print(f'mask {geo} hemi-L [{len(mask_f_l)}], hemi-R [{len(mask_f_r)}]')
            
            # create the full vertice size numpy array
            mask_l_arr = nib.load(glm_files_l[0]).darrays[0].data.shape
            mask_r_arr = nib.load(glm_files_r[0]).darrays[0].data.shape

            # Create dictionaries to store individual masks for each hemisphere.
            mask_l_dic = {}
            mask_r_dic = {}
            all_roi_names = set()
            
            # Process left hemisphere masks
            for lab_l in mask_f_l:
                roi_name = Path(lab_l).stem.replace('lh.', '')
                verts = read_label(lab_l)
                mask_arr = np.zeros(mask_l_arr, dtype=bool)
                mask_arr[verts] = True
                mask_l_dic[roi_name] = mask_arr
                all_roi_names.add(roi_name)
            
            # Process right hemisphere masks
            for lab_r in mask_f_r:
                roi_name = Path(lab_r).stem.replace('rh.', '')
                verts = read_label(lab_r)
                mask_arr = np.zeros(mask_r_arr, dtype=bool)
                mask_arr[verts] = True
                mask_r_dic[roi_name] = mask_arr
                all_roi_names.add(roi_name)
            
            # Create the final dictionary of masks to loop through, handling missing hemispheres.
            mask_imgs = {}
            union_mask_l = np.zeros(mask_l_arr, dtype=bool)
            union_mask_r = np.zeros(mask_r_arr, dtype=bool)
            
            for roi_name in all_roi_names:
                mask_l = mask_l_dic.get(roi_name, np.zeros(mask_l_arr, dtype=bool))
                mask_r = mask_r_dic.get(roi_name, np.zeros(mask_r_arr, dtype=bool))
                mask_imgs[roi_name] = (mask_l, mask_r)
                union_mask_l |= mask_l
                union_mask_r |= mask_r
            
            # Add the 'union' mask to the dictionary
            mask_imgs['union'] = (union_mask_l, union_mask_r)
            print(f"Prepared masks for: {list(mask_imgs.keys())}")
            print("-" * 50)
            
            # -----------------------------------------------------------------------------
            # DECODING: LOOP OVER EACH MASK AND PERFORM DECODING
            
            all_roi_summary_results = []
            all_roi_probability_results = []
            
            for roi_label, (mask_l, mask_r) in mask_imgs.items():
                print(f"Starting decoding for ROI: {roi_label}")
                
                # Prepare NumPy array by applying the current ROI's mask
                glms = []
                for (img_l, img_r) in zip(glm_files_l, glm_files_r):
                    d_l = nib.load(img_l).darrays[0].data
                    d_r = nib.load(img_r).darrays[0].data        
                    # Apply the current mask
                    d_l_masked = d_l[mask_l]
                    d_r_masked = d_r[mask_r]
                    
                    # Concatenate the left and right hemisphere data
                    combined_data = np.concatenate([d_l_masked, d_r_masked])
                    glms.append(combined_data)
                    
                # Data sets, label, cv labels
                X = np.vstack(glms)
                y = np.array(trial_labels)
                groups = np.array(run_labels)
                
                # Check if there are enough features left in the ROI after masking
                if X.shape[1] == 0:
                    print(f"Warning: ROI '{roi_label}' has no vertices after masking. Skipping.")
                    continue
                              
                # Mean of selected C from gridsearc_cv (when you have over 10 runs per session)
                # try:
                #     selected_C = df_c[(df_c['Subject']== subject) & (df_c['Session']==int(session[1]))&df_c['ROI'== roi_label]]["Selected_C_Values"]
                #     c          = np.mean(ast.literal_eval(selected_C[0]))
                #     c          = round(c,2)
                # except:
                #     c = 0.5
                
                # Create the classification pipeline
                classifier_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svc', SVC(C=c, kernel = kernel[1], probability=True, random_state=42))
                ])
                
                cv = LeaveOneGroupOut()
                            
                try:
                    probabilities = run_with_timeout(
                        cross_val_predict,
                        args=(classifier_pipeline, X, y),
                        kwargs={
                            "groups": groups,
                            "cv": cv,
                            "n_jobs": 1,   # no nested parallelism
                            "method": "predict_proba"
                        },
                        timeout=600  # seconds per ROI
                    )
                except TimeoutException:
                    print(f"ROI {roi_label} timed out → assigning chance level.")
                    n_classes = len(np.unique(y))
                    probabilities = np.ones((len(y), n_classes)) / n_classes
                except Exception as e:
                    print(f"ROI {roi_label} failed: {e} → assigning chance level.")
                    n_classes = len(np.unique(y))
                    probabilities = np.ones((len(y), n_classes)) / n_classes
                                    
                # -----------------------------------------------------------------------------
                # AGGREGATE PROBABILITIES AND CALCULATE ACCURACY
                
                # Create a DataFrame to easily group and aggregate the data.
                results_df = pd.DataFrame(probabilities, columns=sorted(np.unique(y)))
                results_df['true_class'] = y
                results_df['run'] = groups
                results_df['subject'] = subject
                results_df['session'] = session
                results_df['roi'] = roi_label
                
                # Store all probability results
                all_roi_probability_results.append(results_df)
            
                # Calculate the mean probability for each class across runs
                mean_probabilities = results_df.groupby(['subject', 'session', 'run', 'true_class', 'roi']).mean()
                
                # Calculate the mean correct prediction probability for each class.
                within_class_probabilities = []
                classes = sorted(np.unique(y))
                for true_class in classes:
                    mean_prob = mean_probabilities.xs(true_class, level='true_class').mean()
                    within_class_probabilities.append(mean_prob)
                
                # Create a summary DataFrame for this ROI and store it.
                summary_df = pd.DataFrame({
                    'Subject': [subject] * len(classes),
                    'Session': [session] * len(classes),
                    'Class': classes,
                    'Mean.Correct.Probability': within_class_probabilities,
                    'ROI': [roi_label] * len(classes)
                })
                
                all_roi_summary_results.append(summary_df)
                
                print(f"Finished decoding for ROI: {roi_label}")
                print("-" * 50)
    
    # -----------------------------------------------------------------------------
    # COMBINE AND SAVE ALL RESULTS TO A CSV FILE
    
            final_summary_df         = pd.concat(all_roi_summary_results, ignore_index=True)
            final_all_probability_df = pd.concat(all_roi_probability_results, ignore_index=True)
            
            f_dir = output_path /'MVPA2' / f"{geo}" / f"{kernel[1]}"
            if not os.path.exists(f_dir):
                os.makedirs(f_dir)
            
            file_name1 = f_dir/f'sub-{subject}_ses-{session}_anat-{geo}_c-{round(c,2)}_kernel-{kernel[1]}_all.csv'
            file_name2 = f_dir/f'sub-{subject}_ses-{session}_anat-{geo}_c-{round(c,2)}_kernel_{kernel[1]}_right_prob.csv'
            
            final_all_probability_df.to_csv(file_name1, index=False)
            print(f'Saved all probability data to {file_name1}')
            
            final_summary_df.to_csv(file_name2, index=False)
            print(f'Saved summary accuracy data to {file_name2}')
            
            print(f"\nDecoding for all ROIs and the union is complete for sub {subject}, ses {session}, {geo}, regularization {c}.")
            print("-" * 50)
    
        # -----------------------------------------------------------------------------        
        # Creat long-data of entire (sub,ses,ROI, mean_correct_prob, class)
        df_files = sorted(f_dir.glob(f"**/*anat-{geo}_c-{c}_kernel_{kernel[1]}_right_prob.csv"))
        
        all_cont_dfs = []  
        
        for f in df_files:
            df = pd.read_csv(f)
            all_cont_dfs.append(df)
        
        # concatenate contrast dataframes
        df_cont_all = pd.concat(all_cont_dfs, ignore_index=True)
        out_file    = f_dir / "all_right_prob.csv"
        df_cont_all.to_csv(out_file, index=False)
        
        print(f"Saved combined file with {len(df_cont_all)} rows to {out_file}")
    
    return print(f'\nBinary MVPA DONE and SAVED in {output_path}/MVPA2 in (sub,ses) level.')
