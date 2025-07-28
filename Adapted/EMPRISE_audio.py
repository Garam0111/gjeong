# This script is adapted to apply to audiotry stimuliation fMRI
# Major adatations are converting trial info from Pyschopy from block-wise to trial-wise,
# aligning fMRI images that there temporal dimensions are unmatched to each other.

"""
EMPRISE - EMergence of PRecISE numerosity representations in the human brain

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-26, 15:31: get_onsets
2023-06-26, 16:39: get_confounds
2023-06-26, 18:03: get_mask_nii, get_bold_nii, get_events_tsv, get_confounds_tsv
2023-06-29, 11:21: load_mask, load_data
2023-06-29, 12:18: onsets_trials2blocks
2023-07-03, 09:56: load_data_all, average_signals, correct_onsets
2023-07-13, 10:11: average_signals
2023-08-10, 13:59: global variables
2023-08-21, 15:42: rewriting to OOP
2023-08-24, 16:36: standardize_confounds
2023-09-07, 16:20: standardize_signals
2023-09-12, 19:23: get_bold_gii, load_surf_data, load_surf_data_all
2023-09-14, 12:46: save_vol, save_surf
2023-09-21, 15:11: plot_surf
2023-09-26, 16:19: analyze_numerosity
2023-09-28, 11:18: threshold_maps
2023-09-28, 12:58: visualize_maps
2023-10-05, 19:12: rewriting to OOP
2023-10-05, 19:34: global variables
2023-10-05, 21:24: rewriting for MPI
2023-10-12, 12:02: threshold_maps, visualize_maps
2023-10-16, 10:56: threshold_maps, testing
2023-10-16, 14:41: load_data_all, load_surf_data_all, get_onsets, get_confounds
2023-10-26, 17:56: get_model_dir, get_results_file, load_mask_data, calc_runs_scans
2023-10-26, 21:06: get_mesh_files, get_sulc_files
2023-11-01, 14:50: get_sulc_files
2023-11-01, 17:53: threshold_and_cluster
2023-11-09, 11:30: refactoring
2023-11-17, 10:34: refactoring
2023-11-20, 12:56: get_mesh_files
2023-11-20, 16:02: threshold_and_cluster
2023-11-23, 12:57: analyze_numerosity
2023-11-28, 14:05: create_fsaverage_midthick, get_mesh_files
2023-11-30, 19:31: threshold_AFNI_cluster
2024-01-22, 17:08: run_GLM_analysis
2024-01-29, 11:50: run_GLM_analysis
2024-01-29, 15:49: threshold_SPM
2024-03-11, 11:08: get_onsets
2024-03-11, 15:34: run_GLM_analysis_group
2024-03-11, 16:44: threshold_SPM_group
2024-03-11, 17:37: refactoring
2024-04-04, 10:22: get_onsets, get_confounds
2024-05-14, 15:03: analyze_numerosity
2024-05-21, 10:35: calculate_Rsq
2024-06-25, 15:18: threshold_maps, threshold_and_cluster
2024-06-27, 13:39: threshold_maps, threshold_and_cluster
2024-07-01, 18:11: analyze_numerosity
"""
"""
Adapted by Garam Jeong (for auditory stimuli analysis)
2024, get_spregressor_csv
      get_valid_runs
      load_surf_data_all
      get_confounds
      get_acom_confounds
      get_spregressor
2025, analyze_numerosity
"""

# import packages
#-----------------------------------------------------------------------------#
import os
import glob
import time
import re
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from nilearn import surface
from surfplot import Plot
import NumpRF_audio

# determine location
#-----------------------------------------------------------------------------#
at_MPI = os.getcwd().startswith('/data/')

# define directories
#-----------------------------------------------------------------------------#
if at_MPI:

    stud_dir = r'/data/pt_02495/emprise7t/'
    data_dir = stud_dir
    #deri_out = input('directory of outputs from preliminary analysis: leave it empty to use basic directory: ')
    #if not deri_out:
    #    deri_out = r'/data/pt_02495/emprise7t_analysis/derivatives/' 
    #else:
    deri_out = r'/data/pt_02495/emprise7t/derivatives/'  # preliminary analysis results directory
    print(f'deri_out of emp: {deri_out}')
    
    """ 
    else:
    stud_dir = r'C:/Joram/projects/MPI/EMPRISE/'
    data_dir = stud_dir + 'data/'
    deri_out = data_dir + 'derivatives/' 
    """
#deri_dir = data_dir + 'derivatives/'
deri_dir = r'/data/pt_02495/emprise7t/derivatives/' # preprocessed files directory

tool_dir = os.getcwd() + '/'   # /data/u_jeong_software/

# define identifiers
#-----------------------------------------------------------------------------#
sub   = '001'                   # pilot subject
ses   = 'visual'                # pilot session
sess  =['visual', 'audio', 'digits', 'spoken']
task  = 'harvey'
acq   =['mprageised', 'fMRI1p75TE24TR2100iPAT3FS']
runs  =[1,2,3,4,5,6,7,8]
spaces=['fsnative', 'fsaverage']
meshs = ['pial'] #['inflated', 'pial', 'white', 'midthickness']
desc  =['brain', 'preproc', 'confounds']

# define subject groups
#-----------------------------------------------------------------------------#
adults = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
childs = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '116']


# specify scanning parameters
#-----------------------------------------------------------------------------#
TR               = 2.1          # fMRI repetition time
mtr              = 41           # microtime resolution (= bins per TR)
mto              = 21           # microtime onset (= reference slice)
n                = 145          # number of scans per run
n2               = 129          # number of scans per congruent/incongruent run
b                = 4*2*6        # number of blocks per run
b2               = 4*2*5        # number of blocks per congruent/incongruent run
num_epochs       = 4            # number of epochs within run
num_scan_disc    = 1            # number of scans to discard before first epoch
scans_per_epoch  = int((n-num_scan_disc)/num_epochs)
scans_per_epoch2 = int((n2-num_scan_disc)/num_epochs)
blocks_per_epoch = int(b/num_epochs)
blocks_per_epoch2 = int(b2/num_epochs)

# specify thresholding parameters
#-----------------------------------------------------------------------------#
dAIC_thr  = 0                   # AIC diff must be larger than this
dBIC_thr  = 0                   # BIC diff must be larger than this
Rsq_def   = 0.3                 # R-squared must be larger than this
alpha_def = 0.05                # p-value must be smaller than this
mu_thr    =[1, 5]               # numerosity must be inside this range
fwhm_thr  =[0, 24]              # tuning width must be inside this range
beta_thr  =[0, np.inf]          # scaling parameter must be inside this range
crit_def  = 'Rsqmb'             # default thresholding option (see "threshold_maps")

# specify default covariates
#-----------------------------------------------------------------------------#
covs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
        'white_matter', 'csf', 'global_signal', \
        'cosine00', 'cosine01', 'cosine02']

# specify additional covariates
#-----------------------------------------------------------------------------#
covs_add = []

# class: subject/session
#-----------------------------------------------------------------------------#
class Session:
    """
    A Session object is initialized by a subject ID and a session ID and then
    allows for multiple operations performed on the data from this session.
    """
    
    # function: initialize subject/session
    #-------------------------------------------------------------------------#
    def __init__(self, subj_id, sess_id):
        """
        Initialize a Session from a Subject
        sess = EMPRISE.Session(subj_id, sess_id, ver_id)
        
            subj_id - string; subject identifier (e.g. "EDY7")
            sess_id - string; session identifier (e.g. "visual")
            
            sess    - a Session object
            o sub   - the subject ID
            o ses   - the session ID
        """
        
        # store subject ID and session name
        self.sub = subj_id
        self.ses = sess_id


    # function: get "mask.nii" filenames
    #-------------------------------------------------------------------------#
    def get_mask_nii(self, run_no, space):
        """
        Get Filename for Brain Mask NIfTI File
        filename = sess.get_mask_nii(run_no, space)
        
            run_no   - int; run number (e.g. 1)
            space    - string; image space (e.g. "T1w")
            
            filename - string; filename of "mask.nii.gz"
        
        filename = sess.get_mask_nii(run_no, space) returns the filename of the
        gzipped brain mask belonging to session sess, run run_no and in the
        selected image space.
        """
        
        # create filename
        filename = deri_dir + 'fmriprep' + \
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_space-' + space + '_desc-' + desc[0] + '_mask.nii.gz'
        return filename

    # function: get "bold.nii" filenames
    #-------------------------------------------------------------------------#
    def get_bold_nii(self, run_no, space=''):
        """
        Get Filename for BOLD NIfTI Files
        filename = sess.get_bold_nii(run_no, space)
        
            run_no   - int; run number (e.g. 1)
            space    - string; image space (e.g. "T1w")
            
            filename - string; filename of "bold.nii.gz"
        
        filename = sess.get_bold_nii(run_no, space) returns the filename of the
        gzipped 4D NIfTI belonging to session sess and run run_no. If space is
        non-empty, then the preprocessed images from the selected image space
        will be returned. By default, space is empty.
        """
        
        # create filename
        if not space:               # raw images in native space
            filename = data_dir + 'sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                       '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                       '_acq-' + acq[1] + '_run-' + str(run_no) + '_bold.nii.gz'
        else:                       # preprocessed images in space
            filename = deri_dir + 'fmriprep' + '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                       '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                       '_acq-' + acq[1] + '_run-' + str(run_no) + '_space-' + space + '_desc-' + desc[1] + '_bold.nii.gz'
        return filename
    
    # function: get "bold.gii" filenames
    #-------------------------------------------------------------------------#
    def get_bold_gii(self, run_no, hemi='L', space='fsnative'):
        """
        Get Filename for BOLD GIfTI Files
        filename = sess.get_bold_gii(run_no, hemi, space)
        
            run_no   - int; run number (e.g. 1)
            hemi     - string; brain hemisphere (e.g. "L")
            space    - string; image space (e.g. "fsnative")
            
            filename - string; filename of "bold.func.gii"
        
        filename = sess.get_bold_gii(run_no, hemi, space) returns the filename
        of the 4D GIfTI belonging to session sess, run run_no and brain hemi-
        sphere hemi.
        """
        
        # create filename
        filename = deri_dir + 'fmriprep' + \
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_hemi-' + hemi + '_space-' + space + '_bold.func.gii'
        return filename

    # function: get "events.tsv" filenames
    #-------------------------------------------------------------------------#
    def get_events_tsv(self, run_no):
        """
        Get Filename for Events TSV File
        filename = sess.get_events_tsv(run_no)
        
            run_no   - int; run number (e.g. 1)
            
            filename - string; filename of "events.tsv"
        
        filename = sess.get_events_tsv(run_no) returns the filename of the
        tab-separated events file belonging to session sess and run run_no.
        """
        
        # create filename
        filename = data_dir + 'sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_events.tsv'
        return filename

    # function: get "timeseries.tsv" filenames
    #-------------------------------------------------------------------------#
    def get_confounds_tsv(self, run_no):
        """
        Get Filename for Confounds TSV File
        filename = sess.get_confounds_tsv(run_no)
        
            run_no   - int; run number (e.g. 1)
            
            filename - string; filename of "timeseries.tsv"
        
        filename = get_confounds_tsv(run_no) returns the filename of the
        tab-separated confounds file belonging to session sess and run run_no.
        """
        
        # create filename
        filename = deri_dir + 'fmriprep' + \
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_desc-' + desc[2] + '_timeseries.tsv'
        return filename
    
    # function: get spike regressors "sub-{sub}_ses-{ses}_run-{run}_spikeregressor.csv" filenames
    #-------------------------------------------------------------------------#
    def get_spregressor_csv(self, run_no):
        """
        Get Filename for Confounds TSV File
        filename = sess.get_spregressor_csv(run_no)
        
            run_no   - int; run number (e.g. 1)
            
            filename - string; filename of "spikeregressor.csv"
        
        filename = get_spregressor_csv(run_no) returns the filename of the
        comma-separated regressor columns file belonging to session sess and run run_no.
        """
        
        # create filename
        filename = '/data/u_jeong_software' + '/EMPRISE' + '/code' + '/Python' +'/Tables' \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_run-' + str(run_no) + '_spikeregressor.csv'
        
        return filename

    # function: get mesh files
    #-------------------------------------------------------------------------#
    def get_mesh_files(self, space='fsnative', surface='inflated'):
        """
        Get Filenames for GIfTI Inflated Mesh Files
        mesh_files = sess.get_mesh_files(space, surface)
        
            space      - string; image space (e.g. "fsnative")
            surface    - string; surface image (e.g. "inflated")
            
            mesh_files - dict; filenames of inflated mesh files
            o left     - string; left hemisphere mesh file
            o right    - string; left hemisphere mesh file
        
        mesh_files = sess.get_mesh_files(space, surface) returns filenames for
        mesh files from specified image space and cortical surface to be used
        for surface plotting.
        """
        
        # if native image space
        if space == 'fsnative':
            
            # specify mesh files (New preprocessing - the name of inflated surface file is /sub-000_ses-visual_run-0_hemi-L_inflated.surf.gii)
            prep_dir  = deri_dir + 'fmriprep'
            mesh_path = prep_dir + '/sub-' + self.sub + '/anat' + \
                                   '/sub-' + self.sub + '*' + '_hemi-'
            mesh_file = mesh_path + 'L' + '_' + surface + '.surf.gii'
            if not glob.glob(mesh_file):
                for ses in sess:
                    mesh_path = prep_dir + '/sub-' + self.sub + '/ses-' + ses + '/anat' + \
                                           '/sub-' + self.sub +  '*' + '_hemi-'
                    mesh_file = mesh_path + 'L' + '_' + surface + '.surf.gii'
                    if glob.glob(mesh_file):
                        break
            if not glob.glob(mesh_file):
                mesh_files = {'left' : 'n/a', \
                              'right': 'n/a'}
            else:
                mesh_files = {'left' : glob.glob(mesh_path+'L'+'_'+surface +'.surf.gii')[0], \
                              'right': glob.glob(mesh_path+'R'+'_'+surface +'.surf.gii')[0]}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh dictionary
            mesh_dict = {'inflated':     'infl', \
                         'pial':         'pial', \
                         'white':        'white', \
                         'midthickness': 'midthick'}
            
            # specify mesh files
            if surface not in mesh_dict.keys():
                mesh_files = {'left' : 'n/a', \
                              'right': 'n/a'}
            else:
                free_dir   = deri_dir + 'freesurfer'
                mesh_path  = free_dir + '/fsaverage/' + mesh_dict[surface]
                mesh_files = {'left' : mesh_path + '_left.gii', \
                              'right': mesh_path + '_right.gii'}
        
        # return mesh files
        return mesh_files
    
    # function: get sulci files
    #-------------------------------------------------------------------------#
    def get_sulc_files(self, space='fsnative'):
        """
        Get Filenames for FreeSurfer Sulci Files
        sulc_files = sess.get_sulc_files(space)
        
            space      - string; image space (e.g. "fsnative")
            
            sulc_files - dict; filenames of FreeSurfer sulci files
            o left     - string; left hemisphere sulci file
            o right    - string; left hemisphere sulci file
        
        sulc_files = sess.get_sulc_files(space) returns filenames for FreeSurfer
        sulci files from specified image space to be used for surface plotting.
        """
        
        # if native image space
        if space == 'fsnative':
            
            # specify sulci files
            free_dir   = deri_dir + 'freesurfer'
            sulc_path  = free_dir + '/sub-' + self.sub + '/surf'
            sulc_files = {'left' : sulc_path + '/lh.sulc', \
                          'right': sulc_path + '/rh.sulc'}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh files
            free_dir   = deri_dir + 'freesurfer'
            sulc_path  = free_dir + '/fsaverage/sulc'
            sulc_files = {'left' : sulc_path + '_left.gii', \
                          'right': sulc_path + '_right.gii'}
        
        # return mesh files
        return sulc_files
    
        # function: get curv files
    #-------------------------------------------------------------------------#
    def get_curv_files(self, space='fsnative'):
        """
        Get Filenames for FreeSurfer Sulci Files
        curv_files = sess.get_curv_files(space)
        
            space      - string; image space (e.g. "fsnative")
            
            curv_files - dict; filenames of FreeSurfer curv files
            o left     - string; left hemisphere curv file
            o right    - string; right hemisphere curv file
        
        curv_files = sess.get_curv_files(space) returns filenames for FreeSurfer
        curv files from specified image space to be used for surface plotting.
        """
        
        # if native image space
        if space == 'fsnative':
            
            # specify sulci files
            free_dir   = deri_dir + 'freesurfer'
            curv_path  = free_dir + '/sub-' + self.sub + '/surf'
            curv_files = {'left' : curv_path + '/lh.curv', \
                          'right': curv_path + '/rh.curv'}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh files
            free_dir   = deri_dir + 'freesurfer'
            curv_path  = free_dir + '/fsaverage/curv'
            curv_files = {'left' : curv_path + '_left.gii', \
                          'right': curv_path + '_right.gii'}
        
        # return mesh files
        return curv_files
    
    #function: get valid runs 
    #-------------------------------------------------------------------------#
    def get_valid_runs(self):

        """

        valid_runs = sess.get_valid_runs(self, crs_val = False)

            valid_runs = { valid run :  its scan number} returns a dictionary.
            o key   - Valid run, integer
            o value - Scan numbers of a valid run, integer


        
        The choice of valid runs are dependent on the following criteria:
         1. The maximum number of scan is equal to the full scan numbers of an exepriment.
            (In EMPRISE7t case, 145)
         2. Only the runs have more than 3/4 of the maximum number of scan are valid runs
            to analyze.(An arbitarary choice for EMPRISE7t)
         3. Among all runs, there is at least one run which its scan number is equal to
           the maximum number of scan.
        """
        
        # extract list of exist runs and their scan number
        
        exist_runs = []
        scan_num = []
        for run in runs:
            filename = self.get_bold_nii(run, space='')
            if os.path.isfile(filename):
                Y = self.get_bold_nii(run, space='')
                Y = nib.load(Y)
                Y = Y.get_fdata()
                n = Y.shape[-1]
                scan_num.append(n)
                exist_runs.append(run)

        max_num = max(scan_num)
        
        # Runs that their scan number meets the criteria
        valid_runs = {}
        
        for j, run in enumerate(exist_runs):
            if scan_num[j] > max_num*3/4:
                valid_runs[run] = scan_num[j]

        return valid_runs        
           

    # function: load brain mask
    #-------------------------------------------------------------------------#
    def load_mask(self, run_no, space=''):
        """
        Load Brain Mask NIfTI File
        M = sess.load_mask(run_no, space)
        
            run_no - int; run number (e.g. 1)
            space  - string; image space (e.g. "T1w")
            
            M      - 1 x V vector; values of the mask image
        """
        
        # load image file
        filename = self.get_mask_nii(run_no, space)
        mask_nii = nib.load(filename)
        
        # extract mask image
        M = mask_nii.get_fdata()
        M = M.reshape((np.prod(M.shape),), order='C')
        return M
    
    # function: load fMRI data
    #-------------------------------------------------------------------------#
    def load_data(self, run_no, space=''):
        """
        Load Functional MRI NIfTI Files
        Y = sess.load_data(run_no, space)
            
            run_no - int; run number (e.g. 1)
            space  - string; image space (e.g. "T1w")
            
            Y      - n x V matrix; scan-by-voxel fMRI data
        """
        
        # load image file
        filename = self.get_bold_nii(run_no, space)
        bold_nii = nib.load(filename)
        
        # extract fMRI data
        Y = bold_nii.get_fdata()
        Y = Y.reshape((np.prod(Y.shape[0:-1]), Y.shape[-1]), order='C')
        Y = Y.T
        return Y

    # function: load fMRI data (all runs)
    #-------------------------------------------------------------------------#
    def load_data_all(self, space=''):
        """
        Load Functional MRI NIfTI Files from All Runs
        Y = sess.load_data_all(space)
            
            space - string; image space (e.g. "T1w")
            
            Y     - n x V x r array; scan-by-voxel-by-run fMRI data
        """
        
        """         
            # prepare 3D array
            for j, run in enumerate(runs):
                filename = self.get_bold_nii(run, space)
                if os.path.isfile(filename):
                    Y = self.load_data(run, space)
                    break
            Y = np.zeros((Y.shape[0], Y.shape[1], len(runs)))
        
            # load fMRI data
            for j, run in enumerate(runs):
                filename = self.get_bold_nii(run, space)
                if os.path.isfile(filename):
                    Y[:,:,j] = self.load_data(run, space)
        
            # select available runs
            Y = Y[:,:,np.any(Y, axis=(0,1))]
            return Y """
    
        # perparing 3D array 
        
        VR = self.get_valid_runs()
        runs = list(VR.keys())

        # get n
        n = min(VR.values())
        
        # get V (assumption: all runs have the same spacial dimension)
        Y = self.load_data(runs[0],space)
        
        # 3D array with zeros
        Y = np.zeros((n, Y.shape[1],len(VR)))

        #load fMRI data
        for j, run in enumerate(runs):
            filename = self.get_bold_nii(run, space)
            if os.path.isfile(filename):
                Y[:,:,j] = self.load_data(run, space)[:n, :]

        return Y

    # function: load surface fMRI data
    #-------------------------------------------------------------------------#
    def load_surf_data(self, run_no, hemi='L', space='fsnative'):
        """
        Load Functional MRI GIfTI Files
        Y = sess.load_surf_data(run_no, hemi, space)
            
            run_no - int; run number (e.g. 1)
            hemi   - string; brain hemisphere (e.g. "L")
            space  - string; image space (e.g. "fsnative")
            
            Y      - n x V matrix; scan-by-vertex fMRI data
        """
        
        # load image file
        filename = self.get_bold_gii(run_no, hemi, space)
        bold_gii = nib.load(filename)
        
        # extract fMRI data
        Y = np.array([y.data for y in bold_gii.darrays])
        return Y
    
    # function: load surface fMRI data (all runs)
    #-------------------------------------------------------------------------#
    def load_surf_data_all(self, hemi='L', space='fsnative'):
        """
        Load Functional MRI GIfTI Files from All Runs
        Y = sess.load_surf_data_all(hemi, space)
            
            hemi  - string; brain hemisphere (e.g. "L")
            space - string; image space (e.g. "fsnative")
            
            Y     - n x V x r array; scan-by-vertex-by-run fMRI data
        """
        
        # Collect valid runs and their scan numbers
        valid_runs = []
        r_scan = []
        for run in runs:
            filename = self.get_bold_gii(run, hemi, space)
            if os.path.isfile(filename):
                Y = self.load_surf_data(run, hemi, space)
                valid_runs.append(run)
                r_scan.append(Y.shape[0])
        
        # Determine minimum scan number
        min_scans = min(r_scan)
        
        # Filter out runs with insufficient scans
        if self.ses in ['visual','audio','audio1','digit','spoken']:
            if min_scans <= int(n*0.75) : # EMPRISE: 108 EMPRISE2: 156 (3/4 of the scans of an entire run)
                valid_runs = [run for run, scans in zip(valid_runs, r_scan) if scans > int(n*0.75)]
                r_scan = [scans for scans in r_scan if scans > int(n*0.75)]
                min_scans = min(r_scan) if r_scan else 0

        elif self.ses in ['audio2']:
            if min_scans <= int(n2*0.75) : #EMPRISE2 audio2: 142 (3/4 of the scans of an entire run)
                valid_runs = [run for run, scans in zip(valid_runs, r_scan) if scans > int(n2*0.75)]
                r_scan = [scans for scans in r_scan if scans > int(n2*0.75)]            
                min_scans = min(r_scan) if r_scan else 0

        
        # Prepare output array
        if not r_scan:
            return None
        
        Y = np.zeros((min_scans, Y.shape[1], len(r_scan)))
        
        # Load data
        for j, run in enumerate(valid_runs):
            Y[:,:,j] = self.load_surf_data(run, hemi, space)[:min_scans,:]
        
        # Remove empty runs
        Y = Y[:,:,np.any(Y, axis=(0,1))]
    
        
        return Y
    
    # function: get onsets and durations
    #-------------------------------------------------------------------------#
    def get_onsets(self, valid_runs, filenames=None):
        """
        Get Onsets and Durations for Single Subject and Session, all Runs
        ons, dur, stim = sess.get_onsets(valid_runs,filenames)
        
            valid_runs- list of integers; from 1 to 9, valid run numbers
            filenames - list of strings; "events.tsv" filenames
        
            ons       - list of arrays of floats; t x 1 vectors of onsets [s]
            dur       - list of arrays of floats; t x 1 vectors of durations [s]
            stim      - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            
        ons, dur, stim = sess.get_onsets(valid_runs,filenames) loads the "events.tsv" file
        belonging to session sess and returns lists of length number of runs,
        containing, as each element, lists of length number of trials per run,
        containing onsets and durations in seconds as well as stimuli in
        numerosity.
        """
        
        # prepare onsets, durations, stimuli as empty lists
        ons  = []
        dur  = []
        stim = []
        
        # prepare labels for trial-wise extraction
        if self.ses == 'visual':
            stimuli = {'1_dot': 1, '2_dot': 2, '3_dot': 3, '4_dot': 4, '5_dot': 5, \
                        '20_dot': 20}
        elif self.ses == 'digits':
            stimuli = {'1_digit': 1, '2_digit': 2, '3_digit': 3, '4_digit': 4, '5_digit': 5, \
                       '20_digit': 20}
        elif self.ses in ['audio','audio1','audio2','spoken']:
            stimuli = {'1_audio': 1, '2_audio': 2, '3_audio': 3, '4_audio': 4, '5_audio': 5, \
                        '20_audio': 20}
        elif self.ses == 'congruent' or self.ses == 'incongruent':
            stimuli = {              '2_mixed': 2, '3_mixed': 3, '4_mixed': 4, '5_mixed': 5, '20_mixed': 20}
        
        # for all runs
        for j, run in enumerate(valid_runs):
            
            # extract filename
            if filenames is None:
                filename = self.get_events_tsv(run)
            else:
                filename = filenames[j]
            
            # if onset file exists
            if os.path.isfile(filename):
                validity_check = self.get_confounds_tsv(run)
                if os.path.isfile(validity_check):
                    print('\n     run ', run, ' is included to analysis.')
                    # extract events of interest
                    events = pd.read_csv(filename, sep='\t')
                    events = events[events['trial_type']!='button_press']
                    for code in stimuli.keys():
                        events.loc[events['trial_type']==code+'_attn','trial_type'] = code
                    
                    # save onsets, durations, stimuli
                    stims = [stimuli[trl] for trl in events['trial_type']]
                    ons.append(np.array(events['onset']))
                    dur.append(np.array(events['duration']))
                    stim.append(np.array(stims))
                
        # return onsets
        return ons, dur, stim

    # function: get confound variables
    #-------------------------------------------------------------------------#
    def get_confounds(self, labels, filenames=None):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_c = sess.get_confounds(labels, filenames)
        
            labels    - list of strings; confound file header entries
            filenames - list of strings; "timeseries.tsv" filenames
            
            X_c       - n x c x r array; confound variables
                       (n = scans, c = variables, r = runs)
            valid_run - list of int; valid runs
        
        X_c = sess.get_confounds(filenames) loads the "timeseries.tsv" file
        belonging to session sess and returns a scan-by-variable-by-run array
        of those confound variables indexed by the list labels. The function
        applies no preprocessing to the confounds. In addition to the confounds,
        a list of valid runs created to use to check number of scans in a run
        then take only runs having more than three quater of the planed scan n.
        """
        
        # prepare confound variables as zero matrix
        c = len(labels)
        
        # Track valid runs and their scan numbers
        valid_runs = []
        run_scans = []
        
        # First pass: determine minimum scan number
        for j, run in enumerate(runs):
            if filenames is None:
                filename = self.get_confounds_tsv(run)
            else:
                filename = filenames[j]
            
            if os.path.isfile(filename):
                confounds = pd.read_csv(filename, sep='\t')
                valid_runs.append(run)
                run_scans.append(len(confounds))
        
        # Determine minimum scan number
        min_scans = min(run_scans)
        
        # AUdio2 has irregular scan numbers, but the onsets lasted for 191 scans.
        if self.ses in ['audio2'] and min_scans > n2:
            min_scans = 191
            
        
        # Filter out runs with insufficient scans
        if self.ses in ['visual','audio','audio1','digit','spoken']:
            if min_scans <= int(n*0.75):
                valid_runs = [run for run, scans in zip(valid_runs, run_scans) if scans > int(n*0.75)]
                run_scans = [scans for scans in run_scans if scans > int(n*0.75)]
                min_scans = min(run_scans) if run_scans else 0
        elif self.ses in ['audio2']:
            if min_scans <= int(n2*0.75):
                valid_runs = [run for run, scans in zip(valid_runs, run_scans) if scans > int(n2*0.75)]
                run_scans = [scans for scans in run_scans if scans > int(n2*0.75)]
                min_scans = min(run_scans) if run_scans else 0
        
        # Prepare output array
        if not run_scans:
            return None
        
        # Initialize confound matrix with minimum scan number
        X_c = np.zeros((min_scans, c, len(valid_runs)))
        
        # Second pass: load confounds
        for j, run in enumerate(valid_runs):
            if filenames is None:
                filename = self.get_confounds_tsv(run)
            else:
                filename = filenames[j]
            
            confounds = pd.read_csv(filename, sep='\t')
            for k, label in enumerate(labels):
                X_c[:,k,j] = np.array(confounds[label][:min_scans])
        
        # select available runs
        X_c = X_c[:,:,np.any(X_c, axis=(0,1))]
        
        return X_c, valid_runs      
    

        # function: get aCompCor confound variables (white_matter + csf)
    #-------------------------------------------------------------------------#
    def get_acom_confounds(self):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_acom_list = sess.get_acom_confounds()
        
            X_acom_list     - list with length r; each item is n x c array; confound variables
                        explaining over 50% of variance in white matter and csf signals
                     (n = scans, c = variables, r = runs)
            
        X_acom_list = sess.get_acom_confounds() loads the "timeseries.tsv" file belonging
        to session sess and returns a scan-by-variable array of those
        confound variables indexed by the list labels and make a list which its length is the numbers of runs. 
        The function applies no preprocessing to the confounds.
        """


        # prepare confound variables as zero matrix
        VR = self.get_valid_runs()
        runs = list(VR.keys())
        r = len(runs)
        n = min(VR.values())

        # initialize X_c_list as list
        X_acom_list= [] 

        #for all valid runs
        for j, run in enumerate(runs):

            #save confound variables
            filename = self.get_confounds_tsv(run)
            confounds = pd.read_csv(filename, sep='\t')
            labels = list(confounds.columns.values)
            pattern = "a_comp_cor_" + ".*"
            labels = [label for label in labels if re.match(pattern, label)]
            c = len(labels)
            X_c = np.zeros((n,c)) # initialize X_c as 2D array   
            X_c[:, :] = np.array(confounds[labels][:n]) # Fill confound data
            X_acom_list.append(X_c)     

        return X_acom_list     

    # function: get spike regressor variables
    #-------------------------------------------------------------------------#
    def get_spregressor(self, threshold = None):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_c_list = sess.get_confounds(threshold)
        
            threshold  - regressor file header entries
            
            X_c_list   - a list of n x c arrays which its length is r; confound variables
                     (n = scans, c = variables, r = runs)
            
        X_c_list = sess.get_spregressor() loads the "spikeregressor.csv" file belonging
        to session sess and returns a list of scan-by-variable array along its run of those
        confound variables indexed by the list labels. The function applies
        no preprocessing to the confounds.
        """
        # get valid runs
        VR = self.get_valid_runs()

        # prepare confound variables as zero matrix
        runs = list(VR.keys())
        r = len(runs)
        n = min(VR.values()) 
        
        X_c_list= [] # initialize X_c_list as list

        # For all runs
        for j, run in enumerate(runs):
            # Get confound labels and data
            filename = self.get_spregressor_csv(run)
            confounds = pd.read_csv(filename)
            labels = list(confounds.columns.values)
            pattern = "th_" + str(threshold) + ".*"
            labels = [label for label in labels if re.match(pattern, label)]
            c = len(labels)
            X_c = np.zeros((n,c)) # initialize X_c as 2D array   
            X_c[:, :] = np.array(confounds[labels][:n]) # Fill confound data
            # Check if there is null column, if so, remove it
            zero_columns = np.all(X_c == 0, axis=0)
            X_c = X_c[:, ~zero_columns]
            X_c_list.append(X_c)     

        return X_c_list

# class: subject/session
#-----------------------------------------------------------------------------#
class Model(Session):
    """
    A Model object is initialized by subject/session/space IDs and model name
    and allows for multiple operations related to numerosity estimation.
    """
    
    # function: initialize model
    def __init__(self, subj_id, sess_id, mod_name, space_id='fsnative'):
        """
        Initialize a Model applied to a Session
        mod = EMPRISE.Model(subj_id, sess_id, mod_name, space_id)
        
            subj_id  - string; subject identifier (e.g. "001")
            sess_id  - string; session identifier (e.g. "visual")
            mod_name - string; name for the model (e.g. "NumAna")
            space_id - string; space identifier (e.g. "fsnative")

            mod      - a Session object
            o sub    - the subject ID
            o ses    - the session ID
            o model  - the model name
            o space  - the space ID
            
        """
        
        # store subject/session/space IDs and model name
        super().__init__(subj_id, sess_id)  # inherit parent class
        self.model = mod_name               # configure child object
        self.space = space_id

    # function: model directory
    #-------------------------------------------------------------------------#
    def get_model_dir(self):
        """
        Get Folder Name for Model
        mod_dir = mod.get_model_dir()
        
            mod_dir - string; directory where the model is saved
        """
        
        # create folder name
        nprf_dir = deri_out + 'numprf'
        mod_dir  = nprf_dir + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.model
        return mod_dir
    
    # function: results file
    #-------------------------------------------------------------------------#
    def get_results_file(self, hemi='L', fold='all'):
        """
        Get Results Filename for Model
        res_file = mod.get_results_file(hemi)
        
            hemi     - string; brain hemisphere (e.g. "L")
            fold     - string; data subset used ("all", "odd" or "even" runs)
        
            res_file - string; results file into which the model is written
        """
        
        # create filename
        mod_dir  = self.get_model_dir()
        filepath = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + '_model-' + self.model + \
                             '_hemi-' + hemi + '_space-' + self.space + '_'
        if fold in ['odd', 'even']:
            filepath = filepath + 'runs-' + fold + '_'
        res_file = filepath + 'numprf.mat'
        return res_file
    
    # function: calculate runs/scans
    #-------------------------------------------------------------------------#
    def calc_runs_scans(self, fold='all'):
        """
        Calculate Number of Runs and Scans
        r0, n0 = mod.calc_runs_scans(fold)
        
            fold - string; data subset used ("all", "odd", "even" runs or "cv")
        
            r0   - int; number of runs analyzed, depending on averaging across runs
            n0   - int; number of scans per run, depending on averaging across epochs
        """
        
        # load results file
        res_file = self.get_results_file('L')
        NpRF     = sp.io.loadmat(res_file)
        
        # count number of runs
        r0  = 0
        for run in runs:
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):
                if (fold == 'all') or (fold == 'cv') or (fold == 'odd'  and run % 2 == 1) or (fold == 'even' and run % 2 == 0):
                    r0 = r0 + 1
        # Explanation: This is the number of available runs. Usually, there
        # are 8 runs, but in case of removed data, there can be fewer runs.
        
        # get number of scans
        avg = list(NpRF['settings']['avg'][0,0][0,:])
        # Explanation: This extracts averaging options from the model settings.
        r0  = [r0,1][avg[0]]
        # Explanation: If averaging across runs, there is only 1 (effective) run.
        if self.ses in ['visual','audio','audio1','digit','spoken']:
            n0  = [n,scans_per_epoch][avg[1]]
        elif self.ses in ['audio2']:
            n0 =  [n2,scans_per_epoch2][avg[1]]
        #Explanation: If averaging across epochs, there are only 52/46 (effective) scans.
        
        # return runs and scans
        return r0, n0
    
    # function: load in-mask data
    #-------------------------------------------------------------------------#
    def load_mask_data(self, hemi='L'):
        """
        Load Functional MRI GIfTI Files and Mask
        Y = sess.load_mask_data(hemi, space)
            
            hemi  - string; brain hemisphere (e.g. "L")
            
            Y     - n x v x r array; scan-by-vertex-by-run fMRI data
        """
        
        # load and mask data
        Y = self.load_surf_data_all(hemi, self.space)
        M = np.all(Y, axis=(0,2))
        Y = Y[:,M,:]
        
        # return data and mask
        return Y, M
    
    # function: analyze numerosities
    #-------------------------------------------------------------------------#
    def analyze_numerosity(self, avg=[True, False], corr='iid', order=1, ver='V2', st_wise = False, hemis = ['L','R'], sh=False, CST = False):
        """
        Estimate Numerosities and FWHMs for Surface-Based Data
        results = mod.analyze_numerosity(avg, corr, order, ver, sh)
        
            avg     - list of bool; see "NumpRF.estimate_MLE" (default: [True, False])
            corr    - string; see "NumpRF.estimate_MLE" (default: "iid")
            order   - int; see "NumpRF.estimate_MLE" (default: 1)
            ver     - string; version identifier (default: "V2")
            st_wise - bool; block wise experimental desgin (False) or stimuli wise experimental design (True) (default: False)
                      audio2 session has three different options, ['audio2_block', 'audio2_stim', 'audio2_seq']
                      audio1 session has two options ['audio1','audio1_seq']
            hemis   - list of string; ['L','R'] or ['L'] or ['R'] (default: ['L','R'])
            sh      - bool; split-half estimation (default: False)
            CST     - bool; include compressive spatio-temporal neuronal activity model or not (default: False)
            
            results - dict of dicts; results filenames
            o L     - results for left hemisphere
            o R     - results for right hemisphere
            
        results = mod.analyze_numerosity(avg, corr, order, ver, sh) loads the
        surface-based pre-processed data belonging to model mod, estimates
        tuning parameters using settings avg, corr, order, ver, sh and saves
        results into a single-subject results directory.
        
        The input parameter "sh" (default: False) specifies whether parameters
        are estimated in a split-half sense (if True: separately for odd and
        even runs) or across all runs (if False: across all available runs).
        
        The input parameter "ver" (default: "V2") controls which version of
        the routine is used (for details, see "NumpRF.estimate_MLE"):
            V0:       mu_grid   = [3, 1]
                      fwhm_grid = [10.1, 5] (see "NumpRF.estimate_MLE_rgs")
            V1:       mu_grid   = {0.05,...,6, 10,20,...,640,1280} (128)
                      fwhm_grid = {0.3,...,18, 24,48,96,192} (64)
            V2:       mu_grid   = {0.8,...,9.2, 20} (170) 
                      sig_grid  = {0.05,...,5} (100)
            V2-lin:   mu_grid   = {0.8,...,9.2, 20} (170)
                      sig_grid  = {0.05,...,5} (100)
        
        Note1: "sig_grid" is calculated into FWHM values, if ver is "V2", and
        into linear sigma values, if ver is "V2-lin" (see "NumpRF.estimate_MLE").
        
        Note2: "analyze_numerosity" uses the results dictionary keys "L" and "R"
        which are identical to the hemisphere labels used by fMRIprep.
        
        Note3: st_wise is added to analyse audiotry stimuli. 
               st_wise =  False is assuming the simultaneous activities in neuronal
                          population when numerosity information is given at a time point 
                          (e.g. visual dots)
               st_wise = 'audio2_block' reflects the long pause between numerosity blocks 
                          (6 times repeted unit tones)   
               st_wise = 'audio1', 'audio2_stim' consider each unit tones 
                          (e.g. for 2 condtion play time of 2 tones are duration)
               st_wise = 'audio1_seq', 'audio2_seq' conunt how many times a tone played sequentially.
                          (e.g. expected neuronal activity assgiend to a tone following its played order
                           for a condition 2, the second tone has the full activity and the first one
                           has lower than the second one following the log gaussian model)
        """
        
        # part 1: load subject data
        #---------------------------------------------------------------------#
        print('\n\n-> Subject "{}", Session "{}":'.format(self.sub, self.ses))
        mod_dir = self.get_model_dir()
        if not os.path.isdir(mod_dir): os.makedirs(mod_dir)
        
        # load confounds
        print('   - Loading confounds ... ', end='')
        X_c, v_run = self.get_confounds(covs)
        X_c          = standardize_confounds(X_c)
        print(f'    - Valid runs:{v_run}')
        print('successful!')
        
        # load onsets
        print('   - Loading onsets ... ', end='')
        ons, dur, stim = self.get_onsets(v_run)
        if not st_wise:
            ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        elif st_wise:
            ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = st_wise)
            print( f'\n    Stimuli-wise analysis: {st_wise} \n')

        print('successful!')
        
      
        
        # specify grids
        if CST:
            tau_grid = range(4,51,1)
            epn_grid = [0, 0.25, 0.5, 1]
        
        if ver == 'V0':
            mu_grid   = [ 3.0, 1.0]
            fwhm_grid = [10.1, 5.0]  
        elif ver == 'V1':
            mu_grid   = np.concatenate((np.arange(0.05, 6.05, 0.05), \
                                        10*np.power(2, np.arange(0,8))))
            fwhm_grid = np.concatenate((np.arange(0.3, 18.3, 0.3), \
                                        24*np.power(2, np.arange(0,4))))
        elif ver == 'V2' or ver == 'V2-lin':
            mu_grid   = np.concatenate((np.arange(0.80, 5.25, 0.05), \
                                        np.array([20]))) # EMPRISE np.arange(0.80, 5.25, 0.05) 
            sig_grid  = np.arange(0.05, 3.05, 0.05) # EMPRISE np.arange(0.05, 3.05, 0.05)
        else:
            err_msg = 'Unknown version ID: "{}". Version must be "V0" or "V1" or "V2"/"V2-lin".'
            raise ValueError(err_msg.format(ver))
        
        # specify folds
        if not sh: folds = {'all': []}        # all runs vs. split-half
        else:      folds = {'odd': [], 'even': []}
        for idx, run in enumerate(v_run):     # for all possible runs
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):      # if data from this run exist
                if not sh:
                    folds['all'].append(idx)  # add slice to all runs
                else:
                    if idx % 2 == 1:          # add slice to odd runs
                        folds['odd'].append(idx)
                    else:                     # add slice to even runs
                        folds['even'].append(idx)
        
        # part 2: analyze both hemispheres
        #---------------------------------------------------------------------#
        #hemis   = ['L', 'R'] # hemis is given as input variable
        results = {}
        for hemi in hemis:
            
            
            # load data
            print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, self.space))
            print('   - Loading fMRI data ... ', end='')
            Y, M  = self.load_mask_data(hemi)
            Y     = standardize_signals(Y)
            V     = M.size
            print('successful!')
            print('\n-> Compare the temporal dimension.')
            
            t_dif = X_c.shape[0]- Y.shape[0] 
            if t_dif == 0:
                print('   - Dimension matched.')
            elif t_dif == 1:
                X_c = X_c[1:,:,:] # cut the t=1 confound since we dropped the first scan
                print('   - Cut the first time point confound.')
            elif t_dif in [2,3,4,5]: # Some fundtional runs scanned over the valid experimental duration
                print('   -Temporal dimension of BOLD and confounds are not matched: \n')
                print(f'   -Bold: {Y.shape[0]} \n Cov: {X_c.shape[0]}')
                X_c = X_c[1:(X_c.shape[0]-t_dif)+1,:,:]
            
                
            # analyze all folds
            results[hemi] = {}
            for fold in folds:
                
                # if fold contains runs
                num_runs = len(folds[fold])
                if num_runs > 0:
                
                    # get fold data
                    print('\n-> Runs "{}" ({} run{}: slice{} {}):'. \
                          format(fold, num_runs, ['','s'][int(num_runs>1)], \
                                 ['','s'][int(num_runs>1)], ','.join([str(i) for i in folds[fold]])))
                    print('   - Estimating parameters ... ', end='\n')
                    Y_f    = Y[:,:,folds[fold]]
                    ons_f  = [ons[i]  for i in folds[fold]]
                    dur_f  = [dur[i]  for i in folds[fold]]
                    stim_f = [stim[i] for i in folds[fold]]
                    Xc_f   = X_c[:,:,folds[fold]]
                    
                    # analyze data
                    ds = NumpRF_audio.DataSet(Y_f, ons_f, dur_f, stim_f, TR, Xc_f)
                    start_time = time.time()
                    if ver == 'V0':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_MLE_rgs(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                    elif ver == 'V1':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                    elif ver == 'V2' and not CST:
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, lin=False)
                    elif ver == 'V2' and CST:
                        mu_est, fwhm_est, beta1_est, beta2_est, beta3_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_CST_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, tau_grid=tau_grid, epn_grid=epn_grid, lin=False)
                    elif ver == 'V2-lin' and not CST:
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, lin=True)
                    elif ver == 'V2-lin' and CST:
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                            ds.estimate_CST_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, tau_grid=tau_grid, epn_grid=epn_grid, lin=True)
                    if True:
                        k_est, k_null, k_const = \
                            ds.free_parameters(avg, corr, order)
                    end_time   = time.time()
                    difference = end_time - start_time
                    del start_time, end_time
                    
                    # save results (mat-file)
                    sett = str(avg[0])+','+str(avg[1])+','+str(corr)+','+str(order)
                    print('\n-> Runs "{}", Model "{}", Settings "{}":'.
                          format(fold, self.model, sett))
                    print('   - Saving results file ... ', end='')
                    filepath = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + \
                                          '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
                    if sh: filepath = filepath + 'runs-' + fold + '_'
                    results[hemi][fold] = filepath + 'numprf.mat'
                    res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr, 'order': order}, \
                                'mu_est':  mu_est,  'fwhm_est': fwhm_est, 'beta_est':  beta_est, \
                                'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
                                'k_est':   k_est,   'k_null':   k_null,   'k_const':   k_const, \
                                'corr_est':corr_est,'version':  ver,      'time':      difference}
                    sp.io.savemat(results[hemi][fold], res_dict)
                    print('successful!')
                    del sett, res_dict
                    
                    # save results (surface images)
                    para_est = {'mu': mu_est, 'fwhm': fwhm_est, 'beta': beta_est}
                    for name in para_est.keys():
                        print('   - Saving "{}" image ... '.format(name), end='')
                        para_map    = np.zeros(V, dtype=np.float32)
                        para_map[M] = para_est[name]
                        surface     = nib.load(self.get_bold_gii(v_run[0],hemi,self.space))
                        filename    = filepath + name + '.surf.gii'
                        para_img    = save_surf(para_map, surface, filename)
                        print('successful!')
                    del para_est, para_map, surface, filename, para_img
        
        # return results filename
        return results
    
    # function: calculate R-squared maps
    #-------------------------------------------------------------------------#
    def calculate_Rsq(self, folds=['all', 'odd', 'even', 'cv'], st_wise = False):
        """
        Calculate R-Squared Maps for Numerosity Model
        maps = mod.calculate_Rsq(folds)
        
            folds - list of strings; runs for which to calculate
            
            maps  - dict of dicts; calculated R-squared maps
            o all  - dict of strings; all runs
            o odd  - dict of strings; odd runs
            o even - dict of strings; even runs
            o cv   - dict of strings; cross-validated R-squared
              o left  - R-squared map for left hemisphere
              o right - R-squared map for right hemisphere
            o st_wise - bool; block wise or stimuli wise experimental design (default: False)
        
        maps = mod.calculate_Rsq(folds) loads results from numerosity analysis
        and calculates R-squared maps for all runs in folds.
        
        Note: "calculate_Rsq" uses the results dictionary keys "left" and "right"
        which are identical to the hemisphere labels used by surfplot.
        """
        
        # part 1: prepare calculations
        #---------------------------------------------------------------------#
        print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(self.sub, self.ses, self.model))
        mod_dir = self.get_model_dir()
        
        # specify slices
        i = -1                              # slice index (3rd dim)
        slices = {'all': [], 'odd': [], 'even': []}
        X_c, v_run = self.get_confounds(covs)
        
        for idx, run in enumerate(v_run):                    # for all possible runs
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):    # if data from this run exist
                i = i + 1                   # increase slice index
                slices['all'].append(i)
                if idx % 2 == 1: slices['odd'].append(idx)
                else:            slices['even'].append(idx)
        
        # part 2: analyze both hemispheres
        #---------------------------------------------------------------------#
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {}
        for fold in folds: maps[fold] = {}
        
        # for both hemispheres
        for hemi in hemis.keys():
            print('   - {} hemisphere:'.format(hemis[hemi]))
            
            # for all folds
            for fold in folds:
                
                # load analysis results
                filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                     '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
                if fold in ['odd', 'even']:
                    filepath = filepath + 'runs-' + fold + '_'
                res_file = filepath + 'numprf.mat'
                mu_map   = filepath + 'mu.surf.gii'
                NpRF     = sp.io.loadmat(res_file)
                surface  = nib.load(mu_map)
                mask     = surface.darrays[0].data != 0
                
                # calculate R-squared (all, odd, even)
                avg   = list(NpRF['settings']['avg'][0,0][0,:])
                MLL1  = np.squeeze(NpRF['MLL_est'])
                MLL00 = np.squeeze(NpRF['MLL_const'])
                r0,n0 = self.calc_runs_scans(fold)
                n1    = r0*n0
                Rsq   = NumpRF_audio.MLL2Rsq(MLL1, MLL00, n1)
                
                # calculate R-squared (cross-validated)
                if fold == 'cv':
                    
                    # load session data
                    print('     - Calculating split-half cross-validated R-squared ... ')
                    Y, M           = self.load_mask_data(hemi)
                    Y              = standardize_signals(Y)
                    X_c, v_run     = self.get_confounds(covs)
                    X_c            = standardize_confounds(X_c)
                    t_dif = abs(X_c.shape[0]- Y.shape[0]) 
                    if t_dif == 0:
                        print('   - Dimension matched.')
                    elif t_dif == 1:
                        X_c = X_c[1:,:,:] # cut the t=1 confound since we dropped the first scan
                        print('   - Cut the first time point confound.')
                    elif t_dif in [2,3,4,5]: # Some fundtional runs scanned over the valid experimental duration
                        print('   -Temporal dimension of BOLD and confounds are not matched: \n')
                        print(f'   -Bold: {Y.shape[0]} \n Cov: {X_c.shape[0]}')
                        X_c = X_c[1:(X_c.shape[0]-t_dif)+1,:,:]
            
                    
                    # load onset, duration, stimuli
                    ons, dur, stim = self.get_onsets(v_run)
                    if not st_wise:
                        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
                    elif st_wise:   
                        ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = st_wise)
                            
                    # cycle through CV folds
                    sets   = ['odd', 'even']
                    oosRsq = np.zeros((len(sets),Rsq.size))
                    for i in range(len(sets)):
                        
                        # load parameters from this fold
                        res_file = self.get_results_file(hemi, sets[i])
                        NpRF     = sp.io.loadmat(res_file)
                        mu1      = np.squeeze(NpRF['mu_est'])
                        fwhm1    = np.squeeze(NpRF['fwhm_est'])
                        beta1    = np.squeeze(NpRF['beta_est'])
                        
                        # get data from the other fold
                        xfold  = sets[1-i]
                        Y2     = Y[:,:,slices[xfold]]
                        ons2   = [ons[i]  for i in slices[xfold]]
                        dur2   = [dur[i]  for i in slices[xfold]]
                        stim2  = [stim[i] for i in slices[xfold]]
                        Xc2    = X_c[:,:,slices[xfold]]
                        
                        # obtain fit across folds
                        ds          = NumpRF_audio.DataSet(Y2, ons2, dur2, stim2, TR, Xc2)
                        oosRsq[i,:] = ds.calculate_Rsq(mu1, fwhm1, beta1, avg)
                        
                    # calculate cross-validated R-squared
                    Rsq = np.mean(oosRsq, axis=0)
                    print()
                
                # threshold tuning maps
                print('     - Saving R-squared image for {} runs ... '.format(fold), end='')
                para_map       = np.zeros(mask.size, dtype=np.float32)
                para_map[mask] = Rsq
                if fold in ['all', 'odd', 'even']:
                    filename   = filepath + 'Rsq.surf.gii'
                else:
                    filename   = filepath + 'cvRsq.surf.gii'
                para_img       = save_surf(para_map, surface, filename)
                maps[fold][hemis[hemi]] = filename
                print('successful!')
                del para_map, surface, filename, para_img
                
        # return results filename
        return maps
    # function: average signals
#-----------------------------------------------------------------------------#
def average_signals(Y, t=None, avg=[True, False]):
    """
    Average Signals Measured during EMPRISE Task
    Y, t = average_signals(Y, t, avg)
    
        Y   - n x v x r array; scan-by-voxel-by-run signals
        t   - n x 1 vector; scan-wise fMRI acquisition times
        avg - list of bool; indicating whether signals are averaged (see below)
        
        Y   - n0 x v x r array; if averaged across epochs OR
              n  x v matrix; if averaged across runs OR
              n0 x v matrix; if averaged across runs and epochs (n0 = scans per epoch)
        t   - n0 x 1 vector; if averaged across epochs OR
              n  x 1 vector; identical to input otherwise
    
    Y, t = average_signals(Y, t, avg) averages signals obtained with the 
    EMPRISE experiment across either runs, or epochs within runs, or both.
    
    If the input variable "t" is not specified, it is automatically set to
    the vector [0, 1*TR, 2*TR, ..., (n-2)*TR, (n-1)*TR].
    
    The input variable "avg" controls averaging. If the first entry of avg is
    true, then signals are averaged over runs. If the second entry of avg is
    true, then signals are averaged over epochs within runs. If both are
    true, then signals are first averaged over runs and then epochs. By
    default, only the first entry is true, causing averaging across runs.
    """
    
    # create t, if necessary
    if t is None:
        t = np.arange(0, n*TR, TR)
    
    # average over runs
    if avg[0]:
        
        # if multiple runs
        if len(Y.shape) > 2:
            Y = np.mean(Y, axis=2)
    
    # average over epochs
    if avg[1]:
        
        # remove discard scans
        Y = Y[num_scan_disc:]
        loc_scans_per_epoch = int(Y.shape[0]/num_epochs) # There is a global variable but redfined here for flexibility.
        
        
        # if averaged over runs
        if len(Y.shape) < 3:
            Y_epochs = np.zeros((loc_scans_per_epoch,Y.shape[1],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,i] = Y[(i*loc_scans_per_epoch):((i+1)*loc_scans_per_epoch),:]
            Y = np.mean(Y_epochs, axis=2)
        
        # if not averaged over runs
        else:
            Y_epochs = np.zeros((loc_scans_per_epoch,Y.shape[1],Y.shape[2],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,:,i] = Y[(i*loc_scans_per_epoch):((i+1)*loc_scans_per_epoch),:,:]
            Y = np.mean(Y_epochs, axis=3)
        
        # correct time vector
        t = t[num_scan_disc:]
        t = t[:loc_scans_per_epoch] - num_scan_disc*TR
    
    # return averaged signals
    return Y, t

# function: standardize signals
#-----------------------------------------------------------------------------#
def standardize_signals(Y, std=[True, True]):
    """
    Standardize Measured Signals for ReML Estimation
    Y = standardize_signals(Y, std)
    
        Y   - n x v x r array; scan-by-voxel-by-run signals
        std - list of bool; indicating which operations to perform (see below)
    
    Y = standardize_signals(Y, std) standardizes signals, i.e. it sets the mean
    of each time series (in each run) to 100, if the first entry of std is
    true, and scales the signal to percent signal change (PSC), if the second
    entry of std is true. By default, both entries are true.
    """
    
    # if Y is a 2D matrix
    if len(Y.shape) < 3:
        for k in range(Y.shape[1]):
            mu     = np.mean(Y[:,k])
            Y[:,k] = Y[:,k] - mu
            if std[1]:
                Y[:,k] = Y[:,k]/mu * 100
            if std[0]:
                Y[:,k] = Y[:,k] + 100
            else:
                Y[:,k] = Y[:,k] + mu
    
    # if Y is a 3D array
    else:
        for j in range(Y.shape[2]):
            for k in range(Y.shape[1]):
                mu      = np.mean(Y[:,k,j])
                Y[:,k,j] = Y[:,k,j] - mu
                if std[1]:
                    Y[:,k,j] = Y[:,k,j]/mu * 100
                if std[0]:
                    Y[:,k,j] = Y[:,k,j] + 100
                else:
                    Y[:,k,j] = Y[:,k,j] + mu
    
    # return standardized signals
    return Y

# function: standardize confounds
#-----------------------------------------------------------------------------#
def standardize_confounds(X, std=[True, True]):
    """
    Standardize Confound Variables for GLM Modelling
    X = standardize_confounds(X, std)
    
        X   - n x c x r array; scan-by-variable-by-run signals
        std - list of bool; indicating which operations to perform (see below)
    
    X = standardize_confounds(X, std) standardizes confounds, i.e. subtracts
    the mean from each variable (in each run), if the first entry of std is
    true, and divides by the mean from each variable (in each run), if the
    second entry of std is true. By default, both entries are true.
    """
    
    # if X is a 2D matrix
    if len(X.shape) < 3:
        for k in range(X.shape[1]):
            if std[0]:          # subtract mean
                X[:,k] = X[:,k] - np.mean(X[:,k])
            if std[1]:          # divide by max
                X[:,k] = X[:,k] / np.max(X[:,k])
    
    # if X is a 3D array
    else:
        for j in range(X.shape[2]):
            for k in range(X.shape[1]):
                if std[0]:      # subtract mean
                    X[:,k,j] = X[:,k,j] - np.mean(X[:,k,j])
                if std[1]:      # divide by max
                    X[:,k,j] = X[:,k,j] / np.max(X[:,k,j])
    
    # return standardized confounds
    return X

# function: correct onsets
#-----------------------------------------------------------------------------#
def correct_onsets(ons, dur, stim, cong = False):
    """
    Correct Onsets Measured during EMPRISE Task
    ons, dur, stim = correct_onsets(ons, dur, stim)
    
        ons    - b x 1 vector; block-wise onsets [s]
        dur    - b x 1 vector; block-wise durations [s]
        stim   - b x 1 vector; block-wise stimuli (b = blocks)
        
        ons    - b0 x 1 vector; block-wise onsets [s]
        dur    - b0 x 1 vector; block-wise durations [s]
        stim   - b0 x 1 vector; block-wise stimuli (b = blocks per epoch)
    
    ons, dur, stim = correct_onsets(ons, dur, stim) corrects onsets ons,
    durations dur and stimuli stim, if signals are averaged across epochs
    within run. This is done by only using onsets, durations and stimuli from
    the first epoch and subtracting the discarded scan time from the onsets.
    """
    
    # correct for epochs
    if not cong:
        ons  = ons[:blocks_per_epoch] - num_scan_disc*TR
        dur  = dur[:blocks_per_epoch]
        stim = stim[:blocks_per_epoch]
    else:        
        ons  = ons[:blocks_per_epoch2] - num_scan_disc*TR
        dur  = dur[:blocks_per_epoch2]
        stim = stim[:blocks_per_epoch2]

        
    # return corrected onsets
    return ons, dur, stim

# function: transform onsets and durations
#-----------------------------------------------------------------------------#
def onsets_trials2blocks(ons, dur, stim, mode='true'):
    """
    Transform Onsets and Durations from Trials to Blocks
    ons, dur, stim = onsets_trials2blocks(ons, dur, stim, mode)
    
        ons  - list of arrays of floats; t x 1 vectors of onsets [s]
        dur  - list of arrays of floats; t x 1 vectors of durations [s]
        stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
        mode - string; duration conversion ("true" or "closed")

        ons  - list of arrays of floats; b x 1 vectors of onsets [s]
        dur  - list of arrays of floats; b x 1 vectors of durations [s]
        stim - list of arrays of floats; b x 1 vectors of stimuli (b = blocks)
        
    ons, dur, stim = onsets_trials2blocks(ons, dur, stim, mode) transforms
    onsets ons, durations dur and stimuli stim from trial-wise vectors to
    block-wise vectors.
    
    If mode is "true" (default), then the actual durations are used. If mode is
    "closed", then each block ends not earlier than when the next block starts.
    """
    
    # prepare onsets, durations, stimuli as empty lists
    ons_in  = ons; dur_in  = dur; stim_in = stim
    ons     = [];  dur     = [];  stim    = []
    
    # for all runs
    for j in range(len(ons_in)):
        
        # prepare onsets, durations, stimuli for this run
        ons.append([])
        dur.append([])
        stim.append([])
        
        # for all trials
        for i in range(len(ons_in[j])):
            
            # detect first block, last block and block change
            if i == 0:
                ons[j].append(ons_in[j][i])
                stim[j].append(stim_in[j][i])
            elif i == len(ons_in[j])-1:
                if mode == 'true':
                    dur[j].append((ons_in[j][i]+dur_in[j][i]) - ons[j][-1])
                elif mode == 'closed':
                    dur[j].append(max(dur[j]))
            elif stim_in[j][i] != stim_in[j][i-1]:
                if mode == 'true':
                    dur[j].append((ons_in[j][i-1]+dur_in[j][i-1]) - ons[j][-1])
                elif mode == 'closed':
                    dur[j].append(ons_in[j][i] - ons[j][-1])
                ons[j].append(ons_in[j][i])
                stim[j].append(stim_in[j][i])
        
        # convert lists to vectors
        ons[j]  = np.array(ons[j])
        dur[j]  = np.array(dur[j])
        stim[j] = np.array(stim[j])
    
    # return onsets
    return ons, dur, stim


def onsets_trials2trials(ons, dur, stim, mode = False):
    """
    Transform duration to the true duration of the stimuli length
    ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode)

        ons  - list of arrays of floats; t x 1 vectors of onsets [s]
        dur  - list of arrays of floats; t x 1 vectors of durations [s]
        stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
        mod  - string; audio1, audio2_block, audio2_stim

        ons  - list of arrays of floats; t x 1 vectors of onsets [s]
        dur  - list of arrays of floats; t x 1 vectors of durations [s]
        stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
        
        
    ons, dur, stim = onsets_trials2trials(ons, dur, stim) transforms
    onsets ons, durations dur and stimuli stim from log based trial-wise vectors 
    to true lenth of trial-wise vectors. 
    
    For audiotry sessions, the play time of auditory stimuli could be critical.
    Play time of the log files does not reflect exact play time of stimuli.
    So we correct the duration of play time considering pause within audio files. 

    """
    
    # true play time of stimuli
    if mode is False:
        print('mode is missing!')
        pass
        
    elif mode:
        if mode in ['audio1','audio1_seq']:         
            # each unit block (numerosity tones, one tone, two tones, three tones and so on) play time 
            # 10 ms tone, 50 ms pause between tones, so for example the corrected play time of numerosity 
            # stim 2 is 0.01 + 0.05 + 0.01  
            len_dict = {1:0.01, 2:0.07, 3:0.13, 4:0.19, 5: 0.25, 6: 0.31, 7: 0.37, 8: 0.43, 9:0.49, 20:0.25}
            
        elif mode in ['audio', 'audio_seq', 'audio_seq2']:
            len_dict = {1:0.05, 2:0.15, 3:0.25, 4:0.35, 5:0.45} # 20 is 20 times of 0.03 or 0.04 tones = 1.93 sec
            
        elif mode == 'audio2_block': 
            # each block (evnely spaced 6 stimuli) play time
            # Played audio fiels containing 6 times unit blocks within it to correct uneven paly time
            # between numerosity condtions.
            len_dict = {1:1.11, 2:1.47, 3:1.83, 4:2.19 ,5:2.55, 6:2.91, 7:3.27, 8:3.63, 9:3.99, 20:2.55}
            
        elif mode in ['audio2_stim', 'audio2_seq']:
            # each unit block (numerosity tones, one tone, two tones, three tones and so on) play time 
            # 10 ms tone, 50 ms pause between tones, so for example the corrected play time of numerosity 
            # stim 2 is 0.01 + 0.05 + 0.01
            for run in range(len(stim)):
                    len_dict = {1:0.01, 2:0.07, 3:0.13, 4:0.19, 5: 0.25, 6: 0.31, 7: 0.37, 8: 0.43, 9:0.49, 20:0.25}
                    stim[run] = np.array([s for s in stim[run] for _ in range(6)]) # audio2 audio file has 6 times tone units
                    ons[run] = np.array([o for o in ons[run] for _ in range(6)])   # therefore extend stim, ons, dur to stimuli-wise level
                    dur[run] = np.array([d for d in dur[run] for _ in range(6)])
                    
                    # update onsets of each stimuil
                    ons[run] = np.array([ons[run][i] + ((0.06 * stim[run][i]) + 0.16) * (j % 6) for i, j in enumerate(range(len(ons[run])))])

        else:
            print(f' {mode} is unknown mode name.')
                    
    # for all runs        
    for run in range(len(stim)):
        	for key, val in len_dict.items():
        	    indices = np.where(stim[run] == key)[0]  
        	    dur[run][indices] = val 
        
    if mode not in ['audio_seq', 'audio1_seq', 'audio2_seq','audio_seq2']:
        # return onsets
        return ons, dur, stim
    
    elif mode == 'audio_seq':
        # Consider sequential order of each tone. First tone is assgined to numerosity condition 1, 
        # second tone is assgined to numerosity condition 2 and so on. 
        # EMPRISE audio stimuli : 50 ms tone +  50 ms pause for [1,5], 
        #                        (30ms tone + 68ms pause + 40ms tone + 68 ms pause + 26ms tone + 67 pause)*6 times
        #                        plus (30ms tone + 68 ms pause + 30ms tone) for 20
        new_ons = []
        new_stim = []
        new_dur = []
    
        for run in range(len(stim)):
            curr_ons = []
            curr_stim = []
            curr_dur = []
            for i, j in zip(stim[run], ons[run]):
                if i in [2, 3, 4, 5]:
                    temp_stim = list(range(1, i)) + [i]
                    temp_ons = [round(j + n * 0.1, 2) for n in range(len(temp_stim))]
                    temp_dur = [0.05] * len(temp_stim)
        
                    curr_stim.extend(temp_stim)
                    curr_ons.extend(temp_ons)
                    curr_dur.extend(temp_dur)
        
                elif i == 20:
                    ons_list = [0.00, 0.098, 0.207, 0.3, 0.398, 0.507, 0.6, 0.698, 0.807, 0.9, 0.998, 1.107, 1.2, 1.298, 1.407, 1.5, 1.598, 1.707, 1.8, 1.898]
                    # round at third place of decimal, the tone length is 0.03, 0.04, 0.26 
                    dur_list = [0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.03] 
                    temp_stim = list(range(1, i)) + [i]
                    temp_ons = [round(j + o, 2) for o in ons_list]
                    temp_dur = dur_list[:len(temp_ons)]  # safety in case mismatch
        
                    curr_stim.extend(temp_stim)
                    curr_ons.extend(temp_ons)
                    curr_dur.extend(temp_dur)
        
                else:  # i == 1
                    curr_stim.append(i)
                    curr_ons.append(j)
                    curr_dur.append(0.05)
        
            new_stim.append(np.array(curr_stim))
            new_ons.append(np.array(curr_ons))
            new_dur.append(np.array(curr_dur))
        
        return new_ons, new_dur, new_stim

    elif mode == 'audio_seq2':
        # Consider sequential order of each tone. First tone is assgined to numerosity condition 1, 
        # second tone is assgined to numerosity condition 2 and so on. 
        # EMPRISE audio stimuli : 50 ms tone +  50 ms pause for [1,5], 
        #                        (30ms tone + 68ms pause + 40ms tone + 68 ms pause + 26ms tone + 67 pause)*6 times
        #                        plus (30ms tone + 68 ms pause + 30ms tone) for 20
        # audio_seq2 condition does not consider the 50ms pause bewteen tones within a numerosity condition.
        # duration of every tone is 0.1 ms, so the neuronal signals could have step like function in a numerosity condition. 
        new_ons = []
        new_stim = []
        new_dur = []
    
        for run in range(len(stim)):
            curr_ons = []
            curr_stim = []
            curr_dur = []
            for i, j in zip(stim[run], ons[run]):
                if i in [2, 3, 4, 5]:
                    temp_stim = list(range(1, i)) + [i]
                    temp_ons = [round(j + n * 0.1, 2) for n in range(len(temp_stim))]
                    temp_dur = [0.1] * len(temp_stim)
        
                    curr_stim.extend(temp_stim)
                    curr_ons.extend(temp_ons)
                    curr_dur.extend(temp_dur)
        
                elif i == 20:
                    ons_list = [0.00, 0.098, 0.207, 0.3, 0.398, 0.507, 0.6, 0.698, 0.807, 0.9, 0.998, 1.107, 1.2, 1.298, 1.407, 1.5, 1.598, 1.707, 1.8, 1.898]
                    # round at third place of decimal, the tone length is 0.03, 0.04, 0.26 
                    #dur_list = [0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.03] 
                    temp_stim = list(range(1, i)) + [i]
                    temp_ons = [round(j + o, 2) for o in ons_list]
                    temp_dur = [round(ons_list[n+1] - o,2) if n!=len(ons_list)-1 else 0.1 for n, o in enumerate(ons_list)]

                    curr_stim.extend(temp_stim)
                    curr_ons.extend(temp_ons)
                    curr_dur.extend(temp_dur)
        
                else:  # i == 1
                    curr_stim.append(i)
                    curr_ons.append(j)
                    curr_dur.append(0.1)
        
            new_stim.append(np.array(curr_stim))
            new_ons.append(np.array(curr_ons))
            new_dur.append(np.array(curr_dur))
        
        return new_ons, new_dur, new_stim

    
    else:
        # Consider sequential order of each tone. First tone is assgined to numerosity condition 1, 
        # second tone is assgined to numerosity condition 2 and so on. 
        new_ons = []
        new_stim = []
        new_dur = []

        for run in range(len(stim)):
            curr_ons = []
            curr_stim = []
            curr_dur = []
            for i, j in zip(stim[run], ons[run]):
                if i in [2, 3, 4, 5, 6, 7, 8, 9]:
                    curr_stim.extend(range(1, i))  
                    curr_stim.append(i)
                    curr_ons.append(j)
                    curr_ons.extend([round(j + n * 0.06, 2) for n in range(1, i)])
                else:
                    curr_stim.append(i)
                    curr_ons.append(j)
            for i in curr_stim:
                if i in [1,2,3,4,5,6,7,8,9]:
                    curr_dur.append(0.01)
                else:
                    curr_dur.append(0.25)
                
            new_stim.append(np.array(curr_stim))
            new_ons.append(np.array(curr_ons))
            new_dur.append(np.array(curr_dur))
            
        return new_ons, new_dur, new_stim
    
# function: create fsaverage midthickness mesh
