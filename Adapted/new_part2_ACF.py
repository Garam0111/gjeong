# Script was adjusted by Garam Jeong and Johanna Finemann
# Original script is from Alexsander Enge, univariate.py [https://github.com/SkeideLab/SLANG-analysis/tree/93890d2cda9aac1ded61d41ae1ed5bd27f4d4bc0/scripts]

from pathlib import Path
from subprocess import PIPE, run, CalledProcessError

# Input parameters: File paths
BIDS_DIR = Path('/data/pt_02825/MPCDF/BIDS')
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate'

# Input parameters: Cluster correction
CLUSTSIM_SIDEDNESS = '2-sided'
CLUSTSIM_NN = 'NN1'  # Must be 'NN1' for Nilearn's `get_clusters_table`
CLUSTSIM_VOXEL_THRESHOLD = 0.001
CLUSTSIM_CLUSTER_THRESHOLD = 0.05
CLUSTSIM_ITER = 10000

### functions

# =============================================================================
# Compute ACF using AFNI 
# =============================================================================

def compute_acf(residual_file, mask_file):
    """
    Computes ACF parameters using AFNI's 3dFWHMx.
    If mask and residual file grids don't match, resample the mask automatically.
    """
    # Ensure file paths are strings
    residual_file = str(residual_file)
    mask_file = str(mask_file)

    # Check if mask and residuals are on the same grid
    check_cmd = ['sc', 'afni',
        '24.3.08',
        '3dinfo', '-same_grid', residual_file, mask_file
    ]
    try:
        check_res = run(check_cmd, stdout=PIPE, stderr=PIPE, text=True, check=True)
    except CalledProcessError as e:
        print("Grid check failed:")
        print(e.stderr)
        raise

    if check_res.stdout.strip().split()[-1] == '0':
        print("Grid mismatch detected. Resampling mask...")

        # Create resampled mask path
        mask_path = Path(mask_file)
        resampled_mask = mask_path.with_name(mask_path.name.replace('.nii', '_resampled.nii'))


        resample_cmd = [ 'sc', 'afni', '24.3.08',
            '3dresample', 
            '-master', residual_file,
            '-inset', mask_file,
            '-prefix', str(resampled_mask)
        ]

        try:
            run(resample_cmd, stdout=PIPE, stderr=PIPE, text=True, check=True)
        except CalledProcessError as e:
            print("Mask resampling failed:")
            print(e.stderr)
            raise

        mask_file = str(resampled_mask)

    # Run 3dFWHMx
    cmd = [
        'sc', 'afni',
        '24.3.08',
        '3dFWHMx',
        '-mask', mask_file,
        '-acf', 'NULL',
        '-input', residual_file
    ]

    print("Running command:")
    print(" ".join(cmd))

    try:
        res = run(cmd, stdout=PIPE, stderr=PIPE, text=True, check=True)
    except CalledProcessError as e:
        print("Command failed with error:")
        print(e.stderr)
        raise

    # Extract a, b, c
    acf = res.stdout.split()[4:7]
    print(f'acf: {acf}')
    return [float(param) for param in acf]

def parse_clustsim_output(output):
    """Parses the output of AFNI's `3dClustSim` command into a dictionary."""

    lines = output.split('\n')
    ns_voxels = [float(line.split()[-1])
                 for line in lines
                 if line and not line.startswith('#')]

    results_dict = {}
    for sidedness in ['1-sided', '2-sided', 'bi-sided']:
        results_dict[sidedness] = {}
        for nn in [1, 2, 3]:
            results_dict[sidedness][f'NN{nn}'] = ns_voxels.pop(0)

    return results_dict

def compute_cluster_threshold(acf, mask_file, sidedness, nn, voxel_threshold,
                              cluster_threshold, iter):
    """Computes the FWE-corrected cluster size threshold using AFNI."""

    cmd = ['sc', 'afni',
        '24.3.08',
        '3dClustSim',
           '-mask', mask_file,
           '-acf', *[str(param) for param in acf],
           '-pthr', str(voxel_threshold),
           '-athr', str(cluster_threshold),
           '-iter', str(iter)]

    # Print the command for debugging
    print(f"Running command: {cmd}")

    res = run(cmd, stdout=PIPE, text=True, check=True)
    cluster_thresholds = parse_clustsim_output(res.stdout)

    # Process the output as needed
    print("Command output:", res.stdout)
    

    return cluster_thresholds[sidedness][nn]


