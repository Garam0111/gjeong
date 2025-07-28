# Script was adjusted by Garam Jeong and Johanna Finemann
# Original script is from Alexsander Enge, univariate.py [https://github.com/SkeideLab/SLANG-analysis/tree/93890d2cda9aac1ded61d41ae1ed5bd27f4d4bc0/scripts]

import nibabel as nib
import numpy as np

from nilearn.image import binarize_img, load_img, math_img, mean_img
from nilearn.reporting import get_clusters_table
from nibabel import Nifti1Image

from scipy.stats import norm
from juliacall import Main as jl

jl.seval("import Pkg; Pkg.add(\"MixedModels\")")


# Input parameters: Group-level linear mixed models
FORMULA = 'beta ~ time + (time | subject)'
            
def compute_beta_img(glm, conditions_plus, conditions_minus):
    """
    output: nii image (z score)
    Computes a beta image from a fitted GLM for a given contrast."""

    design_matrices = glm.design_matrices_
    assert len(design_matrices) == 1
    design_matrix = design_matrices[0]

    contrast_values = np.zeros(design_matrix.shape[1])
    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)
    
    try:
        glm_efct_map = glm.compute_contrast(contrast_values, output_type='z_score')

        # Ensure the output is a valid Nifti1Image
        if not isinstance(glm_efct_map, Nifti1Image):
            raise ValueError("Unexpected return type from compute_contrast.")

        return glm_efct_map

    except Exception as e:
        print(f"Error happens in 1st level effect size: {e}")
        return None

def save_beta_img(beta_img, output_dir, subject, session, task, space,
                  contrast_label):
    """Saves a beta image to a NIfTI file in the output directory."""

    sub = f'sub-{subject}'
    ses = f'ses-{session}'
    tas = f'task-{task}'
    spc = f'space-{space}'
    des = f'desc-{contrast_label}'

    beta_dir = output_dir / sub / ses / 'func'
    beta_dir.mkdir(parents=True, exist_ok=True)
    beta_filename = f'{sub}_{ses}_{tas}_{spc}_{des}_z_score.nii.gz'
    beta_file = beta_dir / beta_filename
    beta_img.to_filename(beta_file)

def fit_mixed_models(formula, dfs):
    """Fits mixed models for a list of DataFrames using the `MixedModels`
       package in Julia. Using thread to prarellize voxel wise calculation """
       
    model_cmd = f"""
        using MixedModels
        using Suppressor
        using Base.Threads

        function fit_mixed_model(df)
            fml = @formula({formula})
            mod = @suppress fit(MixedModel, fml, df)
            bs = mod.beta
            zs = mod.beta ./ mod.stderror
        return bs, zs
        end

        function fit_mixed_models(dfs)
            results = Vector{{Tuple{{Vector{{Float64}}, Vector{{Float64}}}}}}(undef, length(dfs))
            @threads for i in 1:length(dfs)
                try
                    results[i] = fit_mixed_model(dfs[i])
                catch e
                    @warn "Model failed for index $i with error: $e"
                    results[i] = (fill(NaN, 2), fill(NaN, 2))  # fallback if model fails
                end
            end
            return results
        end
        """
    fit_mixed_models_julia = jl.seval(model_cmd)
    return fit_mixed_models_julia(dfs)


def save_array_to_nifti(array, ref_img, voxel_ixs, output_dir, task, space,
                        desc, suffix, subject=None, session=None):
    """Inserts a NumPy array into a NIfTI image and saves it to a file."""

    full_array = np.zeros(ref_img.shape)
    full_array[tuple(voxel_ixs.T)] = array
    if type(ref_img) != nib.nifti1.Nifti1Image:
        aff = nib.load(ref_img)
        aff = aff.affine
        img = nib.Nifti1Image(full_array, aff)
    else:
        img = nib.Nifti1Image(full_array, ref_img.affine)

    img_file = save_img(img, output_dir, task, space, desc, suffix)

    return img, img_file


def save_clusters(img, voxel_threshold, cluster_threshold, output_dir, task,
                  space, contrast_label, suffix):
    """Finds clusters in a z-map and saves them as a table and NIfTI image."""

    voxel_threshold_z = norm.ppf(1 - voxel_threshold / 2)  # p to z

    cluster_df, cluster_imgs = \
        get_clusters_table(img, voxel_threshold_z, cluster_threshold,
                           two_sided=True, return_label_maps=True)

    has_pos_clusters = any(cluster_df['Peak Stat'] > 0)
    has_neg_clusters = any(cluster_df['Peak Stat'] < 0)

    if has_pos_clusters:

        if has_neg_clusters:

            neg_ixs = cluster_df['Peak Stat'] < 0
            cluster_df.loc[neg_ixs, 'Cluster ID'] = \
                '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

            cluster_img = math_img('img_pos - img_neg',
                                   img_pos=cluster_imgs[0],
                                   img_neg=cluster_imgs[1])

        else:

            cluster_img = cluster_imgs[0]

    elif has_neg_clusters:

        neg_ixs = cluster_df['Peak Stat'] < 0
        cluster_df.loc[neg_ixs, 'Cluster ID'] = \
            '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

        cluster_img = math_img('-img', img=cluster_imgs[0])

    else:

        cluster_img = math_img('img - img', img=img)

    save_df(cluster_df, output_dir, task, space, contrast_label,
            suffix=f'{suffix}-clusters')

    save_img(cluster_img, output_dir, task, space, contrast_label,
             suffix=f'{suffix}-clusters')


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

    return file

def save_df(df, output_dir, task, space, desc, suffix):
    """Saves a DataFrame to a TSV file."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.tsv'
    file = output_dir / filename
    df.to_csv(file, sep='\t', index=False, float_format='%.5f')

    return file

def save_fallback(array, output_dir, suffix):
    fallback_file = output_dir / f'failed_save_{suffix}.npy'
    np.save(fallback_file, array)
    print(f"Failed to save NIfTI — array saved to: {fallback_file}")
