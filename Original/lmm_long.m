% This script is to calculate linear mixed model effect of a longitudinal dataset using lme functions distributed with Freesurfer 
% Bernal-Rusiel J.L., Greve D.N., Reuter M., Fischl B., Sabuncu M.R., 2012. Statistical Analysis of Longitudinal Neuroimage Data with Linear Mixed Effects Models, NeuroImage 66C, pp. 249-260, 2012.
% Bernal-Rusiel J.L., Greve D.N., Reuter M., Fischl B., Sabuncu M.R., 2013. Spatiotemporal Linear Mixed Effects Modeling for the Mass-univariate Analysis of Longitudinal Neuroimage Data, NeuroImage 81, pp 358–370, 2013. 

%-------------------------------------------------------------------------------------------------
% read the area data
[Y_l,mri_l] = fs_read_Y('./lme/lh.curv_sm5.mgh');
[Y_r,mri_r] = fs_read_Y('./lme/rh.curv_sm5.mgh');

% create design matrix
Qdec  = fReadQdec('task-Literacy_metadf.csv');
Qdec  = rmQdecCol(Qdec,1); % remove fsid
sID   = Qdec(2:end,1);
Qdec  = rmQdecCol(Qdec,1); % remove fsid-base
M     = Qdec2num(Qdec); 
time2 = M(:,1).^2;  % quadratic time term
X     = [ones(length(M),1) M time2];

[X_l,Y_l,ni_l] = sortData(X,2,Y_l,sID);  % along the second column, time point
[X_r,Y_r,ni_r] = sortData(X,2,Y_r,sID); 

% load verices info
lhsphere = fs_read_surf('/data/pt_02825/MPCDF/freesurfer/fsaverage/surf/lh.sphere'); 
lhcortex = fs_read_label('/data/pt_02825/MPCDF/freesurfer/fsaverage/label/lh.cortex.label');

rhsphere = fs_read_surf('/data/pt_02825/MPCDF/freesurfer/fsaverage/surf/rh.sphere'); 
rhcortex = fs_read_label('/data/pt_02825/MPCDF/freesurfer/fsaverage/label/rh.cortex.label');

% initial vertex-wise temporal covariance estimates
[lhTh0,lhRe] = lme_mass_fit_EMinit(X_l,[1 2],Y_l,ni_l,lhcortex,3);
[rhTh0,rhRe] = lme_mass_fit_EMinit(X_r,[1 2],Y_r,ni_r,rhcortex,3);

% segment covariances into homogeneous regions
[lhRgs, lhRgMeans] = lme_mass_RgGrow(lhsphere, lhRe, lhTh0, lhcortex, 2, 95);
[rhRgs, rhRgMeans] = lme_mass_RgGrow(rhsphere, rhRe, rhTh0, rhcortex, 2, 95);

% optional: compare the covariance and segmentation
surf.faces =  lhsphere.tri;
surf.vertices =  lhsphere.coord';

figure; p1 = patch(surf);
set(p1,'facecolor','interp','edgecolor','none','facevertexcdata',lhTh0(1,:)');

figure; p2 = patch(surf); set(p2,'facecolor','interp','edgecolor','none','facevertexcdata',lhRgMeans(1,:)');

% fit the model (intercept, slope)
lhstats = lme_mass_fit_Rgw(X_l,[1 2],Y_l,ni_l,lhTh0,lhRgs,lhsphere);
rhstats = lme_mass_fit_Rgw(X_r,[1 2],Y_r,ni_r,rhTh0,rhRgs,rhsphere);

% compute the single random effect model (intercept)
% Fit the model with one random effect using the segmentation obtained from the previous model: 
lhTh0_1RF = lme_mass_fit_EMinit(X_l,[1],Y_l,ni_l,lhcortex,3);
rhTh0_1RF = lme_mass_fit_EMinit(X_r,[1],Y_r,ni_r,rhcortex,3);

lhstats_1RF = lme_mass_fit_Rgw(X_l,[1],Y_l,ni_l,lhTh0_1RF,lhRgs,lhsphere); 
rhstats_1RF = lme_mass_fit_Rgw(X_r,[1],Y_r,ni_r,rhTh0_1RF,rhRgs,rhsphere);

%Likelyhood Ratio,
LR_pval_l = lme_mass_LR(lhstats,lhstats_1RF,1); % 1 means the difference in degrees of freedom (2 random effects vs 1 )
LR_pval_r = lme_mass_LR(rhstats,rhstats_1RF,1); 

% correction for multiple comparision
dvtx_l = lme_mass_FDR2(LR_pval_l,ones(1,length(LR_pval_l)),lhcortex,0.05,0); % 0 indiciate Bejamini-Hochberg FDR test method
dvtx_r = lme_mass_FDR2(LR_pval_r,ones(1,length(LR_pval_r)),rhcortex,0.05,0);

% compare the ratio between FDR passed vertices and marked cortex vertices
pass_ratio_l = length(dvtx_l)/length(lhcortex);
pass_ratio_r = length(dvtx_r)/length(rhcortex);

% Dependeing on pass_ratio, we chose the full model (Two random effects: slope and interception)
% We choose the full model
% Show the fixed effect (slope shared by the entire group): contrast matrix

CM.C = [0,1,0]; % time
CM2.C = [0,0,1]; % time quadratic
CM3.C = [0, 1, -1]; % H0: The effect size of time and time quadratic are same.

% F or Z test(vertex wise)
F_lhstats  = lme_mass_F(lhstats,CM);
F2_lhstats = lme_mass_F(lhstats,CM2);
F3_lhstats = lme_mass_F(lhstats,CM3);
 
F_rhstats  = lme_mass_F(rhstats,CM);
F2_rhstats = lme_mass_F(rhstats,CM2);
F3_rhstats = lme_mass_F(rhstats,CM3);

% correction for multiple comparision (FDR2 - see freesurfer web)
vx_l  = lme_mass_FDR2(F_lhstats.pval,F_lhstats.sgn,lhcortex,0.05,0);
vx2_l = lme_mass_FDR2(F2_lhstats.pval,F2_lhstats.sgn,lhcortex,0.05,0);
vx3_l = lme_mass_FDR2(F3_lhstats.pval,F3_lhstats.sgn,lhcortex,0.05,0);

vx_r  = lme_mass_FDR2(F_rhstats.pval,F_rhstats.sgn,rhcortex,0.05,0);
vx2_r = lme_mass_FDR2(F2_rhstats.pval,F2_rhstats.sgn,rhcortex,0.05,0);
vx3_r = lme_mass_FDR2(F3_rhstats.pval,F3_rhstats.sgn,rhcortex,0.05,0);


% convert f statistics to z score (since degree of freedom of F test is 1)
zvals_l      = sqrt(F_lhstats.F) .* F_lhstats.sgn; 
F_lhstats.z  = zvals_l;                            % add to F_lhstats cell
zvals2_l     = sqrt(F2_lhstats.F) .* F2_lhstats.sgn;
F2_lhstats.z = zvals2_l;
zvals3_l     = sqrt(F3_lhstats.F) .* F3_lhstats.sgn;
F3_lhstats.z = zvals3_l;

zvals_r = sqrt(F_rhstats.F) .* F_rhstats.sgn;
F_rhstats.z = zvals_r;     
zvals2_r = sqrt(F2_rhstats.F) .* F2_rhstats.sgn;
F2_rhstats.z = zvals2_r;
zvals3_r = sqrt(F3_rhstats.F) .* F3_rhstats.sgn;
F3_rhstats.z = zvals3_r;

% save uncorrected p values (valid variables: pval, fval, sig)
fs_write_fstats(F_lhstats,mri_l,'./lme/lh_area_t_beta_pval.mgh','pval'); 
fs_write_fstats(F_lhstats,mri_l,'./lme/lh_area_t_beta_f.mgh','fval');
fs_write_fstats(F_lhstats,mri_l,'./lme/lh_area_beta_sign.mgh','sig');

fs_write_fstats(F2_lhstats,mri_l,'./lme/lh_area_t2_beta_pval.mgh','pval'); 
fs_write_fstats(F2_lhstats,mri_l,'./lme/lh_area_t2_beta_pval.mgh','pval'); 
fs_write_fstats(F2_lhstats,mri_l,'./lme/lh_area_t2_beta_sign.mgh','sig');

fs_write_fstats(F3_lhstats,mri_l,'./lme/lh_area_t-t2_beta_pval.mgh','pval'); 
fs_write_fstats(F3_lhstats,mri_l,'./lme/lh_area_t-t2_beta_f.mgh','fval');
fs_write_fstats(F3_lhstats,mri_l,'./lme/lh_area_t-t2_beta_sign.mgh','sig');

fs_write_fstats(F_rhstats,mri_r,'./lme/rh_area_t_beta_pval.mgh','pval'); 
fs_write_fstats(F_rhstats,mri_r,'./lme/rh_area_t_beta_f.mgh','fval');
fs_write_fstats(F_rhstats,mri_r,'./lme/rh_area_beta_sign.mgh','sig');

fs_write_fstats(F2_rhstats,mri_r,'./lme/rh_area_t2_beta_pval.mgh','pval'); 
fs_write_fstats(F2_rhstats,mri_r,'./lme/rh_area_t2_beta_pval.mgh','pval'); 
fs_write_fstats(F2_rhstats,mri_r,'./lme/rh_area_t2_beta_sign.mgh','sig');

fs_write_fstats(F3_rhstats,mri_r,'./lme/rh_area_t-t2_beta_pval.mgh','pval'); 
fs_write_fstats(F3_rhstats,mri_r,'./lme/rh_area_t-t2_beta_f.mgh','fval');
fs_write_fstats(F3_rhstats,mri_r,'./lme/rh_area_t-t2_beta_sign.mgh','sig');

% save beta values
nv_l =length(lhstats);
nv_r =length(rhstats);

% second beta = time
Beta2_l = zeros(1,nv_l);
Beta2_r = zeros(1,nv_r);

% thrid beta = time^2
Beta3_l = zeros(1,nv_l);
Beta3_r = zeros(1,nv_r);

for i=1:nv_l
   if ~isempty(lhstats(i).Bhat)
      Beta2_l(i) = lhstats(i).Bhat(2);
      Beta3_l(i) = lhstats(i).Bhat(3);
      
   end
end

for i=1:nv_r
   if ~isempty(rhstats(i).Bhat)
      Beta2_r(i) = rhstats(i).Bhat(2); 
      Beta3_r(i) = rhstats(i).Bhat(3);
   end
end

mri1_l = mri_l;
mri1_r = mri_r;

mri1_l.volsz(4) = 1;
mri1_r.volsz(4) = 1;

fs_write_Y(Beta2_l,mri1_l,'area_t_beta_l.mgh');
fs_write_Y(Beat2_r,mri1_r,'area_t_beta_r.mgh');

fs_write_Y(Beta3_l,mri1_l,'area_t2_beta_l.mgh');
fs_write_Y(Beat3_r,mri1_r,'area_t2_beta_r.mgh');

% correction for multiple comparision in both hemi
P = [ F_lhstats.pval(lhcortex) F_rhstats.pval(rhcortex) ];
G = [ F_lhstats.sgn(lhcortex) F_rhstats.sgn(rhcortex) ];
[detvtx,sided_pval,pth] = lme_mass_FDR2(P,G,[],0.05,0);
pcor = -log10(pth);

% plot histogram of uncorrected pval
histogram(sided_pval, 100);
title('Uncorrected p-values across cortex');
xlabel('p-value'); ylabel('Vertex count');

% plot measured data of all sub along time points
% find the min p-val vertex
minv_lh = find(F_lhstats.pval == min(F_lhstats.pval));
minv_lh2 = find(F2_lhstats.pval == min(F2_rhstats.pval));

st_lh = Beta2_l(minv_lh); %slope of t at min pval vertex
st_lh2 = Beta3_l(minv_lh2); 

lme_lowessPlot(X(:,2), Y_l(:,minv_lh), 0.7) % span=[0,1]
