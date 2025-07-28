# Run linear mixed model on freeusrfer aparc ROIs 
# With and without quadratic terms and compare them is included. 
# It is generally recommended to include only linear term to reduce the complexity of model.

library(lme4)
library(lmerTest)
library(dplyr)

# Longitudinal data
long.df <- read.csv("/data/pt_02825/MPCDF/freesurfer/lme/table/lh_aparc_area.long.table", sep = "\t")

# Meta data
meta.df <- read.csv("/data/pt_02825/MPCDF/freesurfer/task-Literacy_metadf.csv", sep="\t")
long.df$month <- meta.df$month
long.df$fsid <- meta.df$fsid
long.df$fsid.base <- meta.df$fsid.base

# quadratic
long.df$month2 <- long.df$month^2

# Brain region columns
region_cols <- setdiff(names(long.df), c("rh.aparc.volumn", "fsid", "fsid.base", "month", "month2","eTIV","BrainSegVolNotVent" ))

# Prepare results container
results <- data.frame()

for (region in region_cols) {
  # Build formulas
  f_null   <- as.formula(paste(region, "~ month + (1 + month | fsid.base)"))
  f_quad   <- as.formula(paste(region, "~ month + month2 + (1 + month | fsid.base)"))
  
  # Fit models
  mod_null <- lmer(f_null, REML = FALSE, data = long.df)
  mod_quad <- lmer(f_quad, REML = FALSE, data = long.df)
  comp     <- anova(mod_null, mod_quad)
  
  # Extract summary
  s <- summary(mod_quad)
  
  # Safely extract fixed effects
  get_coef <- function(name, coef_summary) {
    if (name %in% rownames(coef_summary)) {
      est <- coef_summary[name, "Estimate"]
      se  <- coef_summary[name, "Std. Error"]
      p   <- coef_summary[name, "Pr(>|t|)"]
    } else {
      est <- se <- p <- NA
    }
    return(c(est, se, p))
  }
  
  linear   <- get_coef("month", s$coefficients)
  quad     <- get_coef("month2", s$coefficients)
  
  # Store
  results <- rbind(results, data.frame(
    region = region,
    month_est  = linear[1],
    month_se   = linear[2],
    month_pval = linear[3],
    month2_est  = quad[1],
    month2_se   = quad[2],
    month2_pval = quad[3],
    AIC_null = AIC(mod_null),
    AIC_quad = AIC(mod_quad),
    delta_AIC = AIC(mod_null) - AIC(mod_quad),
    anova_pval = comp$`Pr(>Chisq)`[2]
  ))
}

# FDR-corrected q values
pvals <- results$month_pval  
pvals_fdr <- p.adjust(pvals, method = "fdr")
sum(pvals_fdr < 0.05)  # how many survive correction

# Add FDR-corrected q-values using Benjamini-Hochberg (default method)
results$month_qval <- pvals_fdr

write.csv(results, "/data/pt_02825/MPCDF/freesurfer/lme/table/_lme_results_.csv", row.names = FALSE)



# --------------------------Figures--------------------------------------------------------------------
library(ggplot2)
library(lme4)

long.df <- read.csv("/data/pt_02825/MPCDF/freesurfer/lme/table/lh_aparc_volume.long.table", sep = "\t")
meta.df <- read.csv("/data/pt_02825/MPCDF/freesurfer/task-Literacy_metadf.csv", sep="\t")
long.df$month <- meta.df$month
long.df$fsid <- meta.df$fsid
long.df$fsid.base <- meta.df$fsid.base

# Choose your significant region
region <- "rh_pericalcarine_gauscurv" 

# Fit linear mixed model (intercept + slope by subject)
mod <- lmer(as.formula(paste(region, "~ month + (1 + month| fsid.base)")), data = long.df)

slope <- fixef(mod)["month"]
slope_rounded <- round(slope, 4)

# Create predicted fixed-effect line
fixed_line <- data.frame(
  month = seq(min(long.df$month), max(long.df$month), length.out = 100)
)
fixed_line[[region]] <- predict(mod, newdata = fixed_line, re.form = NA)
fsid.base <- long.df$fsid.base

# Plot individual lines + population-level line
ggplot(long.df, aes(x = month, y = .data[[region]])) +
  geom_line(aes(group = fsid.base), alpha = 0.3, color = "gray") +  # Individual subjects
  geom_line(data = fixed_line, aes(x = month, y = .data[[region]]), color = "red", size = 1.2) +  # Fixed effect
  geom_smooth(method = "lm", se = FALSE, color = "blue", size = 1, linetype = "dashed") +  # Quick OLS line (optional)
  annotate("text", x = Inf, y = Inf, label = paste("Fixed Slope =", slope_rounded),
           hjust = 1.1, vjust = 1.5, size = 5, color = "red") +
  labs(
    title = paste("Change in", region, "over Time"),
    subtitle = "Gray: individual trajectories | Red: LME fixed effect | Blue dashed: linear fit",
    x = "Month",
    y = "Surface Area"
  ) +
  theme_minimal()

