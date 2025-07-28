# Original Scripts Written By Joram Soch to analyze topographic distribution of numerosity sensitive neuronal population on 7T fMRI
# Original Scripts: EMPRISE Public Scripts [https://github.com/SkeideLab/EMPRISE-analysis/tree/11ef8f593a4e895d4bd8e53d3161771a83383454/code/Python]
# A Major adpatation: new analysis model implanted (CST model) to consider temporal property of auditory stimuli
# Most of the minor adjustments are not reported: To spot out all adaptation, check the original code from the public repository of EMPRISE project


"""
NumpRF - numerosity population receptive modelling

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-22, 14:34: log2lin, lin2log
2023-06-22, 17:04: f_log, neuronal_signals, simulate
2023-06-22, 17:55: hemodynamic_signals, simulate
2023-06-22, 21:49: testing
2023-06-26, 18:11: refactoring
2023-06-29, 16:58: plot_task_signals
2023-07-03, 10:26: plot_signals_axis, plot_signals_figure, debugging
2023-07-13, 10:45: plot_signals_figure
2023-07-13, 16:58: estimate_MLE_rgs, log2lin, lin2log
2023-07-14, 17:54: estimate_MLE_rgs, hemodynamic_signals
2023-08-10, 14:07: simulate, estimate_MLE_fgs
2023-08-21, 17:50: rewriting to OOP
2023-08-23, 09:22: estimate_MLE
2023-08-24, 16:39: refactoring
2023-08-28, 16:25: hemodynamic_signals, simulate, estimate_MLE
2023-08-31, 19:38: estimate_MLE, refactoring, testing
2023-09-07, 11:24: estimate_MLE, free_parameters
2023-09-14, 14:14: estimate_MLE
2023-09-18, 17:39: estimate_MLE
2023-09-21, 11:47: estimate_MLE, free_parameters
2023-09-26, 16:46: estimate_MLE
2023-10-26, 17:27: MLL2Rsq
2023-11-02, 08:01: yp2Rsq
2023-11-02, 09:52: Rsqtest
2023-11-02, 13:51: Rsqtest
2023-11-07, 15:54: f_lin, fwhm2sigma
2023-11-23, 11:33: estimate_MLE, estimate_MLE_rgs
2023-11-27, 14:43: corrtest, Rsqtest
2023-12-16, 13:04: Rsq2pval
2024-02-07, 13:04: pval2Rsq
2024-04-24, 09:37: hemodynamic_signals
2024-05-28, 10:52: calculate_Rsq
2024-06-25, 15:03: Rsqsig
2024-06-27, 12:23: Rsqsig
2024-07-01, 18:19: sig2fwhm, neuronal_signals, estimate_MLE
"""
"""
Adapted By Garam Jeong
2025-06, gamma_function, biphasic_gamma, apply_nonlinearity 
2025-07, CST_Bold_prediction
2025-07, simulate, free_parameters
2025-07, estimate_CST_MLE
"""

# import packages
#-----------------------------------------------------------------------------#
import math
import numpy as np
import scipy as sp
from scipy.signal import convolve
import PySPMs as PySPM

# function: linear tuning
#-----------------------------------------------------------------------------#
def f_lin(x, mu_lin, sig_lin):
    """
    Calculate Linear Tuning Function
    y = f_lin(x, mu_lin, sig_lin)
        
        x       - array of floats; stimuli at which to evaluate tuning
        mu_lin  - float; mean of tuning in linear space
        sig_lin - float; standard deviation of tuning in linear space
        
        y       - array of floats; expected response based on tuning function
        
    y = f_lin(x, mu_lin, sig_lin) returns the value of the Gaussian tuning
    function with mean mu_lin and standard deviation sig_lin in linear
    stimuli space at the argument x.
    """
    
    # calculate function value
    y = np.exp(-1/2 * (x-mu_lin)**2 / sig_lin**2)
    return y

# function: logarithmic tuning
#-----------------------------------------------------------------------------#
def f_log(x, mu_log, sig_log):
    """
    Calculate Logarithmic Tuning Function
    y = f_log(x, mu_log, sig_log)
        
        x       - array of floats; stimuli at which to evaluate tuning
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
        y       - array of floats; expected response based on tuning function
        
    y = f_log(x, mu_log, sig_log) returns the value of the Gaussian tuning
    function with mean mu_log and standard deviation sig_log in logarithmic
    stimuli space at the argument x.
    """
    
    # calculate function value
    y = np.exp(-1/2 * (np.log(x)-mu_log)**2 / sig_log**2)
    return y

# function: logarithmic to linear
#-----------------------------------------------------------------------------#
def log2lin(mu_log, sig_log):
    """
    Transform Logarithmic Tuning Parameters to Linear Space
    mu, fwhm = log2lin(mu_log, sig_log)
        
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
        mu      - float; mean of tuning in linear space
        fwhm    - float; full width at half maximum in linear space
        
    mu, fwhm = log2lin(mu_log, sig_log) transforms logarithmic tuning parameters
    mu_log and sig_log and returns linear tuning parameters mu and fhwm.
    """
    
    # calculate mu and fwhm
    mu   = np.exp(mu_log)
    fwhm = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log) - \
           np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
    return mu, fwhm
    
# function: linear to logarithmic
#-----------------------------------------------------------------------------#
def lin2log(mu, fwhm):
    """
    Transform Linear Tuning Parameters to Logarithmic Space
    mu_log, sig_log = log2lin(mu, fwhm)
        
        mu      - float; mean of tuning in linear space
        fwhm    - float; full width at half maximum in linear space
        
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
    mu_log, sig_log = log2lin(mu, fwhm) transforms linear tuning parameters
    mu and fhwm and returns logarithmic tuning parameters mu_log and sig_log.
    """
    
    # catch, if arrays
    if type(mu) == np.ndarray and type(fwhm) == np.ndarray:
        mu_log  = np.zeros(mu.shape)
        sig_log = np.zeros(fwhm.shape)
        for i in range(mu.size):
            mu_log[i], sig_log[i] = lin2log(mu[i], fwhm[i])
    
    # otherwise, numericals
    else:
    
        # calculate mu_log
        mu_log = math.log(mu)
        
        # calculate sig_log
        sig_log = 1                             # iterative algorithm to find sig_log
        step    = 0                             # start at 1, increase or decrease by 10^-s,
        sign    = 1                             # if the sign changes, increase s by 1
        while step < 5:
            m, f = log2lin(mu_log, sig_log)     # calculate mu, fwhm, as of now
            if f == fwhm:                       # if f equal fwhm, sig_log is found
                s = 0
                break
            elif f < fwhm:                      # if f smaller fwhm, increase sig_log
                s = +1
            elif f > fwhm:                      # if f larger fwhm, decrease sig_log
                s = -1
            if s != sign:                       # if direction has changed, reduce step size
                step = step + 1                 # calculate new estimate for sig_log
            sig_log = sig_log + s*math.pow(10,-step)
            sign    = s
        del m, f, s
    
    # return mu_log and sig_log
    return mu_log, sig_log

# function: sigma to fwhm
#-----------------------------------------------------------------------------#
def sig2fwhm(sig):
    """
    Transform Standard Deviation to Tuning Width
    
    
        sig  - float; standard deviation of tuning in linear space
        
        fwhm - float; full width at half maximum in linear space
    
    fwhm = sig2fwhm(sig) transforms standard deviation sig to tuning width fwhm
    measured as full width at half maximum of tuning in linear space.
    """
    
    # calculate sigma
    fwhm = sig*(2*math.sqrt(2*math.log(2)))
    return fwhm

# function: fwhm to sigma
#-----------------------------------------------------------------------------#
def fwhm2sig(fwhm):
    """
    Transform Tuning Width to Standard Deviation
    sig = fwhm2sig(fwhm)
        
        fwhm - float; full width at half maximum in linear space
        
        sig  - float; standard deviation of tuning in linear space
    
    sig = fwhm2sig(fwhm) transforms tuning width fwhm measured as full width at
    half maximum to standard deviation sig of tuning in linear space.
    """
    
    # calculate sigma
    sig = fwhm/(2*math.sqrt(2*math.log(2)))
    return sig

# function: MLL to R^2
#-----------------------------------------------------------------------------#
def MLL2Rsq(MLL1, MLL0, n):
    """
    Convert Maximum Log-Likelihoods to Coefficients of Determination
    Rsq = MLL2Rsq(MLL1, MLL0, n)
        
        MLL1 - 1 x v array; maximum log-likelihoods for model of interest
        MLL0 - 1 x v array; maximum log-likelihoods for model with intercept only
        n    - int; number of data points used to calculate MLL1 and MLL0
        
        Rsq  - 1 x v array; coefficients of determination for model of interest
        
    Rsq = MLL2Rsq(MLL1, MLL0, n) converts the difference in maximum log-
    likelihoods (MLL1-MLL0) to coefficients of determination ("R-squared")
    assuming linear regression models and number of observations n [1].
    
    [1] https://statproofbook.github.io/P/rsq-mll
    """
    
    # calculate R-squared
    Rsq = 1 - np.power(np.exp(MLL1-MLL0), -2/n)
    return Rsq

# function: y/y_p to R^2
#-----------------------------------------------------------------------------#
def yp2Rsq(y, yp):
    """
    Convert Predicted Time Series to Coefficients of Determination
    Rsq = yp2Rsq(y, yp)
        
        y   - n x v array; observed time series
        yp  - n x v array; predicted time series
        
        Rsq - 1 x v array; coefficients of determination
        
    Rsq = yp2Rsq(y, yp) converts the observed and predicted time series into
    coefficients of determination ("R-squared") assuming linear regression
    models [1].
    
    [1] https://statproofbook.github.io/P/rsq-der
    """
    
    # calculate R-squared
    RSS = np.sum( np.power(y-yp, 2), axis=0)
    TSS = np.sum( np.power(y-np.tile(np.mean(y,axis=0),(y.shape[0],1)), 2), axis=0)
    Rsq = 1 - RSS/TSS
    return Rsq

# function: test for R-squared
#-----------------------------------------------------------------------------#
def Rsqtest(y, yp, p=2, alpha=0.05):
    """
    Significance Test for Coefficient of Determination based on F-Test
    h, p, stats = Rsqtest(y, yp, p, alpha)
    
        y       - n x v array; observed time series
        yp      - n x v array; predicted time series
        p       - int; number of explanatory variables used to
                       predict signals from which Rsq is calculated
                       (default: 2 [intercept and slope])
        alpha   - float; significance level for the F-test
        
        h       - 1 x v vector; indicating rejectance of the null hypothesis
        p       - 1 x v vector; p-values computed under the null hypothesis
        stats   - dict; further information on statistical inference:
        o Fstat - 1 x v vector; values of the F-statistic
        o df    - list of ints; degrees of freedom for the F-statistic
    
    h, p, stats = Rsqtest(y, yp, p, alpha) performs an F-test for the
    coefficient of determination Rsq assuming predicted signals coming from
    linear regression models with p free parameters and returns a
    vector of logicals h indicating rejectance of the null hypothesis
        H0: Rsq = 0
    and the vector of p-values in favor of the alternative hypothesis
        H1: Rsq > 0
    as well as further information on the statistical test [1].
    
    [1] https://en.wikipedia.org/wiki/F-test#Regression_problems
    """
    
    # calculate residual sum of squares for R^2 model
    RSS2 = np.sum(np.power(y-yp,2), axis=0)
    
    # calculate residual sum of squares for null model
    ym   = np.tile(np.mean(y, axis=0), (y.shape[0],1))
    RSS1 = np.sum(np.power(y-ym,2), axis=0)
    
    # calculate F-statistics
    n  = y.shape[0]
    p2 = p
    p1 = 1
    F  = ((RSS1-RSS2)/(p2-p1))/(RSS2/(n-p2))
    
    # calculate p-values
    stats = {'Fstat': F, 'df': [p2-p1, n-p2]}
    p     = 1 - sp.stats.f.cdf(F, p2-p1, n-p2)
    h     = p < alpha
    
    # return test statistics
    return h, p, stats

# function: R-squared to p-value
#-----------------------------------------------------------------------------#
def Rsq2pval(Rsq, n, p=2):
    """
    Convert Coefficient of Determination to P-Value, given n and p
    pval = Rsq2pval(Rsq, n, p)
    
        Rsq  - 1 x v array; coefficients of determination
        n    - int; number of observations
        p    - int; number of explanatory variables used to
                    predict signals from which Rsq is calculated
                    (default: 2 [intercept and slope])
        
        pval - 1 x v array; p-values, given F-test for Rsq (see "Rsqtest")
    
    pval = Rsq2pval(Rsq, n, p) converts R-squared to a p-value given number
    of observations n and number of predictors p, assuming an F-test for the
    coefficient of determination Rsq (see "Rsqtest") [1].
    
    [1] https://statproofbook.github.io/P/fstat-rsq.html
    """
    
    # calculate F-statistics
    p2 = p
    p1 = 1
    F  = (Rsq/(p2-p1))/((1-Rsq)/(n-p2))
    
    # calculate p-values
    pval = 1 - sp.stats.f.cdf(F, p2-p1, n-p2)
    return pval

# function: p-value to R-squared
#-----------------------------------------------------------------------------#
def pval2Rsq(pval, n, p=2):
    """
    Convert P-Value to Coefficient of Determination, given n and p
    Rsq = pval2Rsq(pval, n, p)
    
        pval - 1 x v array; p-values, given F-test for Rsq (see "Rsqtest")
        n    - int; number of observations
        p    - int; number of explanatory variables used to
                    predict signals from which Rsq is calculated
                    (default: 2 [intercept and slope])
        
        Rsq  - 1 x v array; coefficients of determination
    
    pval = Rsq2pval(Rsq, n, p) converts a p-value to R-squared given number
    of observations n and number of predictors p, assuming an F-test for the
    coefficient of determination Rsq (see "Rsqtest").
    """
    
    # calculate F-statistics
    p2 = p
    p1 = 1
    F  = sp.stats.f.ppf(1-pval, p2-p1, n-p2)
    
    # calculate R-squared
    Rsq = (F/(n-p2))/((1/(p2-p1)) + (F/(n-p2)))
    return Rsq

# function: significance of R-squared
#-----------------------------------------------------------------------------#
def Rsqsig(Rsq, n, p=2, alpha=0.05, meth=''):
    """
    Assess Significance of Coefficient of Determination, given n and p
    sig = Rsqsig(Rsq, n, p, alpha, meth)
    
        Rsq   - 1 x v array; coefficients of determination
        n     - int; number of observations (see "Rsq2pval")
        p     - int; number of explanatory variables (see "Rsq2pval")
        alpha - float; significance level of the statistical test
        meth  - string; method for multiple comparison correction
        
        sig   - 1 x v array; true, if F-test for Rsq is significant
    
    sig = Rsqsig(Rsq, n, p, alpha, meth) converts R-squareds to p-values given
    number of observations n and number of predictors p and assesses statistical
    significance given significance level alpha and multiple comparison
    correction technique meth.
    
    The input parameter "meth" is a string that can be one of the following:
    o "" : no multiple comparison corretion                    [1]
    o "B": Bonferroni correction for multiple comparisons      [2]
    o "H": Holm-Bonferroni correction for multiple comparisons [3]
    o "S": Šidák correction for multiple comparisons           [4]
    
    [1] https://en.wikipedia.org/wiki/Multiple_comparisons_problem
    [2] https://en.wikipedia.org/wiki/Bonferroni_correction
    [3] https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    [4] https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction
    """
    
    # calculate p-values
    pval = Rsq2pval(Rsq, n, p)
    m    = pval.size
    
    # no multiple comparison correction
    if   meth == '':
        sig   = pval < alpha
    
    # Bonferroni correction for multiple comparisons
    elif meth == 'B':
        alpha = alpha/m
        sig   = pval < alpha
    
    # Šidák correction for multiple comparisons
    elif meth == 'S':
        alpha = 1 - np.power(1-alpha, 1/m)
        sig   = pval < alpha
        
    # Holm-Bonferroni correction for multiple comparisons
    elif meth == 'H':
        ind   = np.argsort(pval)
        sig   = np.zeros(pval.shape, dtype=bool)
        for i in range(m):
            if pval[ind[i]] < (alpha/(m-i)):
                sig[ind[i]] = True
            else:
                break
    
    # unknown method for multiple comparison correction
    else:
        err_msg = 'Unknown multiple comparison correction method: "{}". Method must be "", "B", "H" or "S".'
        raise ValueError(err_msg.format(meth))
    
    # return significance
    return sig

# function: test for correlation
#-----------------------------------------------------------------------------#
def corrtest(r, n, p=2, alpha=0.05):
    """
    Significance Test for Correlation Coefficient based on t-Test
    h, p, stats = corrtest(r, n, p, alpha)
    
        r       - 1 x v array; Pearson correlation coefficients
        n       - int; number of data points used to calculate r
        p       - int; number of explanatory variables used to
                       predict signals from which r was calculated
                       (default: 2 [intercept and slope])
        alpha   - float; significance level for the t-test
        
        h       - 1 x v vector; indicating rejectance of the null hypothesis
        p       - 1 x v vector; p-values computed under the null hypothesis
        stats   - dict; further information on statistical inference:
        o tstat - 1 x v vector; values of the t-statistic
        o df    - int; degrees of freedom of the t-statistic
    
    h, p, stats = corrtest(r, n, p, alpha) performs a t-test for the
    correlation coefficients r assuming linear regression models with
    n observations, p free parameters and significance level alpha and returns
    a vector of logicals h indicating rejectance of the null hypothesis
        H0: r = 0
    and the vector of p-values in favor of the alternative hypothesis
        H1: r > 0
    as well as further information on the statistical test [1].
    
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Testing_using_Student's_t-distribution
    """
    
    # calculate t-statistics
    t = r * math.sqrt(n-p) / np.sqrt(1-np.power(r,2))
    
    # calculate p-values
    stats = {'tstat': t, 'df': n-p}
    p     = 1 - sp.stats.t.cdf(t, n-p)
    h     = p < alpha
    
    # return test statistics
    return h, p, stats

# function: sustain impulse response function
#-----------------------------------------------------------------------------#
def gamma_function(t, tau ,kapha, m):
    """
    Parameters
    ----------
    t     : float
           time point.
    tau   : float
    kapha : float
    m     : float; m>1 

    Returns
    -------
    float
        value of impulse function at time point t
        
    # Define gamma function for sustained impulse response
    # kapha and m, m1, m2 are decided by previous study on human visual system 
    # kapha for a sustained impulse response = 1, m1 = 9
    # Ref: [Watson AB (1986) Temporal sensitivity.] [Grill-Spector K (2019) Differential sustained and transient temporal processing across visual streams]    

    """
    if kapha is None:
        k = tau
    else:
        k = tau*kapha 
        
    gamma_values = (t / k)**(m - 1) * np.exp(-t / k) / (k * math.factorial(m - 1))
    
    # maximum occurs at t = k * (m - 1) when m > 1
    if m > 1:
        t_max = k * (m - 1)
        max_value = (t_max / k)**(m - 1) * np.exp(-t_max / k) / (k * math.factorial(m - 1))
    else:
        # For m <= 1, maximum is at t -> 0, use the maximum from the data
        max_value = np.max(gamma_values)
    
    # Normalize by the maximum value
    return gamma_values / max_value    


# function: on-transient impulse response function
#-----------------------------------------------------------------------------#
def biphasic_gamma(t, tau, kapha1, kapha2, m1, m2, s): 
    """

    Parameters
    ----------
    t      : float
             time point
    tau    : float
    kapha1 : float
    kapha2 : float
    m1     : float
    m2     : float
    s      : float ranged[0,1]
             decide the weight of inhibitory biphaisc gamma function

    Returns
    -------
    float
        value of impulse function at time point t
    
    # kapha for a transient impluse response = 1.33, m1 = 9, m2 = 10
    # Define the biphasic gamma function (on transient impulse function)
    # -(on transient impulse function) = off transient impulse function
    # s = 0, sustained impulse function
    """
    h1 = gamma_function(t,tau,kapha1,m1)
    h2 = gamma_function(t,tau,kapha2,m2)
    return h1 - s*h2

# function: Rectified linear unit and non-linear compression
#-----------------------------------------------------------------------------#
def apply_nonlinearity(signal, n):
    """

    Parameters
    ----------
    signal : float, array
        float or array of singal 
    n : float
        ranged [0.1,1]

    Returns
    -------
    float, array
        Reictified linear unit and non-linear compression applied signal 
        i.e., a prolonged stimulus get a low weight  

    """
    return np.power(np.maximum(0, signal), n)

# function: predicted BOLD of CST applying linear time invariant of hemodynamic signals 
#----------------------------------------------------------------------------------------
def CST_Bold_prediction(ons, dur, stim, dt, TR, n, irf_l, mu_log, sig_log, tau, power, lin=False, order=1, block=None):
    """
    Parameters
    ----------
    ons        : list of onset times (block-wise: seconds, sequence-wise: msec)
    dur        : list of durations (same unit as ons)
    stim       : list of stimulus values
    dt         : time resolution of neuronal signals (allowed 0.001 for 1 ms or 0.01 for 10 ms )
    TR         : float; reaction time of fMRI scans (seconds)
    n          : int; scan number
    irf_l      : list of 1D neural impulse response function (allowed items are ['sus','on','off'])
    mu_log     : vector of preferred feature values 
    sig_log    : vector of tuning widths 
    tau        : float defining the peak of IRF
    power      : float defining power of non-linear gain function
    lin        : whether to use linear tuning
    order      : int; order of HRF
    block      : list (for block-wise: [mtr, mto], for seq-wise: None, default: None)

    Returns
    -------
    S          : Dictionary of predicted bolds of irf_l [T, V] array of BOLD signals over time for each voxel
                 if irf_l = ['sus','on'], then S['sus'] = S_sus s.t. S_sus.shape = (T,V)
    Z          : Dictionary of modeled neuronal signals 
    t          : time vector
    """
    
    if block:
        mtr = block[0]
        mto = block[1]
        dt  = TR/mtr
        ons = np.round(np.array(ons) / dt).astype(int)
        dur = np.round(np.array(dur) / dt).astype(int)
        T   = math.ceil(np.max(ons+dur))
        t_z = np.arange(0,T, dt)
        
    ons = np.round(np.array(ons)/dt)
    dur = np.round(np.array(dur)/dt)
    T = math.ceil(np.max(ons+dur))
    V = len(mu_log)
    
    # Time vector
    t_z = np.arange(0, T, 1) 
    
    # define HRF
    bf  = PySPM.spm_get_bf(dt, 'HRF', p=None, order=order)

    # Generate impulse response over time
    irf_d = 500                      # duration of IRF, unit: msec
    if dt == 0.001:
        irf_t = np.arange(0, irf_d, 1)   
    elif dt == 0.01:
        irf_t = np.arange(0, irf_d, 10)
    
    irf_s  = np.array([gamma_function(ti, tau, kapha=1, m=9) for ti in irf_t])  # sustained 
    irf_b  = np.array([biphasic_gamma(ti, tau, kapha1= 1, kapha2 =1.33, m1=9, m2=10, s=1) for ti in irf_t])  # onset - biphasic function

    # apply nonlinearity
    irf_s = apply_nonlinearity(irf_s, power)
    irf_b = apply_nonlinearity(irf_b, power)
    
    # Normalize IRF 
    #irf_s  /= np.sum(irf_s)
    #irf_b  /= np.sum(np.abs(irf_b))
    
    # set up return dictionary 
    S = {}  # predicted BOLD
    Z = {}  # modeled neuronal signal
    
    
    if 'sus' in irf_l:
        Z['sus'] = np.zeros((T, V))
        S['sus'] = np.zeros(((Z['sus'].shape[0]+bf.shape[0]-1),V,order))        
    if 'on' in irf_l:
        Z['on'] = np.zeros((T, V))
        S['on'] = np.zeros(((Z['on'].shape[0]+bf.shape[0]-1),V,order))
    if 'off' in irf_l:
        Z['off'] = np.zeros((T, V))
        S['off'] = np.zeros(((Z['off'].shape[0]+bf.shape[0]-1),V,order))
    
    # Modeled neuronal signal
    for idx in range(len(ons)):
        start    = ons[idx]
        duration = dur[idx]
        stim_m   = stim[idx]
        
        # Evaluate pRF tuning
        if not lin:
            resp = f_log(stim_m, mu_log, sig_log) 
        else:
            resp = f_lin(stim_m, mu_log, sig_log)
            
        if 'sus' in irf_l:
            # neuronal signal: normalized irf signal*log gaussian kernel value
            n_signal = irf_s[:,np.newaxis]*resp[np.newaxis,:] 
            for j in range(V):
                Z['sus'][int(start):int(start+duration),j] = n_signal[:int(duration),j]
        
        if 'on' in irf_l:
            n_signal = irf_b[:,np.newaxis]*resp[np.newaxis,:] 
            for j in range(V):
                Z['on'][int(start):int(start+duration),j] = n_signal[:int(duration),j]
                
        if 'off' in irf_l:
            n_signal = irf_b[:,np.newaxis]*resp[np.newaxis,:] 
            for j in range(V):
                s_idx = int(start + duration)
                if idx != len(ons) - 1:
                    end_idx = int(ons[idx + 1])
                else:
                    end_idx = Z['off'].shape[0]
                d1 = end_idx - s_idx
                d2 = min(d1, n_signal.shape[0])
                Z['off'][s_idx:s_idx + d2, j] = n_signal[:d2, j]

    
    # Convolve with hrf
    for k in range(order):
        for j in range(V):
            if 'sus' in irf_l:
                S['sus'][:,j,k] = convolve(Z['sus'][:,j], bf[:,k])
                S['sus'][:,j,k] = S['sus'][:,j,k] / np.max(np.abs(S['sus'][:,j,k]))
            if 'on' in irf_l:
                S['on'][:,j,k] = convolve(Z['on'][:,j], bf[:,k])
                S['on'][:,j,k] = S['on'][:,j,k] / np.max(np.abs(S['on'][:,j,k]))
            if 'off' in irf_l:
                S['off'][:,j,k] = convolve(Z['off'][:,j], bf[:,k])
                S['off'][:,j,k] = S['off'][:,j,k] / np.max(np.abs(S['off'][:,j,k]))
    
    # down sample signals
    
    mtr = int(TR/dt)        # micro TR 
    mto = math.ceil(mtr/2)  # micro TR sampling point
    
    i = np.arange(mto-1, n*mtr, mtr)
    if 'sus' in irf_l:
        S['sus'] = S['sus'][i,:,:]
    if 'on' in irf_l:
        S['on'] = S['on'][i,:,:]
    if 'off' in irf_l:
        S['off'] = S['off'][i,:,:]
    t = t_z[i]

    return S, Z, t, t_z

# function: neuronal model 
#-----------------------------------------------------------------------------#
def neuronal_signals(ons, dur, stim, TR, mtr, mu_log, sig_log, lin=False):
    """
    Compute Signals According to Neuronal Model
    Z, t = neuronal_signals(ons, dur, stim, TR, mtr, mu_log, sig_log)
    
        ons     - t x 1 vector; trial onsets [s]
        dur     - t x 1 vector; trial durations [s]
        stim    - t x 1 vector; trial stimuli (t = trials)
        TR      - float; repetition time of fMRI acquisition [s]
        mtr     - int; microtime resolution (= bins per TR)
        mu_log  - 1 x v vector; preferred numerosity in logarithmic space
        sig_log - 1 x v vector; tuning width in logarithmic space
        lin     - bool; indicating use of linear tuning functions
                  (in which case mu_log = mu_lin and sig_log = sig_lin)
        
        Z       - m x v matrix; neuronal signals tuned to stimuli
        t       - m x 1 vector; time vector at temporal resolution TR/mtr
    """
    
    # get maximum time points
    T = math.ceil(np.max(ons+dur))
    
    # calculate timing in microtime space
    dt    = TR/mtr
    min_t = []
    for i in range(len(dur)):
        min_t.append(np.min(dur[i]))
    m = min(min_t)
    if dt > m : #find the minimum dur and compare it with dt
        dt = 0.01 # temporal resolution for very short audiotry stimuli, 0.01 sec beep tone
    ons = np.round(np.array(ons)/dt)
    dur = np.round(np.array(dur)/dt)
    

    # compute neuronal signals
    v = mu_log.size
    Z = np.zeros((math.ceil(T/dt),v))
    t = np.arange(0,T,dt)
    for o,d,s in zip(ons,dur,stim):
        if not lin:
            Z[int(o):(int(o)+int(d)),:] = f_log(s, mu_log, sig_log)
        else:
            Z[int(o):(int(o)+int(d)),:] = f_lin(s, mu_log, sig_log)
    
    # due to the around up of ons, there are cases (int(o)+int(d)) - int(o) > int(d) 
    # To fill such false zero value in Z, check the values of previous and next in Z(axis=0)
    # If both are non zero, then it is false zero case. Fill it with the next value.
    
    for i in range(1, len(Z) - 1): 
        if not Z[i].all(): 
            # Check if both previous and next values are non-zero
            for k in range(0,v,1):
                if Z[i-1, k] != 0 and Z[i+1,k] !=0:
                    Z[i,k] = Z[i+1,k] 
    
    # return neuronal signals
    return Z, t

# function: hemodynamic signals
#-----------------------------------------------------------------------------#
def hemodynamic_signals(Z, t, n, TR ,mtr, mto=1, p=None, order=1):
    """
    Compute Signals According to Hemodynamic Model
    S, t = hemodynamic_signals(Z, t, n, TR, mtr, mto, p, order)
    
        Z      - m x v matrix; neuronal signals (see "neuronal_signals")
        t      - m x 1 vector; time vector (see "neuronal_signals")
        n      - int; number of fMRI scans acquired in run
        TR     - float; repetition time of fMRI acquisition [s]
        mtr    - int; microtime resolution (= bins per TR)
        mto    - int; microtime onset (= reference slice; default: 1)
        p      - list of floats; HRF parameters (default: see "PySPM.spm_hrf")
        order  - int; order of HRF basis set (default: 1; see "PySPM.spm_get_bf")
        
        S      - n x v x order array; hemodynamic signals after convolution
        t      - n x 1 vector; time vector after temporal down-sampling
    """
    
    # get hemodynamic response function
    dt = t[1]-t[0]
    print(f'\n -> Neuronal signal time inteval is {dt}')
    bf = PySPM.spm_get_bf(dt, 'HRF', p, order)
    
    # compute hemodynamic signals
    v = Z.shape[1]
    S = np.zeros(((Z.shape[0]+bf.shape[0]-1),v,order))
    for k in range(order):
        for j in range(v):
            S[:,j,k] = np.convolve(Z[:,j], bf[:,k])
            S[:,j,k] = S[:,j,k] / np.max(np.abs(S[:,j,k]))
    
    # check which mtr is used (slice number or minimum stimulus length)
    if dt != TR/mtr:
        mtr = int(TR/dt)
        mto = math.ceil(mtr/2)
    
    print(f' -Stimulus-wise design: mtr is {mtr} and mto is {mto}.')
        
    # add time points, if necessary
    if S.shape[0] < n*mtr:
        S = np.concatenate((S, np.zeros(((n*mtr-S.shape[0]),v,order))), axis=0)
    if t.shape[0] < n*mtr:
        t = np.concatenate((t, np.arange(np.max(t)+dt, np.max(t)+(n*mtr-t.shape[0]+1)*dt, dt)), axis=0)
    
    # down-sample signals temporally

    i = np.arange(mto-1, n*mtr, mtr)
    S = S[i,:,:]
    t = t[i]
    
    
    # return hemodynamic signals
    return S, t
    
# class: data set
#-----------------------------------------------------------------------------#
class DataSet:
    """
    A DataSet object is initialized by a data matrix, onsets/durations/stimuli,
    fMRI repetition time and a matrix of confound variables of no interest.
    """
    
    # function: initialize data setTR
    #-------------------------------------------------------------------------#
    def __init__(self, Y, ons, dur, stim, TR, X_c):
        """
        Initialize a Data Set
        ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
            
            Y    - n x v x r array; measured BOLD signals (n = scans, v = voxels) OR
                   any other type; if purpose is simulation (enter e.g. 0)
            ons  - list of arrays of floats; t x 1 vectors of onsets [s]
            dur  - list of arrays of floats; t x 1 vectors of durations [s]
            stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            TR   - float; repetition time of fMRI acquisition [s]
            X_c  - n x c x r array; confound regressors (c = variables, r = runs)
            
            ds   - a DataSet object
            o Y    - data matrix
            o ons  - run-wise onsets
            o dur  - run-wise durations
            o stim - run-wise stimuli
            o TR   - fMRI repetition time
            o X_c  - confound matrix
        """
        
        # store data set properties
        self.Y    = Y
        self.ons  = ons
        self.dur  = dur
        self.stim = stim
        self.TR   = TR
        self.X_c  = X_c
    
    # function: simulate data set
    #-------------------------------------------------------------------------#
    def simulate(self, mu, fwhm, mu_b=10, mu_c=1, s2_k=1, s2_j=0.1, s2_i=1, tau=0.001, hrf=None, CST=False):
        """
        Simulate Data across Scans, Voxels and Runs
        Y, S, X, B = ds.simulate(mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau, hrf)
            
            mu   - 1 x v vector; preferred numerosities in linear space (v = voxels)
            fwhm - 1 x v vector; tuning widths in linear space
            mu_b - float; expected value of signal betas
            mu_c - float; expected value of confound betas
            s2_k - float; between-voxel variance
            s2_j - float; between-run variance
            s2_i - float; within-run variance
            tau  - float; time constant, serial correlation
            hrf  - list of floats; HRF parameters (default: see "PySPM.spm_hrf")
            CST  - boolean; with or without CST model (default: False)
            
            Y    - n x v x r array; simulated BOLD signals
            S    - n x v x r array; predicted numerosity signals
            X    - n x p x r array; created design matrices
            B    - p x v x r array; sampled regression coefficients
        """
        
        # part 1: create design matrix
        #---------------------------------------------------------------------#
        n = self.X_c.shape[0]       # number of scans
        c = self.X_c.shape[1]       # number of variables
        r = self.X_c.shape[2]       # number of runs
        v = mu.size                 # number of voxels
        
        # preallocate design matrices
        if not CST:
            p = 1 + c + 1           # number of regression coefficients:
                                    # numerosity effect + confounds + implicit baseline (assumed cannonical HRF)
            S = np.zeros((n,v,r))   # prepare hemodynamic signal functions of r runs
            
        else:
            p = 3 + c + 1            # number of regression coefficients:
                                     # numerosity effects(sus,on,off) + confounds + implicit baseline (assumed cannonical HRF)
            S_s   = np.zeros((n,v,r))
            S_on  = np.zeros((n,v,r))
            S_off = np.zeros((n,v,r))
            
        X = np.zeros((n,p,r))
        
        # specify microtime resolution
        import EMPRISE
        mtr  = EMPRISE.mtr
        mto  = EMPRISE.mto
        del EMPRISE
        
        # transform tuning parameters
        mu_log, sig_log = lin2log(mu, fwhm)
        
        # for all runs
        for j in range(r):
            
            # calculate neuronal signals
            if not CST:
                Z, t = neuronal_signals(self.ons[j], self.dur[j], self.stim[j], \
                                    self.TR, mtr, mu_log, sig_log)
            
                # calculate hemodynamic signals
                S[:,:,[j]], t = hemodynamic_signals(Z, t, n, mtr, mto, p=hrf, order=1)
                
                # create design matrix for this run
                X[:,:,j] = np.c_[np.zeros((n,1)), self.X_c[:,:,j], np.ones((n,1))]
               
            else:
                # calculate hemodynamic signals
                S, Z, t, t_z   = CST_Bold_prediction(self.ons[j], self.dur[j], self.stim[j], 0.01, self.TR, n, ['on','off','sus'],mu_log,sig_log, 5, 0.5)
                S_s[:,:,[j]]   = S['sus']
                S_on[:,:,[j]]  = S['on']
                S_off[:,:,[j]] = S['off']
                
                del Z, t, t_z
                # create design matrix for this run
                X[:,:,j] = np.c_[np.zeros((n,3)), self.X_c[:,:,j], np.ones((n,1))]     
                
        # part 2: sample measured signals
        #---------------------------------------------------------------------#
        z = np.zeros(n)             # n x 1 zero vector
        
        # preallocate data matrices
        Y = np.zeros((n,v,r))       # measured signals
        B = np.zeros((p,v,r))       # regression coefficients
        
        # calculate temporal covariance
        V = sp.linalg.toeplitz(np.power(tau, np.arange(0,n))) #Symmatric
            
        # sample mean activations
        if not CST:
            B_mean = np.r_[np.random.normal(mu_b, math.sqrt(s2_k), size=(1,v)), \
                           np.random.normal(mu_c, math.sqrt(s2_k), size=(c,v)), \
                           np.random.normal(mu_b, math.sqrt(s2_k), size=(1,v))]
        else:
            B_mean = np.r_[np.random.normal(mu_b, math.sqrt(s2_k), size=(3,v)), \
                           np.random.normal(mu_c, math.sqrt(s2_k), size=(c,v)), \
                           np.random.normal(mu_b, math.sqrt(s2_k), size=(1,v))]            
        
        # for all runs 
        for j in range(r):
            
            # sample beta values
            B[:,:,j] = B_mean + np.random.normal(0, math.sqrt(s2_j), size=(p,v))
            
            # sample noise terms
            E = np.random.multivariate_normal(z, s2_i*V, size=v).T
            
            # simulate measured signal
            if not CST:
                Y[:,:,j] = S[:,:,j] @ np.diag(B[0,:,j]) + \
                           X[:,1:,j] @ B[1:,:,j] + E
            else:
                Y[:,:,j] = S_s[:,:,j]@np.diag(B[0,:,j]) + S_on[:,:,j]@np.diag(B[1,:,j]) +S_off[:,:,j]@np.diag(B[2,:,j]) + \
                           X[:,3:,j]@B[3:,:,j] + E
            # signal = signal due stimulus + signal due to confounds + noise
            # this is equivalent to the following formulation (but faster)
            # for k in range(v):
            #     X[:,0,j] = S[:,k,j]
            #     Y[:,:,j] = X[:,:,j]*B[:,:,j] + E
                    
        # return simulated signals
        self.Y = Y
        if not CST:
            del Z, t, E
            return Y, S, X, B
        else:
            del E
            return Y, S, X, B

    # function: maximum likelihood estimation with CST model
    #-------------------------------------------------------------------------#
    def estimate_CST_MLE(self, avg=[False, False], corr='iid', order=1, Q_set=None, irf_l = ['sus','on','off'], dt=0.01, mu_grid=None, sig_grid=None, fwhm_grid=None, tau_grid=None, epn_grid=None, lin=False):
        """
        Maximum Likelihood Estimation of Numerosity Tuning Parameters
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =
            ds.estimate_MLE(avg, corr, order, mu_grid, sig_grid, fwhm_grid, Q_set)
            
            avg       - list of bool; indicating whether signals are averaged
                                      (see "EMPRISE.average_signals")
            corr      - string; method for serial correlations ('iid' or 'ar1' for
                                "i.i.d. errors" or "AR(1) model")
            order     - int; order of the HRF model, must be 1, 2 or 3
                             (see "PySPM.spm_get_bf")
            Q_set     - list of matrices; covariance components for AR estimation
                                          (default: see below, part 2)
            irf_l     - list of irf channels; default: ['s','on','off']
            dt        - float; time resolution of neuronal signals
            mu_grid   - vector; candidate values for mu
            sig_grid  - vector; candidate values for sigma (in logarithmic space)
            fwhm_grid - vector; candidate values for fwhm (in linear space)
            tau_grid  - vector: candidate values for tau (CST model)
            epn_grid  - vector: cnadidate values for exponential compression power (CST model)
            lin       - bool; indicating use of linear tuning functions
            
            mu_est    - 1 x v vector; estimated numerosities in linear space
            fwhm_est  - 1 x v vector; estimated tuning widths in linear space
            tau_est   - 1 x v vector; estimated tau of CST model
            epn_est   - 1 x v vector; estimated exponential power of CST model
            beta1_est - 1 x v vector; estimated scaling factors for each vertex of sustaiend IRF
            beta2_est - 1 x v vector; esitmated scaling factors for each vertex of on/off transient IRF
            MLL_est   - 1 x v vector; maximum log-likelihood for model using
                                      the estimated tuning parameters
            MLL_null  - 1 x v vector; maximum log-likelihood for model using
                                      only confound and constant regressors
            MLL_const - 1 x v vector; maximum log-likelihood for model using
                                      only the constant regressor
            corr_est  - dict; specifying estimated covariance structure
            o Q       - 1 x q list of matrices; additive covariance components
            o h       - r x q matrix; multiplicative variance factors
            o V       - n x n x r array; estimated covariance matrix
        
        Note: Only one of the variables "sig_grid" and "fwhm_grid" should be
        specified. If both are (not) specified, "sig_grid" is prioritized.
        """
        
        # part 1: prepare data and design matrix
        #---------------------------------------------------------------------#
        n = self.Y.shape[0]; n_orig = n     # number of scans
        v = self.Y.shape[1]                 # number of voxels
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # order of HRF regressors
        l = len(irf_l)                      # number of irf channels
        
        # if averaging across runs or epochs, regress out confounds first
        if avg[0] or avg[1]:
            Y = np.zeros(self.Y.shape)
            for j in range(r):
                glm      = PySPM.GLM(self.Y[:,:,j], np.c_[self.X_c[:,:,j], np.ones((n,1))])
                B_est    = glm.OLS()
                # subtract confounds from signal, then re-add constant regressor
                Y[:,:,j] = glm.Y - glm.X @ B_est + glm.X[:,[-1]] @ B_est[[-1],:]
        
        # otherwise, use signals without further manipulation
        else:
            Y = self.Y
        
        # then average signals across runs and/or epochs, if applicable
        import EMPRISE_audio
        Y, t = EMPRISE_audio.average_signals(Y, t=None, avg=avg)
        if avg[1]: n = Y.shape[0]           # update number of scans
        
        # since design is identical across runs, always use first run
        ons = self.ons; dur = self.dur; stim = self.stim;
        ons = ons[0];   dur = dur[0];   stim = stim[0];
        
        # if averaging across epochs, correct onsets to first epoch
        if avg[1]:
            ons, dur, stim = EMPRISE_audio.correct_onsets(ons, dur, stim)
        
        # if averaging across runs or epochs, exclude confounds
        if avg[0] or avg[1]:
            p = l*o + 1           # the number of channels * the number of HRF order
            X = np.c_[np.zeros((n,l*o)), np.ones((n,1))]
            if not avg[0]:        # no average across runs
                X = np.repeat(np.expand_dims(X, 2), r, axis=2)
        
        # otherwise, add confounds to design matrix
        else:
            p = l*o + c + 1       # number of regression coefficients
            X = np.zeros((n,p,r))
            for j in range(r):    # run-wise design matrices
                X[:,:,j] = np.c_[np.zeros((n,l*o)), self.X_c[:,:,j], np.ones((n,1))]
        
        # specify further parameters
        mtr = EMPRISE_audio.mtr       # microtime resolution (= bins per TR)
        mto = EMPRISE_audio.mto       # microtime onset (= reference slice)

        del EMPRISE_audio
        
        # part 2: prepare correlation matrix
        #---------------------------------------------------------------------#
        if Q_set is None:
            a     = 0.4         # AR parameter
            Q_set = [np.eye(n), # covariance components
                     sp.linalg.toeplitz(np.power(a, np.arange(0,n))) - np.eye(n)]
        if Q_set is not None:
            q     = len(Q_set)
        
        # prepare condition regressors
        if corr == 'ar1':
            # create names, onsets, durations
            names     = ['1', '2', '3', '4', '5', '20']
            onsets    = []
            durations = []
            # collect onsets/durations from first run
            for name in names:
                onsets.append([o for (o,s) in zip(ons,stim) if s == int(name)])
                durations.append([d for (d,s) in zip(dur,stim) if s == int(name)])
            # call PySPM to create design matrix
            X_d, L_d = PySPM.get_des_mat(names, onsets, durations, \
                                         settings={'n': n, 'TR': self.TR, 'mtr': mtr, 'mto': mto, 'HRF': 'spm_hrf'})
            # add confound variables to design
            if avg[0] or avg[1]:
                X0 = np.c_[X_d, np.ones((n,1))]
                if not avg[0]:
                    X0 = np.repeat(np.expand_dims(X0, 2), r, axis=2)
            else:
                X0 = np.zeros((n,len(names)+c+1,r))
                for j in range(r):
                    X0[:,:,j] = np.c_[X_d, self.X_c[:,:,j], np.ones((n,1))]
            # announce ReML estimation
            print('\n-> Restricted maximum likelihood estimation ({} rows, {} columns):'. \
                  format(n, v))
        
        # prepare correlation matrices
        if corr == 'iid':
            # invoke identity matrices, if i.i.d. errors
            if avg[0]:
                h   = np.array([1]+[0 for x in range(q-1)])
                V   = np.eye(n)
                P   = V
                ldV = 0
            else:
                h   = np.tile(np.array([1]+[0 for x in range(q-1)]), (r,1))
                V   = np.repeat(np.expand_dims(np.eye(n), 2), r, axis=2)
                P   = V
                ldV = np.zeros(r)
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': h, 'V': V}
        elif corr == 'ar1':
            # perform ReML estimation, if AR(1) process
            if avg[0]:
                V, Eh, Ph, F, Acc, Com = PySPM.spm_reml(Y, X0, Q_set)
                V   = (n/np.trace(V)) * V
                P   = np.linalg.inv(V)
                ldV = np.linalg.slogdet(V)[1]
            else:
                Eh  = np.zeros((q,r))
                V   = np.zeros((n,n,r))
                P   = np.zeros((n,n,r))
                ldV = np.zeros(r)
                for j in range(r):
                    print('   Run {}:'.format(j+1))
                    V[:,:,j], Eh[:,[j]], Ph, F, Acc, Com = PySPM.spm_reml(Y[:,:,j], X0[:,:,j], Q_set)
                    V[:,:,j] = (n/np.trace(V[:,:,j])) * V[:,:,j]
                    P[:,:,j] = np.linalg.inv(V[:,:,j])
                    ldV[j]   = np.linalg.slogdet(V[:,:,j])[1]
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': Eh.T, 'V': V}
            del Eh, Ph, F, Acc, Com
        else:
            err_msg = 'Unknown correlation method: "{}". Method must be "iid" or "ar1".'
            raise ValueError(err_msg.format(corr))
        
        # part 3: prepare grid search
        #---------------------------------------------------------------------#
        if mu_grid is None:                 # range: mu = 0.8,...,5.2 | 20
            mu_grid = np.concatenate((np.arange(0.80, 5.25, 0.05), np.array([20])))
        if sig_grid is None:
            if fwhm_grid is None:           # range: sig_log = 0.05,...,3.00
                sig_grid  = np.arange(0.05, 3.05, 0.05)
        # Explanation: If "sig_grid" and "fwhm_grid" are not specified,
        # then "sig_grid" is specified (see comment in help text).
        elif sig_grid is not None:
            if fwhm_grid is not None:
                fwhm_grid = None
        # Explanation: If "sig_grid" and "fwhm_grid" are both specified,
        # then "fwhm_grid" is disspecified (see comment in help text).

        if tau_grid is None:
            tau_grid = np.arange(3, 10.05, 0.05)
            
        if epn_grid is None:
            epn_grid = np.arange(0.1, 1.05, 0.05)  

        # specify parameter grid
        mu   = mu_grid
        sig  = sig_grid
        fwhm = fwhm_grid
        tau  = tau_grid
        epn  = epn_grid

        # initialize parameters
        mu_est    = np.zeros(v)
        fwhm_est  = np.zeros(v)
        tau_est   = np.zeros(v)
        epn_est   = np.zeros(v)
        beta1_est = np.zeros(v)    # S_s
        beta2_est = np.zeros(v)    # S_on
        beta3_est = np.zeros(v)    # S_off

        # prepare maximum likelihood
        MLL_est = -np.inf * np.ones(v)

        # part 4: perform grid search
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation ({} runs, {} voxels, {} scans, '. \
              format(r, v, n_orig), end='')
        print('\n   {}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))

        # for all values of mu
        for k1, m in enumerate(mu):
            
            # define current grid
            if sig is None:
                mus     = m * np.ones(fwhm.size)
                mu_log  = math.log(m) * np.ones(fwhm.size)
                fwhms   = fwhm
                sig_log = lin2log(mus, fwhms)[1]
            else:
                mus     = m * np.ones(sig.size)
                mu_log  = math.log(m) * np.ones(sig.size)
                sig_log = sig
                fwhms   = log2lin(mu_log, sig_log)[1]
            
            if lin:
                sig_lin = fwhm2sig(fwhms)
            
            # Fixed: was tau.szie and epn.size
            MLL  = np.zeros((tau.size, epn.size, fwhms.size, v))
            beta1 = np.zeros((tau.size, epn.size, fwhms.size, v))
            beta2 = np.zeros((tau.size, epn.size, fwhms.size, v))
            beta3 = np.zeros((tau.size, epn.size, fwhms.size, v))
            
            # display message
            print('   - grid chunk {} out of {}:'.format(k1+1, mu.size), end=' ')
            
            for i1, tu in enumerate(tau):
                for i2, pn in enumerate(epn):
                    print('mu = {:.2f}, tau = {:.2f}, epn = {:.2f}, fwhm = {:.2f}, ..., {:.2f}'.format(m, tu, pn, fwhms[0], fwhms[-1]))
            
                    # predict time courses
                    if not lin:
                        S, Z, t, t_z = CST_Bold_prediction(ons, dur, stim, dt, self.TR, n, irf_l, mu_log, sig_log, tu, pn, lin=False, order=1)
                    else:
                        S, Z, t, t_z = CST_Bold_prediction(ons, dur, stim, dt, self.TR, n, irf_l, mu_log, sig_lin, tu, pn, lin=True, order=1)
                    del Z, t, t_z

                    # for all values of fwhm
                    for k2, f in enumerate(fwhms):
                        
                        # generate design & estimate GLM
                        if avg[0]:
                            if l == 1:
                                X[:,0:o] = S[irf_l[0]][:,k2,:]
                            elif l == 2:
                                X[:,0:o]   = S[irf_l[0]][:,k2,:]
                                X[:,o:o*2] = S[irf_l[1]][:,k2,:]
                            elif l == 3:
                                X[:,0:o]     = S[irf_l[0]][:,k2,:]
                                X[:,o:o*2]   = S[irf_l[1]][:,k2,:]
                                X[:,o*2:o*3] = S[irf_l[2]][:,k2,:]
                                
                            glm = PySPM.GLM(Y, X, P=P, ldV=ldV)
                            
                            MLL[i1, i2, k2,:]   = glm.MLL()
                            beta1[i1, i2, k2,:] = glm.WLS()[0,:]
                            if l == 2:
                                beta2[i1, i2, k2,:] = glm.WLS()[1,:]
                            if l == 3:
                                beta2[i1, i2, k2,:] = glm.WLS()[1,:]
                                beta3[i1, i2, k2,:] = glm.WLS()[2,:]
                        else:
                            for j in range(r):
                                if l == 1:
                                    X[:,0:o,j] = S[irf_l[0]][:,k2,:]
                                elif l == 2:
                                    X[:,0:o,j]   = S[irf_l[0]][:,k2,:]
                                    X[:,o:o*2,j] = S[irf_l[1]][:,k2,:]
                                elif l == 3:
                                    X[:,0:o,j]     = S[irf_l[0]][:,k2,:]
                                    X[:,o:o*2,j]   = S[irf_l[1]][:,k2,:]
                                    X[:,o*2:o*3,j] = S[irf_l[2]][:,k2,:]
                                    
                                glm = PySPM.GLM(Y[:,:,j], X[:,:,j], P=P[:,:,j], ldV=ldV[j])

                                MLL[i1, i2, k2,:]   = MLL[i1, i2, k2,:]  + glm.MLL()
                                beta1[i1, i2, k2,:] = beta1[i1, i2, k2,:] + glm.WLS()[0,:]/r
                                if l == 2:
                                    beta2[i1, i2, k2,:] = beta2[i1, i2, k2,:] + glm.WLS()[1,:]/r
                                if l == 3:
                                    beta2[i1, i2, k2,:] = beta2[i1, i2, k2,:] + glm.WLS()[1,:]/r
                                    beta3[i1, i2, k2,:] = beta3[i1, i2, k2,:] + glm.WLS()[2,:]/r

            # find maximum likelihood estimates
            MLL_flat = MLL.reshape(-1, v)             # Flatten first 3 dimensions
            k_max_flat = np.argmax(MLL_flat, axis=0)  # Find max index for each voxel
            
            # Convert flat indices back to 3D indices
            k_max_tau, k_max_epn, k_max_fwhm = np.unravel_index(k_max_flat, MLL.shape[:3])
            
            # Get maximum MLL values
            MLL_max = MLL_flat[k_max_flat, range(v)]
            
            # Update estimates where MLL is better
            update_mask = MLL_max > MLL_est
            
            mu_est[update_mask]   = m
            tau_est[update_mask]  = tau[k_max_tau[update_mask]]
            epn_est[update_mask]  = epn[k_max_epn[update_mask]]
            fwhm_est[update_mask] = fwhms[k_max_fwhm[update_mask]]
        
            for i, update in enumerate(update_mask):
                if update:
                    beta1_est[i] = beta1[k_max_tau[i], k_max_epn[i], k_max_fwhm[i], i]
                    beta2_est[i] = beta2[k_max_tau[i], k_max_epn[i], k_max_fwhm[i], i]
                    beta3_est[i] = beta3[k_max_tau[i], k_max_epn[i], k_max_fwhm[i], i]
            
            MLL_est[update_mask] = MLL_max[update_mask]
        
        # part 5: estimate reduced models
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation of reduced models', end='')
        print('\n   ({}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # estimate model using only confound and constant regressors
        print('   - no-numerosity model ... ', end='')
        if avg[0]:
            MLL_null = PySPM.GLM(Y, X[:,l*o:], V).MLL()
        else:
            MLL_null = np.zeros(v)
            for j in range(r):
                MLL_null = MLL_null + PySPM.GLM(Y[:,:,j], X[:,l*o:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # estimate model using only the constant regressor
        print('   - baseline-only model ... ', end='')
        if avg[0]:
            MLL_const = PySPM.GLM(Y, X[:,-1:], V).MLL()
        else:
            MLL_const = np.zeros(v)
            for j in range(r):
                MLL_const = MLL_const + PySPM.GLM(Y[:,:,j], X[:,-1:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # return estimated parameters
        return mu_est, fwhm_est, beta1_est, beta2_est, beta3_est, MLL_est, MLL_null, MLL_const, corr_est
    
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def estimate_MLE(self, avg=[False, False], corr='iid', order=1, Q_set=None, mu_grid=None, sig_grid=None, fwhm_grid=None, lin=False):
        """
        Maximum Likelihood Estimation of Numerosity Tuning Parameters
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =
            ds.estimate_MLE(avg, corr, order, mu_grid, sig_grid, fwhm_grid, Q_set)
            
            avg       - list of bool; indicating whether signals are averaged
                                      (see "EMPRISE.average_signals")
            corr      - string; method for serial correlations ('iid' or 'ar1' for
                                "i.i.d. errors" or "AR(1) model")
            order     - int; order of the HRF model, must be 1, 2 or 3
                             (see "PySPM.spm_get_bf")
            Q_set     - list of matrices; covariance components for AR estimation
                                          (default: see below, part 2)
            mu_grid   - vector; candidate values for mu
            sig_grid  - vector; candidate values for sigma (in logarithmic space)
            fwhm_grid - vector; candidate values for fwhm (in linear space)
            lin       - bool; indicating use of linear tuning functions
            
            mu_est    - 1 x v vector; estimated numerosities in linear space
            fwhm_est  - 1 x v vector; estimated tuning widths in linear space
            beta_est  - 1 x v vector; estimated scaling factors for each voxel
            MLL_est   - 1 x v vector; maximum log-likelihood for model using
                                      the estimated tuning parameters
            MLL_null  - 1 x v vector; maximum log-likelihood for model using
                                      only confound and constant regressors
            MLL_const - 1 x v vector; maximum log-likelihood for model using
                                      only the constant regressor
            corr_est  - dict; specifying estimated covariance structure
            o Q       - 1 x q list of matrices; additive covariance components
            o h       - r x q matrix; multiplicative variance factors
            o V       - n x n x r array; estimated covariance matrix
        
        Note: Only one of the variables "sig_grid" and "fwhm_grid" should be
        specified. If both are (not) specified, "sig_grid" is prioritized.
        """
        
        # part 1: prepare data and design matrix
        #---------------------------------------------------------------------#
        n = self.Y.shape[0]; n_orig = n     # number of scans
        v = self.Y.shape[1]                 # number of voxels
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # number of HRF regressors
        
        # if averaging across runs or epochs, regress out confounds first
        if avg[0] or avg[1]:
            Y = np.zeros(self.Y.shape)
            for j in range(r):
                glm      = PySPM.GLM(self.Y[:,:,j], np.c_[self.X_c[:,:,j], np.ones((n,1))])
                B_est    = glm.OLS()
                # subtract confounds from signal, then re-add constant regressor
                Y[:,:,j] = glm.Y - glm.X @ B_est + glm.X[:,[-1]] @ B_est[[-1],:]
        
        # otherwise, use signals without further manipulation
        else:
            Y = self.Y
        
        # then average signals across runs and/or epochs, if applicable
        import EMPRISE_audio
        Y, t = EMPRISE_audio.average_signals(Y, t=None, avg=avg)
        if avg[1]: n = Y.shape[0]           # update number of scans
        
        # since design is identical across runs, always use first run
        ons = self.ons; dur = self.dur; stim = self.stim;
        ons = ons[0];   dur = dur[0];   stim = stim[0];
        
        # if averaging across epochs, correct onsets to first epoch
        if avg[1]:
            ons, dur, stim = EMPRISE_audio.correct_onsets(ons, dur, stim)
        
        # if averaging across runs or epochs, exclude confounds
        if avg[0] or avg[1]:
            p = o + 1           # number of regression coefficients
            X = np.c_[np.zeros((n,o)), np.ones((n,1))]
            if not avg[0]:      # run-independent design matrices
                X = np.repeat(np.expand_dims(X, 2), r, axis=2)
        
        # otherwise, add confounds to design matrix
        else:
            p = o + c + 1       # number of regression coefficients
            X = np.zeros((n,p,r))
            for j in range(r):  # run-wise design matrices
                X[:,:,j] = np.c_[np.zeros((n,o)), self.X_c[:,:,j], np.ones((n,1))]
        
        # specify further parameters
        mtr = EMPRISE_audio.mtr       # microtime resolution (= bins per TR)
        mto = EMPRISE_audio.mto       # microtime onset (= reference slice)

        del EMPRISE_audio
        
        # part 2: prepare correlation matrix
        #---------------------------------------------------------------------#
        if Q_set is None:
            a     = 0.4         # AR parameter
            Q_set = [np.eye(n), # covariance components
                     sp.linalg.toeplitz(np.power(a, np.arange(0,n))) - np.eye(n)]
        if Q_set is not None:
            q     = len(Q_set)
        
        # prepare condition regressors
        if corr == 'ar1':
            # create names, onsets, durations
            names     = ['1', '2', '3', '4', '5', '20']
            onsets    = []
            durations = []
            # collect onsets/durations from first run
            for name in names:
                onsets.append([o for (o,s) in zip(ons,stim) if s == int(name)])
                durations.append([d for (d,s) in zip(dur,stim) if s == int(name)])
            # call PySPM to create design matrix
            X_d, L_d = PySPM.get_des_mat(names, onsets, durations, \
                                         settings={'n': n, 'TR': self.TR, 'mtr': mtr, 'mto': mto, 'HRF': 'spm_hrf'})
            # add confound variables to design
            if avg[0] or avg[1]:
                X0 = np.c_[X_d, np.ones((n,1))]
                if not avg[0]:
                    X0 = np.repeat(np.expand_dims(X0, 2), r, axis=2)
            else:
                X0 = np.zeros((n,len(names)+c+1,r))
                for j in range(r):
                    X0[:,:,j] = np.c_[X_d, self.X_c[:,:,j], np.ones((n,1))]
            # announce ReML estimation
            print('\n-> Restricted maximum likelihood estimation ({} rows, {} columns):'. \
                  format(n, v))
        
        # prepare correlation matrices
        if corr == 'iid':
            # invoke identity matrices, if i.i.d. errors
            if avg[0]:
                h   = np.array([1]+[0 for x in range(q-1)])
                V   = np.eye(n)
                P   = V
                ldV = 0
            else:
                h   = np.tile(np.array([1]+[0 for x in range(q-1)]), (r,1))
                V   = np.repeat(np.expand_dims(np.eye(n), 2), r, axis=2)
                P   = V
                ldV = np.zeros(r)
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': h, 'V': V}
        elif corr == 'ar1':
            # perform ReML estimation, if AR(1) process
            if avg[0]:
                V, Eh, Ph, F, Acc, Com = PySPM.spm_reml(Y, X0, Q_set)
                V   = (n/np.trace(V)) * V
                P   = np.linalg.inv(V)
                ldV = np.linalg.slogdet(V)[1]
            else:
                Eh  = np.zeros((q,r))
                V   = np.zeros((n,n,r))
                P   = np.zeros((n,n,r))
                ldV = np.zeros(r)
                for j in range(r):
                    print('   Run {}:'.format(j+1))
                    V[:,:,j], Eh[:,[j]], Ph, F, Acc, Com = PySPM.spm_reml(Y[:,:,j], X0[:,:,j], Q_set)
                    V[:,:,j] = (n/np.trace(V[:,:,j])) * V[:,:,j]
                    P[:,:,j] = np.linalg.inv(V[:,:,j])
                    ldV[j]   = np.linalg.slogdet(V[:,:,j])[1]
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': Eh.T, 'V': V}
            del Eh, Ph, F, Acc, Com
        else:
            err_msg = 'Unknown correlation method: "{}". Method must be "iid" or "ar1".'
            raise ValueError(err_msg.format(corr))
        
        # part 3: prepare grid search
        #---------------------------------------------------------------------#
        if mu_grid is None:                 # range: mu = 0.8,...,5.2 | 20
            mu_grid = np.concatenate((np.arange(0.80, 5.25, 0.05), np.array([20])))
        if sig_grid is None:
            if fwhm_grid is None:           # range: sig_log = 0.05,...,3.00
                sig_grid  = np.arange(0.05, 3.05, 0.05)
        # Explanation: If "sig_grid" and "fwhm_grid" are not specified,
        # then "sig_grid" is specified (see comment in help text).
        elif sig_grid is not None:
            if fwhm_grid is not None:
                fwhm_grid = None
        # Explanation: If "sig_grid" and "fwhm_grid" are both specified,
        # then "fwhm_grid" is disspecified (see comment in help text).
        
        # specify parameter grid
        mu   = mu_grid
        sig  = sig_grid
        fwhm = fwhm_grid
        
        # initialize parameters
        mu_est   = np.zeros(v)
        fwhm_est = np.zeros(v)
        beta_est = np.zeros(v)
        
        # prepare maximum likelihood
        MLL_est = -np.inf*np.ones(v)
        
        # part 4: perform grid search
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation ({} runs, {} voxels, {} scans, '. \
              format(r, v, n_orig), end='')
        print('\n   {}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # for all values of mu
        for k1, m in enumerate(mu):
            
            # define current grid
            if sig is None:
                mus     = m*np.ones(fwhm.size)
                mu_log  = math.log(m)*np.ones(fwhm.size)
                fwhms   = fwhm
                sig_log = lin2log(mus, fwhms)[1]
            else:
                mus     = m*np.ones(sig.size)
                mu_log  = math.log(m)*np.ones(sig.size)
                sig_log = sig
                fwhms   = log2lin(mu_log, sig_log)[1]
            if lin:
                sig_lin = fwhm2sig(fwhms)
            MLL  = np.zeros((fwhms.size,v))
            beta = np.zeros((fwhms.size,v))
            
            # display message
            print('   - grid chunk {} out of {}:'.format(k1+1, mu.size), end=' ')
            print('mu = {:.2f}, fwhm = {:.2f}, ..., {:.2f}'.format(m, fwhms[0], fwhms[-1]))
            
            # predict time courses
            if not lin:
                Z, t = neuronal_signals(ons, dur, stim, self.TR, mtr, mu_log, sig_log, lin=False)
            else:
                Z, t = neuronal_signals(ons, dur, stim, self.TR, mtr, mus, sig_lin, lin=True)
            if True:
                S, t = hemodynamic_signals(Z, t, n, self.TR, mtr, mto, p=None, order=o)
            del Z, t
            
            # for all values of fwhm
            for k2, f in enumerate(fwhms):
                
                # generate design & estimate GLM
                if avg[0]:
                    X[:,0:o]   = S[:,k2,:]
                    glm        = PySPM.GLM(Y, X, P=P, ldV=ldV)
                    MLL[k2,:]  = glm.MLL()
                    beta[k2,:] = glm.WLS()[0,:]
                else:
                    for j in range(r):
                        X[:,0:o,j] = S[:,k2,:]
                        glm        = PySPM.GLM(Y[:,:,j], X[:,:,j], P=P[:,:,j], ldV=ldV[j])
                        MLL[k2,:]  = MLL[k2,:]  + glm.MLL()
                        beta[k2,:] = beta[k2,:] + glm.WLS()[0,:]/r
            
            # find maximum likelihood estimates
            k_max   = np.argmax(MLL, axis=0)
            MLL_max = MLL[k_max, range(v)]
            
            # update, if MLL larger than for previous MLEs
            mu_est[MLL_max>MLL_est]   = m
            fwhm_est[MLL_max>MLL_est] = fwhms[k_max][MLL_max>MLL_est]
            beta_est[MLL_max>MLL_est] = beta[k_max[MLL_max>MLL_est],MLL_max>MLL_est]
            MLL_est[MLL_max>MLL_est]  = MLL_max[MLL_max>MLL_est]
        
        # part 5: estimate reduced models
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation of reduced models', end='')
        print('\n   ({}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # estimate model using only confound and constant regressors
        print('   - no-numerosity model ... ', end='')
        if avg[0]:
            MLL_null = PySPM.GLM(Y, X[:,o:], V).MLL()
        else:
            MLL_null = np.zeros(v)
            for j in range(r):
                MLL_null = MLL_null + PySPM.GLM(Y[:,:,j], X[:,o:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # estimate model using only the constant regressor
        print('   - baseline-only model ... ', end='')
        if avg[0]:
            MLL_const = PySPM.GLM(Y, X[:,-1:], V).MLL()
        else:
            MLL_const = np.zeros(v)
            for j in range(r):
                MLL_const = MLL_const + PySPM.GLM(Y[:,:,j], X[:,-1:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # return estimated parameters
        return mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est

    
    # function: number of free parameters
    #-------------------------------------------------------------------------#
    def free_parameters(self, avg=[False, False], corr='iid', order=1, CST = None):
        """
        Number of Free Parameters for Maximum Likelihood Estimation
        k_est, k_null, k_const = ds.free_parameters(avg, corr, order)
            
            avg     - list of bool; see "estimate_MLE"
            corr    - string; see "estimate_MLE"
            order   - int; see "estimate_MLE"
            CST     - list of cahnnels (e.g. ['sus','on','off'], default: None)
            
            k_est   - int; number of free parameters for
                           model estimating tuning parameters
            k_null  - int; number of free parameters for
                           model using only confound and constant regressors
            k_const - int; number of free parameters for
                           model using only the constant regressor
        """
        
        # get data dimensions
        #---------------------------------------------------------------------#
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # number of HRF regressors
        if CST:
            l = len(CST)
        else:
            l =1
        
        # calculate parameters
        #---------------------------------------------------------------------#
        r = [r,1][int(avg[0])]              # number of runs after averaging
        c = [c,0][int(np.max(avg))]         # number of confounds after averaging
        p = l*o + c + 1                     # number of regression coefficients
        k_est   = 2 + r*(p+1)               # 2+ -> tuning parameters
        k_null  =     r*(p-l*o+1)           # +1 -> noise variance
        k_const =     r*(p-l*o-c+1)         # r* -> per each run
                                            # -o -> w/o tuning regressors
        # return free parameters            # -c -> w/o confound regressors
        #---------------------------------------------------------------------#
        return k_est, k_null, k_const
