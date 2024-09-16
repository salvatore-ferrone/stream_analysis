"""

A module for comparing the density profiles of the control group to the group with gaps and seeing where the under densities are. Log differences only. 


"""
import stream_analysis as sa 
import numpy as np 

####### COMPUTATIONS
def log_difference_detection(profiles,noise_factor,sigma):
    """Return the noise filtered profiles, noise level and the detection candidates.
    """
    CENTERS,CONTROL,COMPARE         =   profiles
    noiselevel                      =   sa.density_profile_gap_detections.noise_log_counts(CONTROL)
    LOGDIFF                         =   np.log10(COMPARE/CONTROL)
    noise_filt                      =   CONTROL > noise_factor*noiselevel
    centers,control,compare,logdiff =   sa.density_profile_gap_detections.apply_filter(noise_filt,[CENTERS,CONTROL,COMPARE,LOGDIFF])
    noise_filtered_profiles         =   centers,control,compare
    ############################
    # zscore filter
    myz                             =   sa.density_profile_gap_detections.zscores(logdiff)
    candidates                      =   myz < -sigma
    return noise_filtered_profiles,noiselevel,candidates

def apply_box_car_multiple_times(signal,box_length,N_apply):
    outsignal = signal.copy()
    i=0
    while i < N_apply:
        outsignal = sa.density_profile_gap_detections.median_box_car(outsignal,box_length)
        i+=1
    return outsignal


def clean_up_stream_profiles(x,c0,c1,box_length,N_APPlY,do_cross_correlation):
    """Return the stream density profiles after (1) projecting, (2) applying a boxcar filter (3) cross correlating for a better match up.

    Parameters
    ----------
    box_length : int
        length of box in indices
    N_APPlY : int
        number of times to apply the boxcar filter
    do_cross_correlation : bool
        consider cross correlation

    Returns
    -------
    np.ndarray,np.ndarray,np.ndarray
        the positions and the profiles of the two streams
    """
    
    # get bins that have at least one count in both profiles
    cond                    =   counts_only_filter(c0,c1)
    CENTERS,CONTROL,COMPARE =   apply_filter(cond,[x,c0,c1])
    # now apply the box car filter
    CONTROL_B = apply_box_car_multiple_times(CONTROL,box_length,N_APPlY)
    COMPARE_B = apply_box_car_multiple_times(COMPARE,box_length,N_APPlY)
    # cross correlate 
    if do_cross_correlation:
        COMPARE_B=cross_correlation_shift(CENTERS,CONTROL_B,COMPARE_B)    
    return CENTERS,CONTROL_B,COMPARE_B


def get_profile(xT,yT,NP,xlims,ylims):
    # prepare the profiles
    xedges,yedges       =   sa.plotters.binned_density.get_edges(NP,xlims,ylims)
    XX_2D,_,H_2D        =   sa.plotters.binned_density.get_2D_density(xT,yT,xedges,yedges)   
    counts = H_2D.sum(axis=0)
    centers = XX_2D[0]
    return centers,counts

def counts_only_filter(counts_control,counts_test):
    cond0 = counts_control > 0
    cond1 = counts_test > 0
    cond = cond0 & cond1
    return cond

def apply_filter(myfilter,mylist):
    myshape = myfilter.shape
    for x in mylist:    
        assert(type(x) == np.ndarray)
        assert(x.shape == myshape)
    assert(type(mylist) == list)
    assert(type(myfilter) == np.ndarray)
    return [array[myfilter] for array in mylist]

def noise_log_counts(counts):
    """Return the noise level of the log counts."""
    return 1/(np.log(10)*np.sqrt(counts))

def median_box_car(xx,boxsize):
    assert(type(xx) == np.ndarray)
    XXout=xx.copy()
    n = xx.shape[0]
    for i in range(0+boxsize,n-boxsize,1):
        XXout[i] = np.median(xx[i-boxsize:i+boxsize])
    return XXout

def zscores(data):
    # return (data - np.mean(data))/np.std(data)
    return (data)/np.std(data)

def cross_correlation_shift(centers,control,compare):
    """Cross correlate two arrays and find the shift that maximizes the cross-correlation.

    Parameters
    ----------
    centers : np.ndarray
        The positions of the centers of the bins.
    control : np.ndarray
        The control signal
    compare : np.ndarray
        The signal to compare to the control signal

    Returns
    -------
    np.ndarray
        control shifted to match compare
    """
    cross_correlation = np.correlate(control, compare, mode='full')
    # Find the index of the maximum value in the cross-correlation array
    max_index = np.argmax(cross_correlation)
    # Calculate the offset
    nstep= (max_index - (len(compare) - 1))
    
    xstep = np.mean(np.diff(centers))
    offset = xstep*nstep
    # interpolate back onto original grid
    compare = np.interp(centers, centers + offset, compare)
    return compare