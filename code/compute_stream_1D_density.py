import numpy as np
import StreamOrbitCoords as SOC
import data_extractors as DE
import filters




#####################################################
####################### LOOPS #######################
#####################################################
def obtain_tau(i, path_stream_orbit, hostorbit, t_stamps, period):
    """
    Given a snapshot index, this function projects the stream onto the host orbit 
    and calculates tau. Tau is the difference in time coordinate of a position on 
    the orbit and the current time.

    Parameters
    ----------
    i : int
        The snapshot index.
    path_stream_orbit : str
        The path to the stream orbit file.
    hostorbit : tuple
        A tuple containing the host orbit data (thost, xhost, yhost, zhost, vxhost, vyhost, vzhost).
    t_stamps : array_like
        An array of time stamps.
    period : float
        The period for filtering the orbit.

    Returns
    -------
    tau : ndarray
        The time difference from ahead/behind for each position in the stream.

    """
    # extract stream positions and orbit 
    xp, yp, zp, _, _, _ = DE.extract_stream_from_path(i, path_stream_orbit)
    thost, xhost, yhost, zhost, vxhost, vyhost, vzhost = hostorbit
    orbit = xhost, yhost, zhost, vxhost, vyhost, vzhost
    # filter orbit by dynamical time
    t_index = np.argmin(np.abs(thost - t_stamps[i].value))
    tOrb, xOrb, yOrb, zOrb, _, _, _ = filters.orbit_by_fixed_time(thost, orbit, t_index, period)
    # get closest orbital index of particles to stream
    indexes = SOC.get_closests_orbital_index(xOrb, yOrb, zOrb, xp, yp, zp)
    # calculate time difference from ahead/behind
    tau = tOrb[indexes] - t_stamps[i].value
    return tau


##############################################################
######################### COMPUTATIONS #######################
##############################################################
def make_2D_mask(X,Y,envelope_time_stamps,top_envelop,bottom_envelop):
    """
    X: np.ndarray
        The time stamps.
    Y: np.ndarray
        The tau values.
    """
    
    mask = np.zeros((X.shape[0],Y.shape[0]),dtype=bool)
    for i in range(X.shape[0]):
        mytime = X[i]
        index = np.argmin(np.abs(envelope_time_stamps - mytime))
        top,bottom = top_envelop[index],bottom_envelop[index]
        above,below = Y > top, Y < bottom
        outside = np.logical_or(above,below)
        mask[i] = outside
    return mask

    
def construct_2D_density_map(projected_tau,time_stamps,tau_max):
    
    # Obtain The Base Parameters
    Nstamps, NP = projected_tau.shape
    NBINS = int(np.ceil(np.sqrt(NP)))
    
    # make the bin edges
    bin_edges = np.linspace(-tau_max,tau_max,NBINS+1)
    tau=(bin_edges[1:]+bin_edges[:-1])/2 # the bin centers 
    
    # build the 2D histogram
    density_array = build_2D_histogram(projected_tau, Nstamps, NBINS, bin_edges)
    
    # normalize the histogram
    density_array/=NP 
    
    # obtain the grid, in integration units
    X_tstamps,Y_tau=np.meshgrid(time_stamps,tau)  

    return X_tstamps,Y_tau,density_array 


def build_2D_histogram(projected_tau, Nstamps, NBINS, bins):
    density_array = np.zeros((Nstamps, NBINS))
    for i in range(Nstamps):
        counts, _ = np.histogram(projected_tau[i], bins=bins)
        density_array[i,:] = counts
    return density_array


def get_envelop_indexes(density_array, density_min):
    """
    Finds the limits of a stream based on a density array and a minimum density threshold.

    Parameters:
    - density_array (ndarray): A 2D array representing the density values.
    - density_min (float): The minimum density threshold.

    Returns:
    - index_from_left (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the left for each row of the density array.
    - index_from_right (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the right for each row of the density array.
    """
    
    Nstamps, _ =density_array.shape
    
    index_from_left, index_from_right = np.zeros(Nstamps), np.zeros(Nstamps)
    for i in range(Nstamps):
        array = density_array[i]
        # Find the first element that surpasses THRESHOLD when scanning from the left
        index_from_left[i] = np.argmax(array > density_min)
        # Find the first element that surpasses THRESHOLD when scanning from the right
        index_from_right[i] = len(array) - np.argmax(array[::-1] > density_min) - 1

    index_from_left = index_from_left.astype(int)
    index_from_right = index_from_right.astype(int)
    return index_from_left, index_from_right


def tau_envelopes(tau, index_from_left, index_from_right):
    """
    we want \tau(t), tau as a function of time...
    """
    tau_left = [tau[xx] for xx in index_from_left]
    tau_right = [tau[xx] for xx in index_from_right]
    return np.array(tau_left), np.array(tau_right)


def sig_clip(quantity, Nstdflag, Nstdclip, sides = 100, trial_max = 10000):
    std,mean    = np.std(quantity),np.mean(quantity)
    flag,clip   = mean+Nstdflag*std,mean+Nstdclip*std
    abs_diff = np.abs(quantity-mean)
    bad_dexes = np.where(abs_diff>flag)[0]
    sig_clipped=quantity.copy()
    cc = 0 
    conditions = cc < trial_max and len(bad_dexes)>0
    while conditions:
        # replace the bad dexes with the average of the two neighbors
        for bd in bad_dexes:
            uplim,lowlim=bd+sides,bd-sides
            # make sure the limits are within the array
            if uplim>len(sig_clipped):
                uplim=len(sig_clipped)
            if lowlim<0:
                lowlim=0
            sig_clipped[bd] = np.mean(sig_clipped[bd-sides:bd+sides-1])
            if sig_clipped[bd]<clip:
                bad_dexes = np.delete(bad_dexes,np.where(bad_dexes==bd))
        cc+=1
        
        if cc == trial_max:
            print('Max number of iterations reached')
        conditions = cc < trial_max and len(bad_dexes)>0
    return sig_clipped


def fourier_strongest_period(t,xt,yt,zt):
    r = np.sqrt(xt**2 + yt**2 + zt**2)
    # Compute the time step
    dt = np.mean(np.diff(t))

    # Compute the Fourier Transform
    R = np.fft.fft(r)

    # Compute the frequencies
    frequencies = np.fft.fftfreq(r.size, dt)

    # Select the half of the arrays that corresponds to the positive frequencies
    positive_frequencies = frequencies[:frequencies.size // 2]
    positive_R = R[:R.size // 2]

    # Compute the magnitude spectrum for the positive frequencies
    magnitude = np.abs(positive_R)

    maxfreq_dex=np.argmax(magnitude[1:]) + 1
    # Find the frequency that corresponds to the maximum value in the magnitude spectrum
    strongest_frequency = positive_frequencies[maxfreq_dex]


    period = 1/strongest_frequency
    return period 

