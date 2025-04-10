"""
A module for measure the velocity dispersion along the stream. Here is the typical usage:


    ```python
    import stream_analysis as sa

    # put the stream in tail coordinates 
    xT,yT,zT,vxT,vyT,vzT,_= sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0,:],phase_space[1,:],phase_space[2,:],
        phase_space[3,:],phase_space[4,:],phase_space[5,:],
        t,x,y,z,vx,vy,vz,t0=0) 
    
    # discretize the data
    num_bins=np.sqrt(len(xT)).astype(int)
    bin_centers,bin_indices = sa.velocity_dispersion.discretize_data(xT, xrange, num_bins)

    # compute the velocity dispersion tensor and uncertainties by boot strapping
    boot_strap_vel_mean, boot_strap_dispersion_tensor, boot_strap_eigenvalues, boot_strap_eigenvectors = sa.velocity_dispersion.boot_strapped_dispersion_along_stream(bin_indices, vxT, vyT, vzT, num_bins, threshold=velthreshold, num_bootstrap=num_bootstrap)

    # get the uncertainties in the sigma tensor
    dispersion_tensor = boot_strap_dispersion_tensor.mean(axis=1)
    dispersion_tensor_uncertainty = boot_strap_dispersion_tensor.std(axis=1)
    sigma_mean=np.array([np.sqrt(np.diag(dispersion_tensor[x])) for x in range(dispersion_tensor.shape[0])])
    sigma_err = np.array([np.sqrt(np.diag(dispersion_tensor_uncertainty[x])) for x in range(dispersion_tensor_uncertainty.shape[0])])    

    # get the uncertainties in the eigenvalues
    eigenvalues = np.power(boot_strap_eigenvalues.mean(axis=1),1/2)
    eigenvalues_uncertainty = np.power(boot_strap_eigenvalues.std(axis=1),1/2)    


    ```
"""
import numpy as np 

def discretize_data(xT,xrange,num_bins):
    bin_edges = np.linspace(xrange[0], xrange[1], num_bins+1)  # Equal-width bins
    bin_indices = np.digitize(xT, bin_edges) - 1  # Assign stars to bins
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers,bin_indices

def speed_threshold(vx_bin, vy_bin, vz_bin, threshold):
    speed = np.sqrt(vx_bin ** 2 + vy_bin ** 2 + vz_bin ** 2)
    mask = np.abs(speed) < threshold
    return vx_bin[mask], vy_bin[mask], vz_bin[mask]


def compute_disperion(v_bin):
    v_mean = np.mean(v_bin, axis=0)
    v_std = np.std(v_bin, axis=0)
    v_sem = v_std/np.sqrt(len(v_bin))
    v_dev = v_bin - v_mean
    v_variance_tensor = np.dot(v_dev.T, v_dev) / (len(v_bin) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(v_variance_tensor)
    return v_mean, v_sem, v_variance_tensor, eigenvalues, eigenvectors


def boot_strap_disperion(v_bin,num_bootstrap=100,debugIndexes=False):
    v_mean_out = np.zeros((num_bootstrap,3))
    v_variance_tensor_out = np.zeros((num_bootstrap,3,3))
    eigenvalues_out = np.zeros((num_bootstrap,3))
    eigenvectors_out = np.zeros((num_bootstrap,3,3))
    for i in range(num_bootstrap):
        indices = np.random.choice(len(v_bin), len(v_bin), replace=True)  # Resample with replacement
        v_bin_bootstrap = v_bin[indices]
        v_mean_temp, _, v_variance_tensor_temp, eigenvalues_temp, eigenvectors_temp = compute_disperion(v_bin_bootstrap)
        v_mean_out[i] = v_mean_temp
        v_variance_tensor_out[i] = v_variance_tensor_temp
        eigenvalues_out[i] = eigenvalues_temp
        eigenvectors_out[i] = eigenvectors_temp
    if debugIndexes:
        return v_mean_out, v_variance_tensor_out, eigenvalues_out, eigenvectors_out, indices
    else:
        return v_mean_out, v_variance_tensor_out, eigenvalues_out, eigenvectors_out


def boot_strapped_dispersion_along_stream(bin_indices, vxT, vyT, vzT, num_bins, threshold=20, num_bootstrap=100):
    """
    Compute velocity dispersion statistics for all bins using bootstrap resampling.

    import stream_analysis as sa

    # put the stream in tail coordinates 
    xT,yT,zT,vxT,vyT,vzT,_= sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0,:],phase_space[1,:],phase_space[2,:],
        phase_space[3,:],phase_space[4,:],phase_space[5,:],
        t,x,y,z,vx,vy,vz,t0=0) 
    
    # discretize the data
    num_bins=np.sqrt(len(xT)).astype(int)
    bin_centers,bin_indices = sa.velocity_dispersion.discretize_data(xT, xrange, num_bins)

    # compute the velocity dispersion tensor and uncertainties by boot strapping
    boot_strap_vel_mean, boot_strap_dispersion_tensor, boot_strap_eigenvalues, boot_strap_eigenvectors = boot_strapped_dispersion_along_stream(bin_indices, vxT, vyT, vzT, num_bins, threshold=velthreshold, num_bootstrap=num_bootstrap)

    # get the uncertainties in the sigma tensor
    dispersion_tensor = boot_strap_dispersion_tensor.mean(axis=1)
    dispersion_tensor_uncertainty = boot_strap_dispersion_tensor.std(axis=1)
    sigma_mean=np.array([np.sqrt(np.diag(dispersion_tensor[x])) for x in range(dispersion_tensor.shape[0])])
    sigma_err = np.array([np.sqrt(np.diag(dispersion_tensor_uncertainty[x])) for x in range(dispersion_tensor_uncertainty.shape[0])])    

    # get the uncertainties in the eigenvalues
    eigenvalues = np.power(boot_strap_eigenvalues.mean(axis=1),1/2)
    eigenvalues_uncertainty = np.power(boot_strap_eigenvalues.std(axis=1),1/2)        
    
    Parameters:
    -----------
    bin_indices : numpy.ndarray
        Array of bin indices for each particle
    vxT, vyT, vzT : numpy.ndarray
        Velocity components in tail coordinates for all particles
    num_bins : int
        Number of bins along the x-axis
    threshold : float, optional
        Maximum speed allowed for particles to be included in analysis (default: 20)
    num_bootstrap : int, optional
        Number of bootstrap samples to generate (default: 100)
    
    Returns:
    --------
    bin_centers : numpy.ndarray
        Centers of each bin
    vel_mean : list of numpy.ndarray
        Mean velocity vectors for each bin
    vel_sem : list of numpy.ndarray
        Standard error of the mean for velocity components in each bin
    dispersion_tensor : list of numpy.ndarray
        Velocity dispersion tensors for each bin
    boot_strap_vel_mean : list of numpy.ndarray
        Bootstrapped mean velocities for each bin
    boot_strap_dispersion_tensor : list of numpy.ndarray
        Bootstrapped dispersion tensors for each bin
    boot_strap_eigenvalues : list of numpy.ndarray
        Eigenvalues of bootstrapped dispersion tensors for each bin
    boot_strap_eigenvectors : list of numpy.ndarray
        Eigenvectors of bootstrapped dispersion tensors for each bin
    """
    boot_strap_vel_mean = []
    boot_strap_dispersion_tensor = []
    boot_strap_eigenvalues = []
    boot_strap_eigenvectors = []

    # Loop over bins
    for b in range(num_bins):
        mask = bin_indices == b
        vx_bin, vy_bin, vz_bin = vxT[mask], vyT[mask], vzT[mask]
        vx_bin, vy_bin, vz_bin = speed_threshold(vx_bin, vy_bin, vz_bin, threshold)

        num_in_bin = len(vx_bin)
        if num_in_bin < 3:
            # Skip bins with insufficient data
            v_means = np.full((num_bootstrap,3), np.nan)
            eigenvalues = np.full((num_bootstrap,3), np.nan)
            eigenvectors = np.full((num_bootstrap,3, 3), np.nan)
            v_variance_tensors = np.full((num_bootstrap, 3, 3), np.nan)
        else:
            v_bin = np.vstack((vx_bin, vy_bin, vz_bin)).T  # Shape (3, N_bin)
            v_means, v_variance_tensors, eigenvalues, eigenvectors = boot_strap_disperion(v_bin,num_bootstrap=num_bootstrap)


        boot_strap_vel_mean.append(v_means)
        boot_strap_dispersion_tensor.append(v_variance_tensors)
        boot_strap_eigenvalues.append(eigenvalues)
        boot_strap_eigenvectors.append(eigenvectors)

    # Convert to arrays
    boot_strap_vel_mean = np.array(boot_strap_vel_mean)
    boot_strap_dispersion_tensor = np.array(boot_strap_dispersion_tensor)
    boot_strap_eigenvalues = np.abs(np.array(boot_strap_eigenvalues)) # gaurd against negative eigenvalues
    boot_strap_eigenvectors = np.array(boot_strap_eigenvectors)

    return boot_strap_vel_mean, boot_strap_dispersion_tensor, boot_strap_eigenvalues, boot_strap_eigenvectors