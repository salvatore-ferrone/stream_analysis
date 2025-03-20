import gcs
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import sys 
sys.path.append('../stream_analysis/')
import tailCoordinates as TC
import matplotlib as mpl
import os 

def get_edges(NP,xlims,ylims):
    '''
    given the number of particles, return the number of bins
    '''
    ylen = ylims[1]-ylims[0]
    xlen = xlims[1]-xlims[0]    
    nBins=int(np.ceil(np.sqrt(NP)))
    nYbins=int(np.ceil(nBins*np.sqrt(ylen/xlen)))
    nXbins=int(np.ceil(nBins)*np.sqrt(xlen/ylen))
    xedges=np.linspace(xlims[0],xlims[1],nXbins)
    yedges=np.linspace(ylims[0],ylims[1],nYbins)
    return xedges,yedges

def get_2D_density(X,Y,xedges,yedges):
    H, xedges, yedges = np.histogram2d(X, Y, bins=(xedges, yedges))
    xcenters,ycenters = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    XX,YY = np.meshgrid(xcenters,ycenters)
    H = H.T
    return XX,YY,H

def discard_empty_bins(XX,YY,H):
    cond = H>0
    H = H[cond]
    XX = XX[cond]
    YY = YY[cond]
    return XX,YY,H

def short_cut(NP,X,Y,xlims,ylims):
    xedges,yedges = get_edges(NP,xlims,ylims)
    XX,YY,H = get_2D_density(X,Y,xedges,yedges)
    XX,YY,H = discard_empty_bins(XX,YY,H)
    return XX,YY,H


def order_by_density(XX,YY,H):
    order = np.argsort(H)
    return XX[order],YY[order],H[order]


def stack_phase_space(fnames,NPs):
    # set the indicies
    cummulative_NPs = np.cumsum(NPs)
    cummulative_NPs = np.insert(cummulative_NPs, 0, 0)
    # initiate the output arrays 
    phase_space = np.zeros((6,NPs.sum()))
    tesc=np.zeros(NPs.sum())
    for i in range(len(fnames)):
        with h5py.File(fnames[i],"r") as f:
            f['phase_space']
            phase_space[:,cummulative_NPs[i]:cummulative_NPs[i+1]] = f['phase_space'][:]
            tesc[cummulative_NPs[i]:cummulative_NPs[i+1]] = f['tesc'][:]
    return phase_space, tesc



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



def one_dimensional_velocity_dispersion_boot_strapped(bin_indices, vxT, vyT, vzT, num_bins, threshold=20, num_bootstrap=100):
    """
    Compute velocity dispersion statistics for all bins using bootstrap resampling.
    
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



def set_up_figure():
    fig = plt.figure(figsize=(6.3, 7.5))
    # Define height ratios: larger value for the first row, equal values for the rest
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1])  # First row taller
    # Create subplots
    axis_map = fig.add_subplot(gs[0, :])  # Second row, spans both columns
    axis_density = fig.add_subplot(gs[1, :])  # Third row
    axis_disp = fig.add_subplot(gs[2, :])  # Fourth row
    axis_mean = fig.add_subplot(gs[3, :])  # Fifth row
    axis_eign = fig.add_subplot(gs[4, :])  # Sixth row
    # Adjust spacing between rows
    fig.subplots_adjust(hspace=0.0)  # Remove vertical space between rows 1–5
    return fig, axis_map, axis_density, axis_disp, axis_mean, axis_eign

def grab_stream_fnames(GCname, montecarlokey, NPs, potential_env, internal_dynamics):
    fnames = []
    valid_NPs=[]
    for i in range(len(NPs)):
        fname=gcs.path_handler.Stream(GCname=GCname,montecarlokey=montecarlokey,NP=NPs[i],potential_env=potential_env,internal_dynamics=internal_dynamics)
        if os.path.exists(fname):
            fnames.append(fname)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fname)
    valid_NPs=np.array(valid_NPs)
    return fnames,valid_NPs





def add_eigenvalues(axis_eign, xcenters, eigenvalues, eigenvalues_uncertainty, AXISEIGEN):
    """Add eigenvalues and their uncertainties to the plot."""
    axis_eign.plot(xcenters, eigenvalues[:, 0], label=r"$\lambda_{\sigma_1}$", color='orangered')
    axis_eign.plot(xcenters, eigenvalues[:, 1], label=r"$\lambda_{\sigma_2}$", color='forestgreen')
    axis_eign.plot(xcenters, eigenvalues[:, 2], label=r"$\lambda_{\sigma_3}$", color='midnightblue')
    axis_eign.fill_between(xcenters, eigenvalues[:, 0] - eigenvalues_uncertainty[:, 0], eigenvalues[:, 0] + eigenvalues_uncertainty[:, 0], color='orangered', alpha=0.2)
    axis_eign.fill_between(xcenters, eigenvalues[:, 1] - eigenvalues_uncertainty[:, 1], eigenvalues[:, 1] + eigenvalues_uncertainty[:, 1], color='forestgreen', alpha=0.2)
    axis_eign.fill_between(xcenters, eigenvalues[:, 2] - eigenvalues_uncertainty[:, 2], eigenvalues[:, 2] + eigenvalues_uncertainty[:, 2], color='midnightblue', alpha=0.2)
    axis_eign.set(**AXISEIGEN)
    axis_eign.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, 1))

def add_velocity_dispersion(axis_disp, xcenters, sigma_mean, sigma_err, AXISDISP):
    """Add velocity dispersion and its uncertainties to the plot."""
    axis_disp.plot(xcenters, sigma_mean[:, 0], label=r"$\sigma_{v_x}$", color='r')
    axis_disp.plot(xcenters, sigma_mean[:, 1], label=r"$\sigma_{v_y}$", color='g')
    axis_disp.plot(xcenters, sigma_mean[:, 2], label=r"$\sigma_{v_z}$", color='b')
    axis_disp.fill_between(xcenters, sigma_mean[:, 0] - sigma_err[:, 0], sigma_mean[:, 0] + sigma_err[:, 0], color='r', alpha=0.2)
    axis_disp.fill_between(xcenters, sigma_mean[:, 1] - sigma_err[:, 1], sigma_mean[:, 1] + sigma_err[:, 1], color='g', alpha=0.2)
    axis_disp.fill_between(xcenters, sigma_mean[:, 2] - sigma_err[:, 2], sigma_mean[:, 2] + sigma_err[:, 2], color='b', alpha=0.2)
    axis_disp.set(**AXISDISP)
    legend = axis_disp.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, 1))


def add_mean_velocity(axis_mean, xcenters, vel_mean, vel_sem, AXISMEAN):
    """Add mean velocity and its uncertainties to the plot."""
    axis_mean.plot(xcenters, vel_mean[:, 0], label=r"$\langle v_x\rangle$", color='r')
    axis_mean.plot(xcenters, vel_mean[:, 1], label=r"$\langle v_y\rangle$", color='g')
    axis_mean.plot(xcenters, vel_mean[:, 2], label=r"$\langle v_z\rangle$", color='b')
    axis_mean.fill_between(xcenters, vel_mean[:, 0] - vel_sem[:, 0], vel_mean[:, 0] + vel_sem[:, 0], color='r', alpha=0.2)
    axis_mean.fill_between(xcenters, vel_mean[:, 1] - vel_sem[:, 1], vel_mean[:, 1] + vel_sem[:, 1], color='g', alpha=0.2)
    axis_mean.fill_between(xcenters, vel_mean[:, 2] - vel_sem[:, 2], vel_mean[:, 2] + vel_sem[:, 2], color='b', alpha=0.2)
    axis_mean.set(**AXISMEAN)



def add_density(axis_density, bin_centers, counts, AXISDEN):
    """Add density plot and vertical lines."""
    axis_density.plot(bin_centers, counts, drawstyle='steps-mid', color='k')
    axis_density.set(**AXISDEN)


def add_map(axis_map, XX, YY, H, SCAT1, AXISMAP):
    """Add scatter plot for the map."""
    axis_map.scatter(XX, YY, c=H, **SCAT1)
    axis_map.set(**AXISMAP)


def main(montecarloindex=0):
    GCname = "Pal5"
    potential_env = "pouliasis2017pii-GCNBody"
    internal_dynamics = "isotropic-plummer"
    
    montecarlokey="monte-carlo-"+str(montecarloindex).zfill(3)

    if montecarloindex in [0,9]:  
        start = 1000
        end = 1097
        delta = end-start
        skip = 1
        NPs=np.arange(start,start+delta+skip+skip,skip)
        NPs=np.concatenate((np.array([4500]),NPs))
    else:
        start=5305
        end=5323
        delta = end-start
        skip = 1
        NPs=np.arange(start,start+delta+skip+skip,skip)        


    # PLOT PARAMS 
    xrange = [-10, 10]
    yrange = [-1, 1]
    num_bootstrap=100
    velthreshold=12    
    norm=mpl.colors.LogNorm(vmin=1,vmax=1e2)
    vrange=np.array([-velthreshold,velthreshold])
    # set the properties 
    SCAT1={"s":1,"alpha":1,"norm":norm,"cmap":"rainbow"}
    AXISMAP={"xlim":xrange,"ylim":yrange,"ylabel":"y' [kpc]","xticks":[]}
    AXISDEN={"xlim":xrange,"yscale":"log","ylim":[1e0,1e4],"ylabel":"N","xticks":[],"yticks":[1e0,1e1,1e2,1e3]}
    AXISDISP={"xlim":xrange,"ylim":[0,8],"ylabel":r"$\sigma_v$ [km/s]","xticks":[],"yticks":[1,3,5,7]}
    AXISMEAN={"xlim":xrange,"ylim":vrange/2,"ylabel":r"$\langle v\rangle$ [km/s]","xticks":[],"yticks":[-4,-2,0,2,4]}
    AXISEIGEN={"xlim":xrange,"ylim":[0,8],"ylabel":r"$\lambda_v$ [$\sigma_{ij}$ km/s]","xlabel":"x' [kpc]","yticks":[1,3,5,7]}


    # load the stream data
    fnames,valid_NPs=grab_stream_fnames(GCname, montecarlokey, NPs, potential_env, internal_dynamics)
    phase_space,tesc=stack_phase_space(fnames,valid_NPs)
    # with h5py.File(fnames[0],'r') as myfile:
        # RC,MASS=myfile.attrs['HALF_MASS_RADIUS'],myfile.attrs['MASS']

    # load the GC orbit
    fnameorbit=gcs.path_handler.GC_orbits(MWpotential=potential_env,GCname=GCname)
    t,x,y,z,vx,vy,vz=gcs.extractors.GCOrbits.extract_whole_orbit(fnameorbit,montecarlokey=montecarlokey)
    t,x,y,z,vx,vy,vz=TC.filter_orbit_by_dynamical_time(t,x,y,z,vx,vy,vz,time_of_interest=0,nDynTimes=3)        

    # put the stream in tail coordinates 
    xT,yT,zT,vxT,vyT,vzT,_= TC.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0,:],phase_space[1,:],phase_space[2,:],
        phase_space[3,:],phase_space[4,:],phase_space[5,:],
        t,x,y,z,vx,vy,vz,t0=0)    
    
    # discretize the data
    num_bins=np.sqrt(len(xT)).astype(int)
    bin_centers,bin_indices = discretize_data(xT, xrange, num_bins)
    
    # BOOT STRAP !
    boot_strap_vel_mean, boot_strap_dispersion_tensor, boot_strap_eigenvalues, boot_strap_eigenvectors = one_dimensional_velocity_dispersion_boot_strapped(bin_indices, vxT, vyT, vzT, num_bins, threshold=velthreshold, num_bootstrap=num_bootstrap)

    # compute the statistics for the mean
    vel_mean = boot_strap_vel_mean.mean(axis=1)
    vel_sem = boot_strap_vel_mean.std(axis=1)    

    # get the uncertainties in the sigma tensor
    dispersion_tensor = boot_strap_dispersion_tensor.mean(axis=1)
    dispersion_tensor_uncertainty = boot_strap_dispersion_tensor.std(axis=1)
    sigma_mean=np.array([np.sqrt(np.diag(dispersion_tensor[x])) for x in range(dispersion_tensor.shape[0])])
    sigma_err = np.array([np.sqrt(np.diag(dispersion_tensor_uncertainty[x])) for x in range(dispersion_tensor_uncertainty.shape[0])])    

    # get the uncertainties in the eigenvalues
    eigenvalues = np.power(boot_strap_eigenvalues.mean(axis=1),1/2)
    eigenvalues_uncertainty = np.power(boot_strap_eigenvalues.std(axis=1),1/2)    

    # the 2D map 
    XX,YY,HH=short_cut(len(xT),xT,yT,xrange,yrange)
    XX,YY,HH=discard_empty_bins(XX,YY,HH)
    XX,YY,HH=order_by_density(XX,YY,HH)
    # the 1D density 

    counts,xedges = np.histogram(xT,bins=num_bins,range=xrange)
    xcenters=(xedges[:-1]+xedges[1:])/2


    plot_title = "{:s} {:s} ".format(GCname, montecarlokey)
    figure_name = "./frames/{:s}-{:s}-rerun.png".format(GCname, montecarlokey)
    print("plotting",figure_name)
    # START THE PLOT 
    fig, axis_map, axis_density, axis_disp, axis_mean, axis_eign = set_up_figure()
    add_map(axis_map, XX, YY, HH, SCAT1, AXISMAP)
    add_density(axis_density, bin_centers, counts, AXISDEN)
    add_mean_velocity(axis_mean, xcenters, vel_mean, vel_sem, AXISMEAN)
    add_velocity_dispersion(axis_disp, xcenters, sigma_mean, sigma_err, AXISDISP)
    add_eigenvalues(axis_eign, xcenters, eigenvalues, eigenvalues_uncertainty, AXISEIGEN)
    AXISMAP['title']=plot_title
    axis_map.set(**AXISMAP)

    fig.savefig(figure_name,dpi=300)
    print("done with ", figure_name)


    plt.close(fig)


if __name__ == "__main__":
    import multiprocessing as mp 
    ncpu = mp.cpu_count()
    with mp.Pool(ncpu) as pool:
        pool.map(main,range(50))

# END OF FILE
 