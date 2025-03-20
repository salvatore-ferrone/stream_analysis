import gcs
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import sys 
sys.path.append('../stream_analysis/')
import tailCoordinates as TC
import matplotlib as mpl
import multiprocessing as mp 
import os 



## GLOBAL PLOT VARIABLES 
xrange = [-20, 20]
yrange = [-1, 1]
vel_range = [-12, 12]
norm=mpl.colors.LogNorm(vmin=1,vmax=1e2)
SCAT1={"s":1,"alpha":1,"norm":norm,"cmap":"rainbow"}
AXISMAP={"xlim":xrange,"ylim":yrange,"ylabel":"y' [kpc]","xticks":[]}
AXISDEN={"xlim":xrange,"yscale":"log","ylim":[1e0,1e4],"ylabel":"N","xticks":[]}
AXISDISP={"xlim":xrange,"ylim":[0,8],"ylabel":r"$\sigma_v$ [km/s]","xticks":[]}
AXISMEAN={"xlim":xrange,"ylim":[1/10,100],"ylabel":r"$\langle v\rangle$ [km/s]","yscale":"log","xticks":[]}
AXISEIGEN={"xlim":xrange,"ylim":[0,8],"ylabel":r"$\sigma_v$ [km/s]","xlabel":"x' [kpc]",}
AXIS_HIST={"xlim":vel_range,"xlabel":"$\delta v$ [km/s]","ylabel":"N","yscale":"log","ylim":[1,2e2]}
AXIS_ELIPSE={"xlim":[-5,5],"ylim":[-5,5]}
### GLOBAL DATA VARIABLES 
GCname = "Pal5"
potential_env = "pouliasis2017pii-GCNBody"
internal_dynamics = "isotropic-plummer_mass_radius_grid"
montecarlokey="monte-carlo-009"
start = 9100
end = 9900
delta = end-start
skip = 100
NPs=np.arange(start,start+delta+skip+skip,skip)
NPs=np.concatenate((np.array([4500]),NPs))
cummulative_NPs = np.cumsum(NPs)
cummulative_NPs = np.insert(cummulative_NPs, 0, 0)
N_GRID = 5 
MASS_INDEX = 0
RADIUS_INDEX = 4


### FORR THE 2D MAP 
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


#### DATA IO
def grab_valid_fnames():
    fnames = []
    valid_NPs=[]
    FNAME = "{:s}-Stream-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS_INDEX).zfill(3), str(RADIUS_INDEX).zfill(3))
    for i in range(len(NPs)):
        path=gcs.path_handler._Stream(GCname=GCname,NP=NPs[i],potential_env=potential_env,internal_dynamics=internal_dynamics)
        fpath=path+FNAME
        if os.path.exists(fpath):
            fnames.append(fpath)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fpath)
    valid_NPs=np.array(valid_NPs)
    return fnames,valid_NPs

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

## COMPUTING THE VELOCITY DISPERSION TENSOR
def discretize_data(xT,xrange,num_bins):
    bin_edges = np.linspace(xrange[0], xrange[1], num_bins+1)  # Equal-width bins
    bin_indices = np.digitize(xT, bin_edges) - 1  # Assign stars to bins
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers,bin_indices

def speed_threshold(vx_bin, vy_bin, vz_bin, threshold):
    speed = np.sqrt(vx_bin ** 2 + vy_bin ** 2 + vz_bin ** 2)
    mask = np.abs(speed) < threshold
    return vx_bin[mask], vy_bin[mask], vz_bin[mask]


def compute_bin_statistics(vx_bin, vy_bin, vz_bin):
    N_bin = len(vx_bin)
    if N_bin > 1:  # Need at least 2 stars for variance
        v_bin = np.vstack((vx_bin, vy_bin, vz_bin))  # Shape (3, N_bin)
        v_mean = np.mean(v_bin, axis=1)  # Mean velocity per component

        # compute the standard error of the mean 
        v_std = np.std(v_bin, axis=1)
        v_sem = v_std/np.sqrt(N_bin)
        # Compute velocity dispersion tensor
        v_dev = v_bin - v_mean[:, None]  # Deviations from mean
        sigma_tensor = np.dot(v_dev, v_dev.T) / (N_bin - 1)  # Covariance matrix

        # Compute uncertainties in velocity dispersions
        sigma_unc = np.sqrt(2 * np.diag(sigma_tensor) ** 2 / (N_bin - 1))

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_tensor)  # Ensures sorted order
        eigenvalues = np.sqrt(eigenvalues)  # Convert to std deviations

        return v_mean, v_sem, sigma_tensor, sigma_unc, eigenvalues, eigenvectors
    else:
        # Return NaNs for insufficient data
        return (
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            np.full((3, 3), np.nan),
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            np.full((3, 3), np.nan),
        )


def compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, num_bins, threshold=np.max(np.abs(vel_range)),stats_func=compute_bin_statistics_bootstrap):
    # Discretize data
    bin_centers,bin_indices = discretize_data(xT, xrange, num_bins)

    # Storage for results
    mean_vel = []
    vel_sem = []
    disp_tensors = []
    disp_uncertainties = []
    eigenvectors_list = []
    eigenvalues_list = []

    # Loop over bins
    for b in range(num_bins):
        # filter by particles within the bin 
        mask = bin_indices == b
        vx_bin, vy_bin, vz_bin = vxT[mask], vyT[mask], vzT[mask]
        # reject outliers
        vx_bin, vy_bin, vz_bin = speed_threshold(vx_bin, vy_bin, vz_bin, threshold)  
  
        v_mean, v_sem, sigma_tensor, sigma_unc, eigenvalues, eigenvectors = \
            compute_bin_statistics(vx_bin, vy_bin, vz_bin)

        mean_vel.append(v_mean)
        vel_sem.append(v_sem)
        disp_tensors.append(sigma_tensor)
        disp_uncertainties.append(sigma_unc)
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    # Convert to arrays
    vel_sem = np.array(vel_sem)
    mean_vel = np.array(mean_vel)
    disp_tensors = np.array(disp_tensors)
    disp_uncertainties = np.array(disp_uncertainties)
    eigenvalues_list = np.array(eigenvalues_list)
    eigenvectors_list = np.array(eigenvectors_list)

    return bin_centers, mean_vel, vel_sem, disp_tensors, disp_uncertainties, eigenvalues_list, eigenvectors_list


def mass_radius_fnames(GCname,NPs,potential_env,internal_dynamics,montecarlokey,MASS_INDEX,RADIUS_INDEX):
    fnames = []
    valid_NPs=[]
    FNAME = "{:s}-Stream-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS_INDEX).zfill(3), str(RADIUS_INDEX).zfill(3))
    for i in range(len(NPs)):
        path=gcs.path_handler._Stream(GCname=GCname,NP=NPs[i],potential_env=potential_env,internal_dynamics=internal_dynamics)
        fpath=path+FNAME
        if os.path.exists(fpath):
            fnames.append(fpath)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fpath)
    valid_NPs=np.array(valid_NPs)
    return fnames,valid_NPs



def mass_radius_plots(montecarloindex=0):
    # data params
    GCname = "Pal5"
    potential_env = "pouliasis2017pii-GCNBody"
    internal_dynamics = "isotropic-plummer"
    montecarlokey="monte-carlo-"+str(montecarloindex).zfill(3)
    
    title = "{:s} {:s} {:s}".format(GCname,potential_env,montecarlokey)
    outname = "three_plot_{:s}_{:s}_{:s}_{:s}.png".format(GCname,potential_env,internal_dynamics,montecarlokey)
    if montecarloindex not in [0,9]:
        start=5305
        skip=1
        delta=18
    else:
        start = 1001
        end = 1097
        delta = end-start
        skip = 1
        # NotImplementedError("Montecarlo index not implemented")
    NPs = np.arange(start, start+delta+skip, skip)
    
    # get the orbit 
    fnameorbit=gcs.path_handler.GC_orbits(MWpotential=potential_env,GCname=GCname)
    t,x,y,z,vx,vy,vz=gcs.extractors.GCOrbits.extract_whole_orbit(fnameorbit,montecarlokey=montecarlokey)
    t,x,y,z,vx,vy,vz=TC.filter_orbit_by_dynamical_time(t,x,y,z,vx,vy,vz,time_of_interest=0,nDynTimes=3)    

    fnames,valid_NPs=mass_radius_fnames(GCname,NPs,potential_env,internal_dynamics,montecarlokey,MASS_INDEX,RADIUS_INDEX)
    phase_space, tesc = stack_phase_space(fnames,NPs)

    xT,yT,zT,vxT,vyT,vzT,_= TC.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0,:],phase_space[1,:],phase_space[2,:],
        phase_space[3,:],phase_space[4,:],phase_space[5,:],
        t,x,y,z,vx,vy,vz,t0=0)
    
    XX,YY,HH=short_cut(len(xT),xT,yT,xrange,yrange)
    XX,YY,H=discard_empty_bins(XX,YY,HH)
    XX,YY,H=order_by_density(XX,YY,H)

    # the histogram
    nbins=int(np.floor(np.sqrt(len(xT))))
    counts,xedges = np.histogram(xT,bins=nbins,range=xrange)
    xcenters=(xedges[:-1]+xedges[1:])/2
    bin_centers, mean_vel, disp_tensors, disp_uncertainties, eigenvalues_list, eigenvectors_list = compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, nbins)    

    norm=mpl.colors.LogNorm(vmin=1,vmax=1e2)
    SCAT1={"s":1,"alpha":1,"norm":norm,"cmap":"rainbow"}
    AXIS0={"xlim":xrange,"ylim":[-0.5,0.5],"ylabel":"y' [kpc]"}
    AXIS1={"xlim":xrange,"ylim":[1e0,1e4],"yscale":"log","ylabel":"N",}
    AXIS2={"xlim":xrange,"ylim":[0,10],"ylabel":r"$\sigma_v$ [km/s]","xlabel":"x' [kpc]",}    




    fig,axis=plt.subplots(3,1,figsize=(10,5),sharex=True)
    # axis[0].scatter(xT,yT,**SCAT1)
    axis[0].scatter(XX,YY,c=H,**SCAT1)
    axis[0].set(**AXIS0)

    axis[1].plot(bin_centers, counts)
    axis[1].set(**AXIS1)

    # Fill for sigma_x
    axis[2].fill_between(
        bin_centers,
        np.sqrt(disp_tensors[:, 0, 0]) - disp_uncertainties[:, 0],
        np.sqrt(disp_tensors[:, 0, 0]) + disp_uncertainties[:, 0],
        alpha=0.3, label=r'$\sigma_x$ uncertainty'
    )
    axis[2].plot(bin_centers, np.sqrt(disp_tensors[:, 0, 0]), label=r'$\sigma_x$', color='blue')
    # Fill for sigma_y
    axis[2].fill_between(
        bin_centers,
        np.sqrt(disp_tensors[:, 1, 1]) - disp_uncertainties[:, 1],
        np.sqrt(disp_tensors[:, 1, 1]) + disp_uncertainties[:, 1],
        alpha=0.3, 
    )
    axis[2].plot(bin_centers, np.sqrt(disp_tensors[:, 1, 1]), label=r'$\sigma_y$', color='orange')
    # Fill for sigma_z
    axis[2].fill_between(
        bin_centers,
        np.sqrt(disp_tensors[:, 2, 2]) - disp_uncertainties[:, 2],
        np.sqrt(disp_tensors[:, 2, 2]) + disp_uncertainties[:, 2],
        alpha=0.3, 
    )
    axis[2].plot(bin_centers, np.sqrt(disp_tensors[:, 2, 2]), label=r'$\sigma_z$', color='green')
    axis[2].set(**AXIS2)    
    axis[2].legend(frameon=False)

    axis[0].set_title(title)
    fig.tight_layout()
    fig.savefig(outname)

    plt.close(fig)
    return None

if __name__ == "__main__":
    
    # ncpu = mp.cpu_count()
    # with mp.Pool(ncpu) as pool:
        # pool.map(main,range(50))

    # main(0)
    # main(9)
    
