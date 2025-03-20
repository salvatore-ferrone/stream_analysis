#%%
import gcs
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import sys 
sys.path.append('../stream_analysis/')
import tailCoordinates as TC
import matplotlib as mpl
import os 
from matplotlib.patches import Ellipse



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
    

def compute_bin_statistics_bootstrap(vx_bin, vy_bin, vz_bin, num_bootstrap=100):
    N_bin = len(vx_bin)

    if N_bin > 1:  # Need at least 2 stars for variance
        v_bin = np.vstack((vx_bin, vy_bin, vz_bin))  # Shape (3, N_bin)
        v_mean = np.mean(v_bin, axis=1)  # Mean velocity per component
        v_dev = v_bin - v_mean[:, None]  # Deviations from mean
        sigma_tensor = np.dot(v_dev, v_dev.T) / (N_bin - 1)  # Covariance matrix

        # Bootstrapping to estimate uncertainties
        bootstrap_diagonals = []
        bootstrap_sigma_tensors=[]
        v_mean_bootstrap = []
        for _ in range(num_bootstrap):
            indices = np.random.choice(N_bin, N_bin, replace=True)  # Resample with replacement
            v_bin_bootstrap = v_bin[:, indices]
            v_mean = np.mean(v_bin_bootstrap, axis=1)
            v_dev_bootstrap = v_bin_bootstrap - v_mean[:,None]
            sigma_tensor_bootstrap = np.dot(v_dev_bootstrap, v_dev_bootstrap.T) / (N_bin - 1)
            bootstrap_sigma_tensors.append(sigma_tensor_bootstrap)
            bootstrap_diagonals.append(np.diag(sigma_tensor_bootstrap))
            v_mean_bootstrap.append(np.mean(v_bin_bootstrap, axis=1))
        
        v_mean_bootstrap=np.array(v_mean_bootstrap)
        bootstrap_diagonals = np.array(bootstrap_diagonals)
        bootstrap_sigma_tensors = np.array(bootstrap_sigma_tensors)

        # report the uncertainties using the book strapping
        sigma_unc = np.std(bootstrap_diagonals, axis=0) # standard deviation of the diagonal
        v_sem = np.std(v_mean_bootstrap, axis=0) # standard error of the mean

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_tensor)  # Ensures sorted order
        eigenvalues = np.sqrt(eigenvalues)  # Convert to std deviations

        return v_mean, v_sem, sigma_tensor, sigma_unc, eigenvalues, eigenvectors,bootstrap_sigma_tensors
    else:
        # Return NaNs for insufficient data
        return (
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            np.full((3, 3), np.nan),
            [np.nan,np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            np.full((3, 3), np.nan),
            np.full((num_bootstrap,3, 3), np.nan)
        )



def discretize_data(xT,xrange,num_bins):
    bin_edges = np.linspace(xrange[0], xrange[1], num_bins+1)  # Equal-width bins
    bin_indices = np.digitize(xT, bin_edges) - 1  # Assign stars to bins
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers,bin_indices


def speed_threshold(vx_bin, vy_bin, vz_bin, threshold):
    speed = np.sqrt(vx_bin ** 2 + vy_bin ** 2 + vz_bin ** 2)
    mask = np.abs(speed) < threshold
    return vx_bin[mask], vy_bin[mask], vz_bin[mask]

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


        # Compute statistics for the bin
        if stats_func == compute_bin_statistics_bootstrap:
            v_mean, v_sem, sigma_tensor, sigma_unc, eigenvalues, eigenvectors,_ = stats_func(
                vx_bin, vy_bin, vz_bin
            )
        else:

            v_mean, v_sem, sigma_tensor, sigma_unc, eigenvalues, eigenvectors = stats_func(
                vx_bin, vy_bin, vz_bin
            )

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

def set_up_figure():
    fig = plt.figure(figsize=(6.3, 8.5))
    
    # Define height ratios: larger value for the first row, equal values for the rest
    gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 1, 1, 1, 1, 1])  # First row taller
    
    # Create subplots
    axis_elipse= fig.add_subplot(gs[0, 0])
    axis_hist = fig.add_subplot(gs[0, 1])  # First row, second column
    axis_map = fig.add_subplot(gs[1, :])  # Second row, spans both columns
    axis_density = fig.add_subplot(gs[2, :])  # Third row
    axis_disp = fig.add_subplot(gs[3, :])  # Fourth row
    axis_mean = fig.add_subplot(gs[4, :])  # Fifth row
    axis_eign = fig.add_subplot(gs[5, :])  # Sixth row
    
    # Adjust spacing between rows
    fig.subplots_adjust(hspace=0.0)  # Remove vertical space between rows 1–5

    pos = axis_hist.get_position()
    axis_hist.set_position([pos.x0, pos.y0+0.05, pos.width, pos.height])

    pos = axis_elipse.get_position()
    axis_elipse.set_position([pos.x0, pos.y0+0.05, pos.width, pos.height])
    
    return fig, axis_hist, axis_elipse, axis_map, axis_density, axis_disp, axis_mean, axis_eign

def fill_plot(handles,bin_centers,mean_vel,vel_sem,disp_tensors,disp_uncertainties,eigenvalues_list):
    fig, axis_hist, axis_elipse, axis_map, axis_density, axis_disp, axis_mean, axis_eign=handles

    # MAP
    axis_map.scatter(XX,YY,c=H,**SCAT1)
    axis_map.set(**AXISMAP)
    # DENSITY profile 
    axis_density.plot(xcenters, counts)
    axis_density.set(**AXISDEN)

    # DISPERSION
    axis_disp.plot(xcenters, np.sqrt(disp_tensors[:, 0, 0]), label=r'$\sigma_x$', color='blue')
    axis_disp.plot(xcenters, np.sqrt(disp_tensors[:, 1, 1]), label=r'$\sigma_y$', color='orange')
    axis_disp.plot(xcenters, np.sqrt(disp_tensors[:, 2, 2]), label=r'$\sigma_z$', color='green')
    axis_disp.fill_between(
        xcenters,
        np.sqrt(disp_tensors[:, 0, 0]) - disp_uncertainties[:, 0],
        np.sqrt(disp_tensors[:, 0, 0]) + disp_uncertainties[:, 0],
        alpha=0.3, label=r'$\sigma_x$ uncertainty'
    )
    axis_disp.fill_between( 
        xcenters,
        np.sqrt(disp_tensors[:, 1, 1]) - disp_uncertainties[:, 1],
        np.sqrt(disp_tensors[:, 1, 1]) + disp_uncertainties[:, 1],
        alpha=0.3, 
    )
    axis_disp.fill_between(
        xcenters,
        np.sqrt(disp_tensors[:, 2, 2]) - disp_uncertainties[:, 2],
        np.sqrt(disp_tensors[:, 2, 2]) + disp_uncertainties[:, 2],
        alpha=0.3, 
    )
    axis_disp.set(**AXISDISP)

    # MEAN VELOCITY
    axis_mean.plot(bin_centers, np.abs(mean_vel[:, 0]), color='blue')
    axis_mean.plot(bin_centers, np.abs(mean_vel[:, 1]), color='orange')
    axis_mean.plot(bin_centers, np.abs(mean_vel[:, 2]), color='green')
    axis_mean.fill_between(
        bin_centers,
        np.abs(mean_vel[:, 0]) - vel_sem[:,0],
        np.abs(mean_vel[:, 0]) + vel_sem[:,0],
        alpha=0.3, 
    )
    axis_mean.fill_between(
        bin_centers,
        np.abs(mean_vel[:, 1]) - vel_sem[:,1],
        np.abs(mean_vel[:, 1]) + vel_sem[:,1],
        alpha=0.3, 
    )
    axis_mean.fill_between(
        bin_centers,
        np.abs(mean_vel[:, 2]) - vel_sem[:,2],
        np.abs(mean_vel[:, 2]) + vel_sem[:,2],
        alpha=0.3, 
    )


    axis_mean.set(**AXISMEAN)

    axis_eign.plot(bin_centers, eigenvalues_list[:, 0], label=r'$\lambda_1$', color='blue')
    axis_eign.plot(bin_centers, eigenvalues_list[:, 1], label=r'$\lambda_2$', color='orange')
    axis_eign.plot(bin_centers, eigenvalues_list[:, 2], label=r'$\lambda_3$', color='green')
    axis_eign.legend(frameon=False)
    axis_eign.set(**AXISEIGEN)
    





fnames,valid_NPs=grab_valid_fnames()
phase_space,tesc=stack_phase_space(fnames,valid_NPs)
with h5py.File(fnames[0],'r') as myfile:
    RC,MASS=myfile.attrs['HALF_MASS_RADIUS'],myfile.attrs['MASS']

# load and filter orbit 
fnameorbit=gcs.path_handler.GC_orbits(MWpotential=potential_env,GCname=GCname)
t,x,y,z,vx,vy,vz=gcs.extractors.GCOrbits.extract_whole_orbit(fnameorbit,montecarlokey=montecarlokey)
t,x,y,z,vx,vy,vz=TC.filter_orbit_by_dynamical_time(t,x,y,z,vx,vy,vz,time_of_interest=0,nDynTimes=3)

# put in tail coordiantes
xT,yT,zT,vxT,vyT,vzT,_= TC.transform_from_galactico_centric_to_tail_coordinates(
    phase_space[0,:],phase_space[1,:],phase_space[2,:],
    phase_space[3,:],phase_space[4,:],phase_space[5,:],
    t,x,y,z,vx,vy,vz,t0=0)


# make the 2D density plot
XX,YY,HH=short_cut(len(xT),xT,yT,xrange,yrange)
XX,YY,H=discard_empty_bins(XX,YY,HH)
XX,YY,H=order_by_density(XX,YY,H)

# make the 1D histogram 
nbins=int(np.floor(np.sqrt(len(xT))))
counts,xedges = np.histogram(xT,bins=nbins,range=xrange)
xcenters=(xedges[:-1]+xedges[1:])/2
# bin_centers, mean_vel, vel_sem, disp_tensors, disp_uncertainties_normal, eigenvalues_list, eigenvectors_list= compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, nbins, stats_func=compute_bin_statistics)
bin_centers, mean_vel, vel_sem,disp_tensors, disp_uncertainties_bootstrap, eigenvalues_list, eigenvectors_list = compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, nbins, stats_func=compute_bin_statistics_bootstrap)



# # pick a method for the uncertainties 
disp_uncertainties=disp_uncertainties_bootstrap

fig, axis_hist, axis_elipse, axis_map, axis_density, axis_disp, axis_mean, axis_eign=set_up_figure()
handles=fig, axis_hist, axis_elipse, axis_map, axis_density, axis_disp, axis_mean, axis_eign
#%% 
fill_plot(handles,bin_centers,mean_vel,vel_sem,disp_tensors,disp_uncertainties,eigenvalues_list)
## SCANNING HISTOGRAM

threshold=np.max(np.abs(vel_range))
bin_centers,bin_indices = discretize_data(xT, xrange, nbins)
# for b in range(0,54):
for b in range(nbins):
    axis_hist.clear()
    axis_elipse.clear()
    mask = bin_indices == b
    vx_bin, vy_bin, vz_bin = vxT[mask], vyT[mask], vzT[mask]
    vx_bin, vy_bin, vz_bin = speed_threshold(vx_bin, vy_bin, vz_bin, threshold)
    vel_bin = np.stack((vx_bin, vy_bin, vz_bin), axis=1)
    vel_deviations=vel_bin - np.mean(vel_bin,axis=0)
    nbins_vel=int(3*np.ceil(np.sqrt(len(vx_bin))))
    if nbins_vel > 1:
        axis_hist.hist(vel_deviations[:,0],bins=nbins_vel,range=vel_range,histtype='step',label=r"$\delta v_x$")
        axis_hist.hist(vel_deviations[:,1],bins=nbins_vel,range=vel_range,histtype='step',label=r"$\delta v_y$")
        axis_hist.hist(vel_deviations[:,2],bins=nbins_vel,range=vel_range,histtype='step',label=r"$\delta v_z$")
        axis_hist.legend(frameon=False)
    # add the scan
    line_map=axis_map.vlines(bin_centers[b],yrange[0],yrange[1],color='black',linestyle='--')
    line_den=axis_density.vlines(bin_centers[b],AXISDEN['ylim'][0],AXISDEN['ylim'][1],color='black',linestyle='--')
    line_disp=axis_disp.vlines(bin_centers[b],AXISDISP['ylim'][0],AXISDISP['ylim'][1],color='black',linestyle='--')
    line_eigen=axis_eign.vlines(bin_centers[b],AXISEIGEN['ylim'][0],AXISEIGEN['ylim'][1],color='black',linestyle='--')


    # add the ellipsoide
    eigvecs=eigenvectors_list[b]
    eigvals=eigenvalues_list[b]
    # Take the two largest eigenvalues for projection
    major_axis = eigvecs[:, 2] * eigvals[2]  # Largest eigenvalue
    minor_axis = eigvecs[:, 1] * eigvals[1]  # Second largest eigenvalue
    
    ellipse = Ellipse(
    xy=(0, 0),  # Centered along x-axis
    width= eigvals[2],  # Largest axis (scaled for visibility)
    height=  eigvals[1],  # Second largest axis
    angle=np.degrees(np.arctan2(major_axis[1], major_axis[0])),
    edgecolor='black',
    facecolor='none'
    )
    axis_elipse.add_patch(ellipse)

    axis_elipse.plot([AXIS_ELIPSE['xlim'][0],AXIS_ELIPSE['xlim'][1]],[0,0],color='black',linestyle='-',linewidth=0.5)
    axis_elipse.plot([0,0],[AXIS_ELIPSE['ylim'][0],AXIS_ELIPSE['ylim'][1]],color='black',linestyle='-',linewidth=0.5)

    axis_hist.set(**AXIS_HIST)
    axis_elipse.set(**AXIS_ELIPSE)
    axis_elipse.set_aspect('equal')
    axis_elipse.set_axis_off()
    figname="{:s}-{:s}-mass-{:s}-radius-{:s}-dispersionScanFrame-{:s}.png".format(GCname, montecarlokey, str(MASS_INDEX).zfill(3), str(RADIUS_INDEX).zfill(3),str(b).zfill(3))
    # fig.tight_layout()
    print("saving",figname)
    fig.savefig("./frames/"+figname,dpi=300)
    line_map.remove()
    line_den.remove()
    line_disp.remove()
    line_eigen.remove()
