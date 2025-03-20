import gcs
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import sys 
sys.path.append('../stream_analysis/')
import tailCoordinates as TC
import matplotlib as mpl
import os 
import multiprocessing as mp 


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

xrange = [-20, 20]
yrange = [-1, 1]



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
        # only do the boot strapping for variane of the variance and eigen 
        for _ in range(num_bootstrap):
            indices = np.random.choice(N_bin, N_bin, replace=True)  # Resample with replacement
            v_bin_sampling = v_bin[:, indices]
            v_mean_sampling = np.mean(v_bin_sampling, axis=1)
            v_dev_sampling = v_bin_sampling - v_mean_sampling[:,None]
            sigma_tensor_bootstrap = np.dot(v_dev_sampling, v_dev_sampling.T) / (N_bin - 1)
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

def load_fnames(MASS_INDEX,RADIUS_INDEX):
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

def doplot(XX,YY,H,bin_centers,counts,disp_tensors,disp_uncertainties,properties):
    AXIS0,AXIS1,AXIS2,SCAT1=properties
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
    return fig,axis

def main(MASS_INDEX,RADIUS_INDEX):
    fnames,valid_NPs=load_fnames(MASS_INDEX,RADIUS_INDEX)
    phase_space,tesc=stack_phase_space(fnames,valid_NPs)
    with h5py.File(fnames[0],'r') as myfile:
        RC,MASS=myfile.attrs['HALF_MASS_RADIUS'],myfile.attrs['MASS']

    # load orbit 
    fnameorbit=gcs.path_handler.GC_orbits(MWpotential=potential_env,GCname=GCname)
    t,x,y,z,vx,vy,vz=gcs.extractors.GCOrbits.extract_whole_orbit(fnameorbit,montecarlokey=montecarlokey)
    t,x,y,z,vx,vy,vz=TC.filter_orbit_by_dynamical_time(t,x,y,z,vx,vy,vz,time_of_interest=0,nDynTimes=3)    
    # get the tail coordinates
    xT,yT,zT,vxT,vyT,vzT,_= TC.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0,:],phase_space[1,:],phase_space[2,:],
        phase_space[3,:],phase_space[4,:],phase_space[5,:],
        t,x,y,z,vx,vy,vz,t0=0)    

    num_bins = len(xT)
    bin_centers, mean_vel, disp_tensors, disp_uncertainties, eigenvalues_list, eigenvectors_list = compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, num_bins)

    # get the 2D distribution of the particles
    XX,YY,HH=short_cut(len(xT),xT,yT,xrange,yrange)
    XX,YY,H=discard_empty_bins(XX,YY,HH)
    XX,YY,H=order_by_density(XX,YY,H)

    # the 1D histogram
    nbins=int(np.floor(np.sqrt(len(xT)))/2)
    counts,xedges = np.histogram(xT,bins=nbins,range=xrange)
    xcenters=(xedges[:-1]+xedges[1:])/2
    bin_centers, mean_vel, disp_tensors, disp_uncertainties, eigenvalues_list, eigenvectors_list = compute_velocity_dispersion_tensor(xT, vxT, vyT, vzT, xrange, nbins)    



    norm=mpl.colors.LogNorm(vmin=1,vmax=1e2)
    SCAT1={"s":1,"alpha":1,"norm":norm,"cmap":"rainbow"}
    AXIS0={"xlim":xrange,"ylim":yrange,"ylabel":"y' [kpc]"}
    AXIS1={"xlim":xrange,"yscale":"log","ylim":[1e0,1e4],"ylabel":"N",}
    AXIS2={"xlim":xrange,"ylim":[0,10],"ylabel":r"$\sigma_v$ [km/s]","xlabel":"x' [kpc]",}

    properties=AXIS0,AXIS1,AXIS2,SCAT1

    fig,axis=doplot(XX,YY,H,bin_centers,counts,disp_tensors,disp_uncertainties,properties)
    figtitle="{:s} {:s} {:s} msun {:s} pc".format(GCname, montecarlokey, str(int(MASS)).zfill(3), str(int(1000*RC)).zfill(3))
    outfname="{:s}-{:s}-{:s}-mass-{:s}-radius-{:s}".format(GCname, montecarlokey, internal_dynamics, str(int(MASS_INDEX)).zfill(3), str(int(1000*RADIUS_INDEX)).zfill(3))
    fig.suptitle(figtitle)
    fig.tight_layout()
    fig.savefig(outfname+".png",dpi=300)
    plt.close(fig)


    

if __name__ == "__main__":
    MASS_INDEX = 0
    RADIUS_INDEX = 0
    ncpu=mp.cpu_count()
    pool=mp.Pool(ncpu)
    N_GRID = 5 
    for MASS_INDEX in range(N_GRID):
        for RADIUS_INDEX in range(N_GRID):
            pool.apply_async(main,args=(MASS_INDEX,RADIUS_INDEX))
    pool.close()
    pool.join()
    #




