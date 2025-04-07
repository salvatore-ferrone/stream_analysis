"""
This script extracts the stream and makes the 1D density maps in time (which is a faster computation)
It projects it on to the orbit. 
It follows the evolution of the stream density in time.
It uses multiprocessing to speed up the computation.
It has to open many different files since I have them spaced apart in different data files.
"""

import gcs 
import numpy as np
import h5py
import stream_analysis as sa 
import os 
import multiprocessing as mp
import datetime
import sys

# MASS GRID AND RADIUS GRID
SIZE_GRID = 5
N_MASS_SAMPLING = SIZE_GRID
N_RADIUS_SAMPLING = SIZE_GRID # square grid
MASS_GRID = np.logspace(4,4.8, N_MASS_SAMPLING) # in Msun
RADIUS_GRID = np.logspace(np.log10(2),np.log10(30),N_RADIUS_SAMPLING)/1000 # in kpc


ATTRS = {
    "Author": "Salvatore Ferrone",
    "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
    "email":"salvatore.ferrone.1996@gmail.com",
    "Institution": "l'Università di Roma, La Sapienza",
    "Description": "a series of 1D density maps of the stream projected onto the orbit for each snapshot",
}


def taudensity_fname(GCname,NP,MWpotential,internal_dynamics,montecarlokey,MASS,RADIUS):
    fname = "{:s}-tauDensity-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS).zfill(3), str(RADIUS).zfill(3))
    pathname=gcs.path_handler._tauDensityMaps(GCname=GCname,NP=NP, MWpotential=MWpotential, internal_dynamics=internal_dynamics)
    if not os.path.exists(pathname):
        os.makedirs(pathname, exist_ok=True)
    return pathname+fname


def get_streamSnapShotsFileName(GCname, NP, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype='physical'):
    """
    Get the filename for the StreamSnapShots file based on the GC name, Monte Carlo key, mass index, and radius index.
    
    Parameters:
    ----------
    GCname : str
        Name of the globular cluster.
    montecarlokey : str
        Key for the Monte Carlo simulation.
    MASS_INDEX : int
        Index for the mass.
    RADIUS : int
        Index for the radius.
    Returns:
    -------
    str
        The formatted filename.
    """ 
    valid_types = ['index', "physical"]
    if mytype not in valid_types:
        raise ValueError("type must be one of the following: {:s}".format(str(valid_types)))
    if mytype == 'index':
        assert isinstance(MASS, int), "MASS must be an integer but was {:}".format(type(MASS))
        assert isinstance(RADIUS, int), "RADIUS must be an integer but was {:}".format(type(RADIUS))
        path=gcs.path_handler._StreamSnapShots(GCname=GCname,NP=NP,potential_env=potential_env,internal_dynamics=internal_dynamics)
        fname = "{:s}-StreamSnapShots-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS).zfill(3), str(RADIUS).zfill(3))
        fname = path + fname
    elif mytype == 'physical':
        assert isinstance(MASS, int), "MASS must be a int IN SOLAR MASSES but was {:}".format(type(MASS))
        assert isinstance(RADIUS, int), "RADIUS must be a int in PARSECS but was  {:}".format(type(RADIUS))
        fname = gcs.path_handler.StreamSnapShotsMassRadius(GCname, NP, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS)
    return fname


def grab_valid_fnames(GCname, NPs, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype='physical'):

    """
    iterate over the NPs and check if the file exists
    Parameters
    ----------
    GCname : str
        Name of the globular cluster.
    NPs : list
        List of number of particles.
    potential_env : str
        Name of the potential environment.          
    internal_dynamics : str
        Name of the internal dynamics.
    montecarlokey : str
        Key for the Monte Carlo simulation.
    MASS : int
        Mass of the globular cluster in solar masses.
    RADIUS : int
        Half Mass Radius of the cluster in parsecs.
    mytype : str
        Type of the file name, either 'index' or 'physical'.
    Returns
    -------
    fnames : list
        List of valid file names.
    valid_NPs : list
        List of valid number of particles corresponding to the file names.
    """

    fnames = []
    valid_NPs=[]
    
    for i in range(len(NPs)):
        # get the file name
        fpath=get_streamSnapShotsFileName(GCname, NPs[i], potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype=mytype)
        # check if the file exists
        if os.path.exists(fpath):
            fnames.append(fpath)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fpath)
    valid_NPs=np.array(valid_NPs)
    return fnames, valid_NPs



def stack_phase_space(fnames,NPs,time_of_interest=0):
    """ assumes all fnames are valid files of the same format 
    """
    # set the indicies
    cummulative_NPs = np.cumsum(NPs)
    cummulative_NPs = np.insert(cummulative_NPs, 0, 0)
    # initiate the output arrays 
    phase_space = np.zeros((6,NPs.sum()))
    tesc=np.zeros(NPs.sum())
    snapshottime=np.zeros(len(fnames))
    for i in range(len(fnames)):
        with h5py.File(fnames[i],"r") as f:
            target_index = np.argmin(np.abs(f['time_stamps'][:]-time_of_interest))
            phase_space[:,cummulative_NPs[i]:cummulative_NPs[i+1]] = f["StreamSnapShots"][str(target_index)][:]
            tesc[cummulative_NPs[i]:cummulative_NPs[i+1]] = f['tesc'][:]
            snapshottime[i]=f['time_stamps'][target_index]
    return phase_space, tesc, snapshottime



def median_dynamical_time(hostorbit):
    # unpack the host orbit
    _,xHost,yHost,zHost,vxHost,vyHost,vzHost = hostorbit
    # calculate the dynamical time
    rH          =   np.sqrt(xHost**2+yHost**2+zHost**2)
    vH          =   np.sqrt(vxHost**2+vyHost**2+vzHost**2)
    tdynamical  =   np.median(rH/vH)
    return tdynamical

def initialize_tau_edges(NP,tdynamical,ndyn):
    """
    initialize the tau edges for the histogram
    Parameters
    ----------
    NP : int
        Number of particles.
    tdynamical: float
        The median dynamical time.
    ndyn : float
        Number of dynamical times.
    Returns
    -------
    tau_edges : numpy.ndarray
        The edges of the histogram bins.
    """
    # create the bin edges 
    tau_edges   =   np.linspace(-ndyn*tdynamical,ndyn*tdynamical,int(np.ceil(np.sqrt(NP))))
    return tau_edges

def initialize_2D_counts(tau_edges,n_time_stamps):
    """
    initialize the output arrays for the histogram
    Parameters
    ----------
    NP : int
        Number of particles.
    Returns
    -------
    tau_centers : numpy.ndarray
        The histogram of the tau values.
    counts : numpy.ndarray
        the counts in each 1D profile 
    """
    # create the bin edges 
    tau_centers = 0.5*(tau_edges[:-1]+tau_edges[1:])
    # create the bin centers
    counts = np.zeros((n_time_stamps,len(tau_centers)))
    return tau_centers, counts


def loop_extract_density_profile(ii, time_stamps, fnames, NPs, hostorbit, ndyn, tau_edges):
        # unpack the host orbit
        tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost = hostorbit
        # grab the current time 
        time_of_interest = time_stamps[ii]
        # extract the phase space
        phase_space, tesc, _ = stack_phase_space(fnames,NPs,time_of_interest=time_of_interest)
        # reduce to only the escaped stars 
        my_phase_space=phase_space[:,tesc > -990]
        # shorten the orbit 
        TORB, XORB, YORB, ZORB, _, _, _=sa.tailCoordinates.filter_orbit_by_dynamical_time(\
                tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost,time_of_interest,ndyn)
        # project the particles onto the orbit 
        indexes = sa.tailCoordinates.get_closests_orbital_index(XORB, YORB, ZORB,
                                my_phase_space[0],my_phase_space[1],my_phase_space[2])
        # calculate the tau
        tau = TORB[indexes] - time_of_interest
        # make the histogram
        tau_hist, _ = np.histogram(tau, bins=tau_edges)
        return tau_hist




def main(dataparams,hyperparams):
    GCname,MWpotential,NPs,internal_dynamics,montecarloindex,MASS,RADIUS,fnametype = dataparams

    ndyn,ncpu,start_index=hyperparams
    montecarlokey   =   "monte-carlo-{:s}".format(str(montecarloindex).zfill(3))

    
    # extract the valid fnames 
    fnames,NPs=grab_valid_fnames(GCname, NPs, MWpotential, internal_dynamics, montecarlokey, MASS, RADIUS, mytype=fnametype)
    # get the total particle number 
    NP=NPs.sum()

    # see if the output is already made:
    outfname=taudensity_fname(GCname, NP, MWpotential, internal_dynamics, montecarlokey, MASS, RADIUS)
    if os.path.exists(outfname):
        print(outfname, "already exists. \n Delete if you want to recompute.")
        return

    # extract the time stamps 
    with h5py.File(fnames[0],"r") as myfile:
        time_stamps=myfile['time_stamps'][:]
    
    # extract the host orbit 
    hostorbit       =   gcs.extractors.GCOrbits.extract_whole_orbit(gcs.path_handler.GC_orbits(MWpotential=MWpotential,GCname=GCname),montecarlokey=montecarlokey)
    
    # initialize the output arrays
    tdynamical=median_dynamical_time(hostorbit)
    tau_edges=initialize_tau_edges(NP,tdynamical,ndyn)
    tau_centers, counts=initialize_2D_counts(tau_edges,len(time_stamps))

    
    END_FRAMES = len(time_stamps)

    pool = mp.Pool(processes=ncpu)
    # do this in parallel because its slow
    start_time = datetime.datetime.now()
    results = [pool.apply_async(loop_extract_density_profile, args=(i, time_stamps, fnames, NPs, hostorbit, ndyn, tau_edges))for i in range(start_index,END_FRAMES)]
    end_time = datetime.datetime.now()
    comp_time = end_time - start_time

    output = [p.get() for p in results]
    pool.close()
    pool.join()
    # stack the results
    for i in range(len(output)):
        counts[i+start_index] = output[i]
        # set the edges to zero as well 
        counts[i+start_index][0] = 0
        counts[i+start_index][-1] = 0
    

    # add attributes 
    ATTRS['GCname'] = GCname
    ATTRS['MWpotential'] = MWpotential
    ATTRS['montecarlokey'] = montecarlokey
    ATTRS['NP'] = NP
    ATTRS['MASS'] = MASS
    ATTRS['RADIUS'] = RADIUS
    ATTRS['ndyn'] = ndyn
    ATTRS['internal_dynamics'] = internal_dynamics
    ATTRS["computation_time"] = str(comp_time)
    ATTRS["ncpu"] = ncpu

    # create the output file and save it! 
    with h5py.File(outfname, 'w') as f:
        # create the attributes
        for key, value in ATTRS.items():
            f.attrs[key] = value
        # create the datasets
        f.create_dataset('time_stamps', data=time_stamps)
        f.create_dataset('tau_centers', data=tau_centers)
        f.create_dataset('counts', data=counts)
        f.create_dataset('inputdata/fnames', data=fnames)
        f.create_dataset("inputdata/NPs", data=NPs)

    print("done with {:s}".format(outfname))
    # clean up
    del counts
    del tau_centers
    del tau_edges
    del time_stamps
    del fnames
    del NPs
    del hostorbit   
    del pool
    del results
    del output
    del start_time
    del end_time
    del comp_time
    del myfile


    return None





if __name__=="__main__":
    # dataparams
    GCname              =   "Pal5"
    MWpotential         =   "pouliasis2017pii-GCNBody"
    NPs                 =   np.arange(3317,3347+1,1)
    internal_dynamics   =   "isotropic-plummer_mass_radius_grid"
    montecarloindex     =   9
    fnametype           =   'physical'

    
    # hyper params
    ndyn,ncpu,start_index= 3,10,10

    myinput=int(sys.argv[1])
    nmass = 5
    nradius = 5
    MASS_INDEX = myinput // nradius
    RADIUS_INDEX = myinput % nradius
    MASS = int(np.floor(MASS_GRID[MASS_INDEX]))
    RADIUS = int(np.floor(1000*RADIUS_GRID[RADIUS_INDEX]))    

    dataparams = GCname,MWpotential,NPs,internal_dynamics,montecarloindex,MASS,RADIUS,fnametype
    hyperparams = ndyn,ncpu,start_index
    # run the main function
    main(dataparams,hyperparams)
    



