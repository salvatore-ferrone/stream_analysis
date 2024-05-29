import numpy as np 
import h5py 
import sys 
from astropy import units as u
sys.path.append("/obs/sferrone/gc-tidal-loss/bin")
import potentials

######################################
####### COMPUTATION FUNCTIONS ########
######################################


def GetTemporalIndexesForOrbitSampling(GCname: str, orbitpath: str, montecarlokey: str, orbitsamplingskip: int = 5, timeskip: int = 10, dtimefactor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts sample time stamps for the orbit of a Globular Cluster (GC) and shifts in time indexes for sampling the orbit.

    Parameters:
    GCname (str): The name of the Globular Cluster.
    orbitpath (str): The path to the directory containing the HDF5 files.
    montecarlokey (str): The key for accessing the Monte Carlo simulation data in the HDF5 file.
    orbitsamplingskip (int, optional): At a given time step, the number of indices to skip while sampling the orbit. Defaults to 5.
    timeskip (int, optional): Sampling the times of the orbit of the GC. Defaults to 10.
    dtimefactor (int, optional): The dynamical time means we're going above and behind the GC by dtimefactor times. Defaults to 1.

    Returns:
    tuple: Returns a tuple containing arrays of the time indexes and the sampling index shift.
    """
    with h5py.File(orbitpath+GCname+"-orbits.hdf5",'r') as target:
        
        # open the orbit of the cluster
        # take from minus one dynamical time to the present day  
        r=np.sqrt(target[montecarlokey]['xt'][:]**2+target[montecarlokey]['yt'][:]**2+target[montecarlokey]['zt'][:]**2)
        v=np.sqrt(target[montecarlokey]['vxt'][:]**2+target[montecarlokey]['vyt'][:]**2+target[montecarlokey]['vzt'][:]**2)
        dynamicalTime=np.mean(r/v)
        cond0=target[montecarlokey]["t"][:]>target[montecarlokey]["t"][0]+dynamicalTime
        cond1=target[montecarlokey]["t"][:]<0
        cond=np.logical_and(cond0,cond1)
        # get the time indexes
        timedexes=np.arange(len(target[montecarlokey]["t"]))[cond]
        timedexes=timedexes[::timeskip]
        # get the total length of the orbit to sample (in time steps)
        cond0=target[montecarlokey]['t'] < dynamicalTime*dtimefactor
        cond1=target[montecarlokey]['t'] > -dynamicalTime*dtimefactor
        cond2=np.logical_and(cond0,cond1)
        # divide the number of points by the orbitsamplingskip
        divisor=cond2.sum()//orbitsamplingskip
        npoints=divisor*orbitsamplingskip
        upsampling=np.arange(orbitsamplingskip,npoints//2)[::orbitsamplingskip]
        downsampling=-np.arange(npoints//2)[::orbitsamplingskip]
        samplingIndexShift=np.append(downsampling[::-1],upsampling)
    return timedexes,samplingIndexShift


def GetPerturberKinematicsAtSampledTimes(
        orbitfolder: str,
        montecarlokey: str,
        perturbers: list,
        myTimeIndexes: list) -> tuple:
    """
    Extracts the kinematics of perturbers at specific times from a given HDF5 file.

    Parameters:
    basepath (str): The base path to the directory containing the HDF5 files.
    gfieldname (str): The name of the gravitational field.
    montecarlokey (str): The key for accessing the Monte Carlo simulation data in the HDF5 file.
    perturbers (list): A list of perturbers for which to extract data.
    myTimeIndexes (list): A list of time indexes at which to sample the perturber data.

    Returns:
    tuple: Returns a tuple containing arrays of the X, Y, Z coordinates and VX, VY, VZ velocities of the perturbers at the sampled times, as well as an array of the perturber masses.
    """
    # open any mass to get the proper array size for logical slicing
    filename=orbitfolder+"/"+perturbers[0]+"-orbits.hdf5"
    with h5py.File(filename,'r') as myFile:
        pertubertimes=myFile[montecarlokey]["t"][:]
    comTimeDexes=np.array(myTimeIndexes,dtype=int) # make sure they are ints
    slicearray=np.in1d(np.arange(0,len(pertubertimes),1),comTimeDexes)
    # initialize arrays
    perturbersX = np.zeros((len(perturbers),slicearray.sum()))
    perturbersY = np.zeros((len(perturbers),slicearray.sum()))
    perturbersZ = np.zeros((len(perturbers),slicearray.sum()))
    perturbersVX=np.zeros((len(perturbers),slicearray.sum()))
    perturbersVY=np.zeros((len(perturbers),slicearray.sum()))
    perturbersVZ=np.zeros((len(perturbers),slicearray.sum()))
    perturberMasses=np.zeros(len(perturbers))
    # extract the data
    for i in range(len(perturbers)):
        filename=orbitfolder+"/"+perturbers[i]+"-orbits.hdf5"
        with h5py.File(filename,'r') as myFile:
            perturbersX[i,:]=myFile[montecarlokey]["xt"][slicearray]
            perturbersY[i,:]=myFile[montecarlokey]["yt"][slicearray]
            perturbersZ[i,:]=myFile[montecarlokey]["zt"][slicearray]
            perturbersVX[i,:]=myFile[montecarlokey]["vxt"][slicearray]
            perturbersVY[i,:]=myFile[montecarlokey]["vyt"][slicearray]
            perturbersVZ[i,:]=myFile[montecarlokey]["vzt"][slicearray]
            perturberMasses[i]=myFile[montecarlokey]['initialConditions']['Mass'][0]
    pertubertimes=pertubertimes[slicearray]
    return pertubertimes,\
        perturbersX.T,perturbersY.T,perturbersZ.T,\
        perturbersVX.T,perturbersVY.T,perturbersVZ.T,\
        perturberMasses


def ComputeOutForcesAtTimeStamp(G: float, xGCs: np.ndarray, yGCs: np.ndarray, zGCs: np.ndarray, masses: np.ndarray, samplingX: np.ndarray, samplingY: np.ndarray, samplingZ: np.ndarray) -> np.ndarray:
    """
    Evaluates the force from the GCs on some sampling points at a given time step.
    Intended to be from the GCs along various points of an orbit ahead and behind the target GC.

    Parameters:
    G (float): The gravitational constant.
    xGCs (np.ndarray): The x-coordinates of the GCs.
    yGCs (np.ndarray): The y-coordinates of the GCs.
    zGCs (np.ndarray): The z-coordinates of the GCs.
    masses (np.ndarray): The masses of the GCs.
    mytimeindex (int): The index of the time step.
    samplingIndexShift (np.ndarray): The shift in the time indexes for sampling the orbit.
    samplingX (np.ndarray): The x-coordinates of the sampling points.
    samplingY (np.ndarray): The y-coordinates of the sampling points.
    samplingZ (np.ndarray): The z-coordinates of the sampling points.

    Returns:
    np.ndarray: An array of the time indexes, the shift in the time indexes, the sampling points, and the accelerations.
    """
    assert isinstance(xGCs,np.ndarray),"xGCs is not an np.ndarray and is instead {:s}".format(str(type(xGCs)))
    assert isinstance(yGCs,np.ndarray),"yGCs is not an np.ndarray and is instead {:s}".format(str(type(yGCs)))
    assert isinstance(zGCs,np.ndarray),"zGCs is not an np.ndarray and is instead {:s}".format(str(type(zGCs)))
    assert isinstance(samplingX,np.ndarray),"samplingX is not an np.ndarray and is instead {:s}".format(str(type(samplingX)))
    assert isinstance(samplingY,np.ndarray),"samplingY is not an np.ndarray and is instead {:s}".format(str(type(samplingY)))


    assert isinstance(masses,np.ndarray)



    ax,ay,az,_=potentials.potentials.pointmassconfiguration(\
        G.value,masses,\
        xGCs,yGCs,zGCs,\
        samplingX,samplingY,samplingZ)


    outarray=np.zeros((samplingX.shape[0],2))
    outarray[:,0],outarray[:,1],outarray[:,2],outarray[:,3],outarray[:,4],outarray[:,5]=\
        samplingX,samplingY,samplingZ,ax,ay,az
    return outarray


def getApproxDistanceFromTime(pathtoOrbit: str, montecarlokey: str, timeindexes: np.ndarray, comDEX: int, integrationTimeUnit: u.Unit = u.kpc/(u.km/u.s)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the axis information for the plot_spectrogram function.

    Parameters:
    pathtoOrbit (str): The path to the HDF5 file containing the orbit data.
    montecarlokey (str): The key for accessing the Monte Carlo simulation data in the HDF5 file.
    timeindexes (np.ndarray): The time indexes for sampling the orbit.
    comDEX (int): The index of the center of mass.
    integrationTimeUnit (u.Unit, optional): The unit of time for integration. Defaults to u.kpc/(u.km/u.s).

    Returns:
    tuple: Returns a tuple containing arrays of the integration time steps (xaxistimes), the approximate distance ahead and behind the GC (approxDist), and the times ahead and behind the GC (yaxistimes).
    """
    with h5py.File(pathtoOrbit,'r') as myorbit:
        mytimes=myorbit[montecarlokey]["t"][:]
        myvx=myorbit[montecarlokey]["vxt"][:]
        myvy=myorbit[montecarlokey]["vyt"][:]
        myvz=myorbit[montecarlokey]["vzt"][:]
    v=np.sqrt(myvx**2+myvy**2+myvz**2)
    comtime=mytimes[timeindexes[1,comDEX]] # choice of 1 in arbitrary
    
    # the time head and behind the GC
    yaxistimes=mytimes[timeindexes[1,:]]-comtime
    approxDist=yaxistimes*np.mean(v)
    yaxistimes*=integrationTimeUnit
    yaxistimes=yaxistimes.to(u.Myr).value
    
    # the time steps of the GC COM
    xaxistimes=mytimes[timeindexes[:,comDEX]]
    xaxistimes*=integrationTimeUnit
    xaxistimes=xaxistimes.to(u.Gyr).value 
    return xaxistimes,approxDist,yaxistimes  

