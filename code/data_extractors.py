import numpy as np
import h5py
import astropy.units as u # type: ignore
import sys 
codefilepath = "/obs/sferrone/gc-tidal-loss/code/"
sys.path.append(codefilepath)
import StreamOrbitCoords as SOC






###################################################
#################### COMPOSITE ####################
###################################################
def coordinate_impact_indicies_from_force_file(
    force_file:h5py._hl.files.File,
    streamOrbit:h5py._hl.files.File,
    perturberOrbit:h5py._hl.files.File,
    monte_carlo_key:str,
    perturberName:str,
    unitT:u.core.CompositeUnit=u.kpc/(u.km/u.s)):


    # extract the time stamps that were saved from the stream orbit
    stream_orbit_time_stamps = extract_time_steps_from_stream_orbit(streamOrbit)
    # extract the impact time from the force file
    impact_time   =   extract_impact_time_from_force_file(force_file,perturber_name=perturberName)
    # get the time index for the base simulation
    index_simulation_time_of_impact=int(\
        np.argmin(np.abs(perturberOrbit[monte_carlo_key]['t'][:]-impact_time)))
    # find the time index for the stream orbit. 
    index_stream_orbit_time_of_impact=np.argmin(np.abs(stream_orbit_time_stamps.value-impact_time))
    return index_simulation_time_of_impact, index_stream_orbit_time_of_impact


def convert_instant_to_tail_coordinates(galactic_coordinates: tuple,
                                       orbit_galactic_coordinates: tuple,
                                       time_of_interest: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the stream and perturber galactic coordinates to tail coordinates at a given impact time.

    Parameters:
        stream_galactic_coordinates (np.ndarray): Array of stream galactic coordinates.
        host_orbit_galactic_coordinates (np.ndarray): Array of host orbit galactic coordinates.
        perturber_galactic_coordinates (np.ndarray): Array of perturber galactic coordinates.
        impact_time (float): Time of impact.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the stream time coordinate,
        stream tail coordinates, perturber time coordinate, and perturber tail coordinates.
        stream_time_coordinate is the time coordinate of the stream in tail coordinates, i.e. relative to the globular cluster.
        
        the time coordinate is the time ahead of the host globular cluster. 
    """

    TORB = orbit_galactic_coordinates[0]
    # put stream in tail coordinates
    xpT,ypT,zpT,vxT,vyT,vzT,indexes_pT  =   SOC.transform_from_galactico_centric_to_tail_coordinates(
                                                *galactic_coordinates,
                                                *orbit_galactic_coordinates,
                                                t0=time_of_interest)
    # get the perturber in tail coordinates
    tail_coordinates             =   np.array([xpT,ypT,zpT,vxT,vyT,vzT])
    time_coordinate              =   TORB[indexes_pT] - time_of_interest


    
    return time_coordinate,tail_coordinates
  
  
  

##################################################
##################### ORBITS #####################
##################################################
def get_orbit(path_orbit,mcarlokey):
    with h5py.File(path_orbit,'r') as fp:
        thost=fp[mcarlokey]['t'][:]
        xhost=fp[mcarlokey]['xt'][:]
        yhost=fp[mcarlokey]['yt'][:]
        zhost=fp[mcarlokey]['zt'][:]
        vxhost=fp[mcarlokey]['vxt'][:]
        vyhost=fp[mcarlokey]['vyt'][:]
        vzhost=fp[mcarlokey]['vzt'][:]
    return thost,xhost,yhost,zhost,vxhost,vyhost,vzhost




############################################
################## STREAM ##################
############################################
def get_stream(path_stream,mcarlokey,NP=int(1e5),internal_dynamics="isotropic-plummer"):
    with h5py.File(path_stream,'r') as fp:
        xstream=fp[internal_dynamics][str(NP)][mcarlokey]['x'][:]
        ystream=fp[internal_dynamics][str(NP)][mcarlokey]['y'][:]
        zstream=fp[internal_dynamics][str(NP)][mcarlokey]['z'][:]
        vxstream=fp[internal_dynamics][str(NP)][mcarlokey]['vx'][:]
        vystream=fp[internal_dynamics][str(NP)][mcarlokey]['vy'][:]
        vzstream=fp[internal_dynamics][str(NP)][mcarlokey]['vz'][:]
        tesc=fp[internal_dynamics][str(NP)][mcarlokey]['tesc'][:]
    return (xstream,ystream,zstream,vxstream,vystream,vzstream,),tesc

##################################################
################## STREAM ORBIT ##################
##################################################
def extract_time_steps_from_stream_orbit(
    streamOrbit:h5py._hl.files.File,
    outunitT:u.core.CompositeUnit=u.kpc/(u.km/u.s)):
    """
    Extracts the time steps from a stream orbit file.

    Parameters:
    - streamOrbit (h5py._hl.files.File): The path to the stream orbit file.

    Returns:
    - tStamps (numpy.ndarray): An array of time steps in integration units.
    """
    tStamps=[int(x) for x in streamOrbit["timestamps"]]
    unitT=u.Unit(streamOrbit.attrs['unitT'])
    tStamps=np.array(tStamps)
    tStamps=(np.sort(tStamps)*streamOrbit.attrs["dt"]*streamOrbit.attrs["writeStreamNSkip"] + streamOrbit.attrs["initialtime"])*u.Unit(streamOrbit.attrs['unitT'])
    tStamps=tStamps.to(outunitT)
    return tStamps


def extract_stream_from_path(index_stream_orbit,path_stream_orbit):
    with h5py.File(path_stream_orbit, 'r') as stream:
        xp=stream['timestamps'][str(index_stream_orbit)][0,:]
        yp=stream['timestamps'][str(index_stream_orbit)][1,:]
        zp=stream['timestamps'][str(index_stream_orbit)][2,:]
        vxp=stream['timestamps'][str(index_stream_orbit)][3,:]
        vyp=stream['timestamps'][str(index_stream_orbit)][4,:]
        vzp=stream['timestamps'][str(index_stream_orbit)][5,:]
    return xp,yp,zp,vxp,vyp,vzp


def get_galactic_coordinates_of_stream(
    index_stream_orbit:int,\
    streamOrbit:h5py._hl.files.File,):
    """
    Retrieves the galactic coordinates of a stream from the given streamOrbit dataset.

    Parameters:
    index_stream_orbit (int): The index of the stream orbit.
    streamOrbit (h5py._hl.files.File): The streamOrbit dataset.

    Returns:
    stream_galactic_coordinates: The galactic coordinates of the stream.
    """
    stream_galactic_coordinates=streamOrbit['timestamps'][str(index_stream_orbit)][:]
    return stream_galactic_coordinates
   
    
############################################################
###################### FORCE ON ORBIT ######################
############################################################
def extract_impact_time_from_force_file(
    force_file:h5py._hl.files.File,
    perturber_name:str):
    """
    Calculates the impact time from a force file.

    Parameters:
    - force_file  (h5py._hl.files.File): the force file containing the acceleration data.
    - perturber_name (str): The name of the perturber.

    Returns:
    - impactTime (Quantity): The impact time in integration units.
    """
    
    Xs, _, magA = extract_acceleration_arrays_from_force_file(force_file, perturber_name)
    magA=magA.T
    timeDex, shiftDex = np.unravel_index(np.argmax(magA), magA.shape)
    impactTime = Xs[timeDex, shiftDex]
    impactTime = impactTime

    # convert to unitT
    return impactTime


def extract_acceleration_arrays_from_force_file(
    force_file:h5py._hl.files.File, 
    perturber:str):
    """
    Get the acceleration arrays from a force file for a specific perturber. T

    Parameters:
    - force_file (h5py._hl.files.File): The force file containing the acceleration data.
    - perturber (str): The name of the perturber to extract the acceleration data for.

    Returns:
    TIMES ARE IN UNITS OF MYRS
    - Xs (ndarray): 2D array of x-coordinates for the acceleration data.
    - Ys (ndarray): 2D array of y-coordinates for the acceleration data.
    - magA (ndarray): 2D array of magnitudes of acceleration from the perturber on orbit

    """
    perturbers=force_file['pertrubers'][:].astype(str)
    # if it is a string, convert it to a list
    if isinstance(perturbers,str):
        perturbers=[perturbers]
    # get index of perturber in the 3D force array
    i=np.where(perturbers==perturber)[0][0]
    # make the time axes
    dts     =   force_file['tTarg'][force_file['comTimeIndexes'][0]+force_file["samplingIndexes"][:]] - force_file['tTarg'][force_file['comTimeIndexes'][0]]
    xs      =   force_file["tSampling"][:]
    ys      =   dts
    Xs,Ys   =   np.meshgrid(xs,ys)
    # get the accelerations
    magA=np.sqrt(force_file["AXs"][:,i,:]**2 + force_file["AYs"][:,i,:]**2 + force_file["AZs"][:,i,:]**2)
    Xs,Ys=Xs,Ys
    return Xs,Ys,magA











########################################################
########################################################

######################## HELPERS ######################

########################################################
########################################################


















######################## ######################## #####
######################## ORBIT ########################
#################### ######################## #########
def filter_orbit_by_fixed_time(t,orbit,current_index,filter_time):
    xt,yt,zt,vxt,vyt,vzt=orbit
    down_time = t[current_index] - filter_time
    down_dex = np.argmin(np.abs(t-down_time))
    up_time = t[current_index] + filter_time
    up_dex = np.argmin(np.abs(t-up_time))
    tOrb=t[down_dex:up_dex]
    xOrb=xt[down_dex:up_dex]
    yOrb=yt[down_dex:up_dex]
    zOrb=zt[down_dex:up_dex]
    vxOrb=vxt[down_dex:up_dex]
    vyOrb=vyt[down_dex:up_dex]
    vzOrb=vzt[down_dex:up_dex]

    return tOrb,xOrb,yOrb,zOrb,vxOrb,vyOrb,vzOrb


def filter_orbit_by_dynamical_time(\
    tORB:np.ndarray, \
    orbit:tuple,\
    time_of_interest:float,\
    n_dynamic_time:float=2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the orbit coordinates based on the dynamical time about a time of interest.
    This is useful since I am doing a naive computation of finding the nearest point of each particle to the orbit.

    Args:
        tORB (np.ndarray): Array of time values.
        xtORB (np.ndarray): Array of x-coordinate values.
        ytORB (np.ndarray): Array of y-coordinate values.
        ztORB (np.ndarray): Array of z-coordinate values.
        vxtORB (np.ndarray): Array of x-velocity values.
        vytORB (np.ndarray): Array of y-velocity values.
        vztORB (np.ndarray): Array of z-velocity values.
        currenttime (float): Current time value.
        nDynTimes (int): Number of dynamical times.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
        the filtered orbit coordinates: TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB.
    """
    XORB, YORB, ZORB, VXORB, VYORB, VZORB = orbit
    assert ( n_dynamic_time > 0), "nDynTimes must be greater than 0"
    assert ( n_dynamic_time < 5), "nDynTimes must be less than 5"
    assert( time_of_interest > np.min(tORB)), "time_of_interest must be greater than the minimum time in tORB"
    assert( time_of_interest < np.max(tORB)), "time_of_interest must be less than the maximum time in tORB"
    
    rORB = np.sqrt(XORB ** 2 + YORB ** 2 + ZORB ** 2)
    vORB = np.sqrt(VXORB ** 2 + VYORB ** 2 + VZORB ** 2)

    todayindex  = np.argmin(np.abs(tORB - time_of_interest))
    ttoday      = tORB[todayindex]

    Tdyn_all    = rORB / vORB
    Tdyn        = np.mean(Tdyn_all)

    cond1   = (tORB - ttoday) < (n_dynamic_time * Tdyn)
    cond2   = (tORB - ttoday) > (-n_dynamic_time * Tdyn)
    cond    = cond1 * cond2

    XORB    = XORB[cond]
    YORB    = YORB[cond]
    ZORB    = ZORB[cond]
    VXORB   = VXORB[cond]
    VYORB   = VYORB[cond]
    VZORB   = VZORB[cond]
    TORB    = tORB[cond]
    
    return TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB

    
####################################    
############## STREAM ##############
####################################
def filter_impacted_stream_side(
    x_stream_tail_coordinates: np.ndarray, \
    x_perturber_tail_coordinates: float) -> np.ndarray:
    """
    Filter the impacted stream side based on the coordinates of the stream tail and perturber tail.

    Parameters:
        x_stream_tail_coordinates (np.ndarray): The coordinates of the stream tail.
        x_perturber_tail_coordinates (float): The coordinates of the perturber tail.

    Returns:
        np.ndarray: An array indicating the impacted side of the stream.

    """
    if x_perturber_tail_coordinates > 0:
        impacted_side = x_stream_tail_coordinates > 0
    else:
        impacted_side = x_stream_tail_coordinates < 0
    return impacted_side


def filter_stream_in_tail_coordinates(
    stream_tail_coordinates: np.ndarray,
    xlim: float,
    ylim: float,
    zlim: float) -> np.ndarray:
    """
    Filters the stream tail coordinates based on the given limits.

    Parameters:
        stream_tail_coordinates (np.ndarray): The stream tail coordinates to filter.
        xlim (float): The limit for the x-coordinate.
        ylim (float): The limit for the y-coordinate.
        zlim (float): The limit for the z-coordinate.

    Returns:
        np.ndarray: A boolean array indicating which coordinates pass the filter.

    """
    cond0 = np.abs(stream_tail_coordinates[0, :]) < xlim
    cond1 = np.abs(stream_tail_coordinates[1, :]) < ylim
    cond2 = np.abs(stream_tail_coordinates[2, :]) < zlim
    cond = cond0 & cond1 & cond2
    return cond

####################################
###### TO BE MOVED OR DELETED ######
####################################
def get_velocity_change_impact(
    pathStreamOrbit,
    pathPal5Orbit,
    montecarlokey,
    logicalParticles,
    impactTime,
    impactDuration,
    timeWindowFactor=1,
    unitT=u.kpc/(u.km/u.s),
    nDynTimes=3):
    """
    
    Also assumes that you know the correct moment of the impact. 
    
    Assumes that the impactDuration was computed else where.
    It was found from doing 2b/W, where b is the impact parameter, W is the 
    relative speed of the perturber and the stream. 
    
    Also assumes that you know which subset of particles to study. 
        ideally, this is found by grabing all particles within one FWHM radius
        of the impact as reported in Erkal et al 2015. 
    """
    
    # import the stream orbit data
    streamOrbit=h5py.File(pathStreamOrbit,"r")
    # extract the orbit sinippit at the time of impact
    tP5,xP5,yP5,zP5,vxP5,vyP5,vzP5=SOC.dynamicalTimeFiler(
        pathPal5Orbit,
        montecarlokey=montecarlokey,
        currenttime=impactTime.to(unitT).value,
        nDynTimes=nDynTimes)
    
    # get the indicies of interest 
    downTime = impactTime.to(unitT) - timeWindowFactor*impactDuration.to(unitT)
    upTime = impactTime.to(unitT) + timeWindowFactor*impactDuration.to(unitT)
    # extract the timeStamps from the streamOrbit
    tStamps=extract_time_steps_from_stream_orbit(pathStreamOrbit)
    
    # get the index of the impact time
    downTimeIndex = np.argmin(np.abs(tStamps.value-downTime.value))
    upTimeIndex = np.argmin(np.abs(tStamps.value-upTime.value))

    # extract the particle time stamps
    xp00,yp00,zp00,vxp00,vyp00,vzp00=streamOrbit['timestamps'][str(downTimeIndex)]
    xp01,yp01,zp01,vxp01,vyp01,vzp01=streamOrbit['timestamps'][str(upTimeIndex)]
    # convert to tail coordinates
    xprime00,yprime00,zprime00,vxprime00,vyprime00,vzprime00,indx=\
        SOC.transformGalcentricToOrbit(\
        xp00[logicalParticles],
        yp00[logicalParticles],
        zp00[logicalParticles],
        vxp00[logicalParticles],
        vyp00[logicalParticles],
        vzp00[logicalParticles],tP5,xP5,yP5,zP5,vxP5,vyP5,vzP5,t0=downTime.to(unitT).value)
    xprime01,yprime01,zprime01,vxprime01,vyprime01,vzprime01,indx=\
        SOC.transformGalcentricToOrbit(\
        xp01[logicalParticles],
        yp01[logicalParticles],
        zp01[logicalParticles],
        vxp01[logicalParticles],
        vyp01[logicalParticles],
        vzp01[logicalParticles],tP5,xP5,yP5,zP5,vxP5,vyP5,vzP5,t0=upTime.to(unitT).value)
    
    # GO FROM MY TAIL COORDINATES TO ERKAL'S COORDAINTES 
    initCoords = np.zeros((6,xprime00.shape[0]))
    finalCoords = np.zeros((6,xprime01.shape[0]))
    initCoords[0,:],initCoords[1,:],initCoords[2,:]=-yprime00,xprime00,zprime00
    initCoords[3,:],initCoords[4,:],initCoords[5,:]=-vyprime00,vxprime00,vzprime00
    finalCoords[0,:],finalCoords[1,:],finalCoords[2,:]=-yprime01,xprime01,zprime01
    finalCoords[3,:],finalCoords[4,:],finalCoords[5,:]=-vyprime01,vxprime01,vzprime01
    return initCoords,finalCoords



