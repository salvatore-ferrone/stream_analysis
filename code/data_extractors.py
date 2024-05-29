import numpy as np
import h5py
import astropy.units as u
import sys 
codefilepath = "/obs/sferrone/gc-tidal-loss/code/"
sys.path.append(codefilepath)
import StreamOrbitCoords as SOC





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



def get_galactic_coordinates_of_host_orbit_and_perturber(
    index_simulation,\
    pal5Orbit,\
    perturberOrbit,\
    monte_carlo_key):
    """
    Get the stream line from a stream file.
    Parameters:
    - pathStreamFile (str): The path to the stream file.

    Returns:
    - streamLine (numpy.ndarray): The stream line.
    """
    # hard coded parameters
    nDynTimes=3
    unitT=u.kpc/(u.km/u.s)

    # extract the orbit os the host 
    tP5,xtP5,ytP5,ztP5=pal5Orbit[monte_carlo_key]['t'][:],pal5Orbit[monte_carlo_key]['xt'][:],pal5Orbit[monte_carlo_key]['yt'][:],pal5Orbit[monte_carlo_key]['zt'][:]
    vxtP5,vytP5,vztP5= pal5Orbit[monte_carlo_key]['vxt'][:],pal5Orbit[monte_carlo_key]['vyt'][:],pal5Orbit[monte_carlo_key]['vzt'][:]
    # grab the time of impact as reported here
    impactTime=tP5[index_simulation]*unitT
    # take only a snippit of the orbit
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB=SOC.filter_orbit_by_dynamical_time(\
        tP5,xtP5,ytP5,ztP5,vxtP5,vytP5,vztP5,time_of_interest=impactTime.to(unitT).value,nDynTimes=nDynTimes)
    # extract the position of the perturber at this time stamp
    xGC=np.array([perturberOrbit[monte_carlo_key]["xt"][index_simulation]])
    yGC=np.array([perturberOrbit[monte_carlo_key]["yt"][index_simulation]])
    zGC=np.array([perturberOrbit[monte_carlo_key]["zt"][index_simulation]])
    vxGC=np.array([perturberOrbit[monte_carlo_key]["vxt"][index_simulation]])
    vyGC=np.array([perturberOrbit[monte_carlo_key]["vyt"][index_simulation]])
    vzGC=np.array([perturberOrbit[monte_carlo_key]["vzt"][index_simulation]])
    
    host_orbit_galactic_coordinates = np.array([TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB])
    perturber_galactic_coordinates = np.array([xGC,yGC,zGC,vxGC,vyGC,vzGC])
    
    return host_orbit_galactic_coordinates, perturber_galactic_coordinates
    

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
    index_simulation_time_of_impact=int(np.argmin(np.abs(perturberOrbit[monte_carlo_key]['t'][:]-impact_time.to(unitT).value)))
    # find the time index for the stream orbit. 
    index_stream_orbit_time_of_impact=np.argmin(np.abs(stream_orbit_time_stamps-impact_time.to(unitT)))
    return index_simulation_time_of_impact, index_stream_orbit_time_of_impact

  
  
  
##################################################
################## STREAM ORBIT ##################
##################################################
def extract_time_steps_from_stream_orbit(
    streamOrbit:h5py._hl.files.File,
    unitT=(u.kpc/(u.km/u.s))):
    """
    Extracts the time steps from a stream orbit file.

    Parameters:
    - streamOrbit (h5py._hl.files.File): The path to the stream orbit file.

    Returns:
    - tStamps (numpy.ndarray): An array of time steps in integration units.
    """
    tStamps=[int(x) for x in streamOrbit["timestamps"]]
    tStamps=np.array(tStamps)
    tStamps=(np.sort(tStamps)*streamOrbit.attrs["dt"]*streamOrbit.attrs["writeStreamNSkip"] + streamOrbit.attrs["initialtime"])*u.Unit(streamOrbit.attrs['unitT'])
    tStamps=tStamps.to(unitT)
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
   
    
#############################################
################ FORCE FILES ################
#############################################





############################################################
###################### FORCE ON ORBIT ######################
############################################################
def extract_impact_time_from_force_file(
    force_file:h5py._hl.files.File,
    perturber_name:str,
    unitT=u.kpc/(u.km/u.s),
    baseUnitT=u.Myr):
    """
    Calculates the impact time from a force file.

    Parameters:
    - force_file  (h5py._hl.files.File): the force file containing the acceleration data.
    - perturber_name (str): The name of the perturber.

    Returns:
    - impactTime (Quantity): The impact time in integration units.
    """
    
    Xs, _, magA = extract_acceleration_arrays_from_force_file(force_file, perturber_name)
    timeDex, shiftDex = np.unravel_index(np.argmax(magA), magA.shape)
    impactTime = Xs[timeDex, shiftDex] * baseUnitT
    impactTime = impactTime.to(unitT)
    # convert to unitT
    return impactTime


def extract_acceleration_arrays_from_force_file(
    force_file:h5py._hl.files.File, 
    perturber:str,
    outTimeUnit:u.Unit=u.Myr):
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
    dts=force_file['tTarg'][force_file['comTimeIndexes'][0]+force_file["samplingIndexes"][:]] - force_file['tTarg'][force_file['comTimeIndexes'][0]]
    xs=force_file["tSampling"][:]
    ys=dts
    Xs,Ys=np.meshgrid(xs,ys)
    # get the accelerations
    magA=np.sqrt(force_file["AXs"][:,i,:]**2 + force_file["AYs"][:,i,:]**2 + force_file["AZs"][:,i,:]**2)
    Xs,Ys=Xs.T,Ys.T
    return Xs,Ys,magA


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