import numpy as np
import h5py
from astropy import units as u



###########################################################################
######################## POST-PROCESSING FUNCTIONS ########################
###### FOR ANALYZING FORCEONORBIT DATA FILES AND EXTRACTING IMPACTS #######
###########################################################################

    
def getAccelArrayFromForceFile(forceFile:h5py._hl.files.File, perturber:str,outTimeUnit:u.Unit=u.Myr):
    """
    Get the acceleration arrays from a force file for a specific perturber.

    Parameters:
    - forceFile (h5py._hl.files.File): The force file containing the acceleration data.
    - perturber (str): The name of the perturber to extract the acceleration data for.

    Returns:
    TIMES ARE IN UNITS OF MYRS
    - Xs (ndarray): 2D array of x-coordinates for the acceleration data.
    - Ys (ndarray): 2D array of y-coordinates for the acceleration data.
    - magA (ndarray): 2D array of magnitudes of acceleration from the perturber on orbit

    """
    perturbers=forceFile['pertrubers'][:].astype(str)
    # if it is a string, convert it to a list
    if isinstance(perturbers,str):
        perturbers=[perturbers]
    # get index of perturber in the 3D force array
    i=np.where(perturbers==perturber)[0][0]
    # make the time axes
    dts=forceFile['tTarg'][forceFile['comTimeIndexes'][0]+forceFile["samplingIndexes"][:]] - forceFile['tTarg'][forceFile['comTimeIndexes'][0]]
    unitV=1*u.km/u.s
    unitT=u.kpc/unitV
    xs=forceFile["tSampling"][:]*unitT
    ys=dts*unitT
    xs,ys=xs.to(outTimeUnit).value,ys.to(outTimeUnit).value
    Xs,Ys=np.meshgrid(xs,ys)
    # get the accelerations
    magA=np.sqrt(forceFile["AXs"][:,i,:]**2 + forceFile["AYs"][:,i,:]**2 + forceFile["AZs"][:,i,:]**2)
    Xs,Ys=Xs.T,Ys.T
    mass=forceFile["masses"][i]
    return Xs,Ys,magA,mass

def getImpact(forceFile:h5py._hl.files.File, perturbOrb:h5py._hl.files.File, 
              pal5Orbit:h5py._hl.files.File, montecarlokey: str, 
              Xs:np.ndarray, Ys:np.ndarray, magA:np.ndarray,mintime:np.int64=-500):
    """
    Calculate the impact parameters and relative distances and velocities between a perturber and a target orbit.
        We store the impact time index, the shift index, the displacement, and the difference in velocity between the perturber and the target orbit.
        This occurs at the time of maximum acceleration. which is when the perturber is closest to the target orbit.
        This data can then be used to study the star particles that are most affected by the perturber.

    Parameters:
        forceFile (h5py._hl.files.File): Array containing the perturber data.
        perturbOrb (h5py._hl.files.File): File containing orbit of perturber
        pal5Orbit (h5py._hl.files.File): Dictionary containing the host orbit data.
        montecarlokey (str):  for accessing the specific orbit data.
        perturber (str): Name of the perturber.

    Returns:
        tuple: A tuple containing the following values:
            - TheTimeIndex (int): Time index for the impact.
            - theShiftIndx (int): shift in time ahead of impact time to recover the poisition along the orbit
            - dX (numpy.ndarray): Displacement, perturber - nearest point along orbit
            - dV (numpy.ndarray): difference in velocity, perturber - nearest point along orbit
            - maxAccel (float): Maximum acceleration value.
    """
    # shortening the time axis
    xaxismask=Xs[:,0] > mintime
    Xs,Ys,C=Xs[xaxismask,:],Ys[xaxismask,:],magA[xaxismask,:]
    temp=~xaxismask
    undoshorten=temp.sum()
    
    
    # Get the maximum value
    peakX,peakY=np.where(C==np.max(C))
    maxAccel=C[peakX[0],peakY[0]]
    #  the indicies for the impact time
    TheTimeIndex=forceFile["comTimeIndexes"][undoshorten+peakX[0]]
    theShiftIndx=forceFile["samplingIndexes"][peakY[0]]    
    
    # extract the perturber at this point
    xp=perturbOrb[montecarlokey]['xt'][TheTimeIndex]
    yp=perturbOrb[montecarlokey]['yt'][TheTimeIndex]
    zp=perturbOrb[montecarlokey]['zt'][TheTimeIndex]
    vxp=perturbOrb[montecarlokey]['vxt'][TheTimeIndex]
    vyp=perturbOrb[montecarlokey]['vyt'][TheTimeIndex]
    vzp=perturbOrb[montecarlokey]['vzt'][TheTimeIndex]
    
    # extract the point on the orbit closest to the impact
    xh= pal5Orbit[montecarlokey]['xt'][TheTimeIndex+theShiftIndx]
    yh= pal5Orbit[montecarlokey]['yt'][TheTimeIndex+theShiftIndx]
    zh= pal5Orbit[montecarlokey]['zt'][TheTimeIndex+theShiftIndx]
    vxh=pal5Orbit[montecarlokey]['vxt'][TheTimeIndex+theShiftIndx]
    vyh=pal5Orbit[montecarlokey]['vyt'][TheTimeIndex+theShiftIndx]
    vzh=pal5Orbit[montecarlokey]['vzt'][TheTimeIndex+theShiftIndx]

    # get relative distance and velocity
    dx,dy,dz=xp-xh,yp-yh,zp-zh
    dvx,dvy,dvz=vxp-vxh,vyp-vyh,vzp-vzh
    
    dX=np.array([dx,dy,dz])
    dV=np.array([dvx,dvy,dvz])
    
    return TheTimeIndex,theShiftIndx,dX,dV,maxAccel

def pack_orbit_snippits(perturberOrbit:h5py._hl.files.File, pal5Orbit:h5py._hl.files.File, 
                        montecarlokey:str, TheTimeIndex:int, theShiftIndx:int, margin:int=350):
    """
    Packs the relevant orbit data for a perturber and a host object at a specific time index.
    
        It's to coordinate between the data set containing the perturber's orbit, the orbit of the host, and the time index of the impact.
            the indicies between the data sets are different. This code is used be first to get the impact time index and then to get the orbit data at that time index.

    Parameters:
    - perturberOrbit (h5py._hl.files.File): HDF5 containing the perturber's orbit data.
    - pal5Orbit (h5py._hl.files.File): HDF5 containing the host object's orbit data.
    - montecarlokey (str): Key to access the specific perturber orbit data within the perturberOrbit dictionary.
    - TheTimeIndex (int): Index of the desired time step in the orbit data.
    - theShiftIndx (int): Index of the desired shifted time step in the host object's orbit data.
    - margin (int, optional): Number of time steps to include before and after the desired time index for the orbit. Default is 350.

    Returns:
    - orbitPerturber (tuple): Tuple containing the perturber's x, y, and z coordinates.
    - orbitHost (tuple): Tuple containing the host object's x, y, and z coordinates.
    - perturberKinematics (tuple): Tuple containing the perturber's x, y, z, vx, vy, and vz coordinates.
    - hostPosition (tuple): Tuple containing the host object's x, y, and z coordinates at the time of impact.
    - impactPointKinematics (tuple): Tuple containing the host object's x, y, z, vx, vy, and vz coordinates at the shifted time step.
    - dX (tuple): Tuple containing the displacement in x, y, and z coordinates between the perturber and the impact point.
    - dV (tuple): Tuple containing the relative velocity in vx, vy, and vz coordinates between the perturber and the impact point.
    """
    
    # extract the perturber orbit
    xps=perturberOrbit[montecarlokey]['xt'][TheTimeIndex-margin:TheTimeIndex+margin]
    yps=perturberOrbit[montecarlokey]['yt'][TheTimeIndex-margin:TheTimeIndex+margin]
    zps=perturberOrbit[montecarlokey]['zt'][TheTimeIndex-margin:TheTimeIndex+margin]
    # extract perturber at time of impact
    xp=  perturberOrbit[montecarlokey]['xt'][TheTimeIndex]
    yp=  perturberOrbit[montecarlokey]['yt'][TheTimeIndex]
    zp=  perturberOrbit[montecarlokey]['zt'][TheTimeIndex]
    vxp=perturberOrbit[montecarlokey]['vxt'][TheTimeIndex]
    vyp=perturberOrbit[montecarlokey]['vyt'][TheTimeIndex]
    vzp=perturberOrbit[montecarlokey]['vzt'][TheTimeIndex]
    # extract host orbit
    xhs=pal5Orbit[montecarlokey]['xt'][TheTimeIndex-margin:TheTimeIndex+margin]
    yhs=pal5Orbit[montecarlokey]['yt'][TheTimeIndex-margin:TheTimeIndex+margin]
    zhs=pal5Orbit[montecarlokey]['zt'][TheTimeIndex-margin:TheTimeIndex+margin]
    # extract the host position at the time of impact
    xh = pal5Orbit[montecarlokey]['xt'][TheTimeIndex]
    yh = pal5Orbit[montecarlokey]['yt'][TheTimeIndex]
    zh = pal5Orbit[montecarlokey]['zt'][TheTimeIndex]
    
    # extract the impacted point along orbit 
    xipp=  pal5Orbit[montecarlokey]['xt'][TheTimeIndex+theShiftIndx]
    yipp=  pal5Orbit[montecarlokey]['yt'][TheTimeIndex+theShiftIndx]
    zipp=  pal5Orbit[montecarlokey]['zt'][TheTimeIndex+theShiftIndx]
    vxipp=pal5Orbit[montecarlokey]['vxt'][TheTimeIndex+theShiftIndx]
    vyipp=pal5Orbit[montecarlokey]['vyt'][TheTimeIndex+theShiftIndx]
    vzipp=pal5Orbit[montecarlokey]['vzt'][TheTimeIndex+theShiftIndx]
    
    # displacement and relatice vel
    dX=(xp-xipp,yp-yipp,zp-zipp)
    dV=(vxp-vxipp,vyp-vyipp,vzp-vzipp)
    
    # pack up the data
    orbitPerturber=(xps,yps,zps)
    orbitHost=(xhs,yhs,zhs)
    perturberKinematics=(xp,yp,zp,vxp,vyp,vzp)
    hostPosition=(xh,yh,zh)
    impactPointKinematics = (xipp,yipp,zipp,vxipp,vyipp,vzipp)
    return orbitPerturber,orbitHost,perturberKinematics,hostPosition,impactPointKinematics,dX,dV
