import ForceOnOrbitNearGC as FOORB
import numpy as np 
from astropy import units as u
from astropy import constants as const
import h5py
import sys 
sys.path.append("../bin/")
import potentials
from matplotlib import pyplot as plt

##############################################  
######### MULTIPROCESSING FUNCTIONS ##########
##############################################


def ComputeForceOnOrbit(\
        targetGCname:str,
        montecarlokey:str,
        gfieldname:str,
        orbitpath:str,
        perturberlistpath:str,
        fnameNote:str = "") -> None:
    """
    Computes the gravitational force on a target globular cluster (GC) due to potential perturbers.

    Parameters:
        targetGCname (str): Name of the target globular cluster.
        montecarlokey (str): Key for the Monte Carlo simulation run.
        gfieldname (str): Name of the gravitational field.
        orbitpath (str): Path to the file containing the orbit data.
        perturberlistpath (str): Path to the file containing the list of potential perturbers.

    Returns:
        None: The function writes the computed forces to an output file and does not return a value.

    The function first loads the positions and velocities of the target and potential perturbers at different times.
    It then computes the gravitational force on the target due to each perturber at each time step.
    The computed forces are stored in a 3D array, with one dimension for time, one for the perturbers, and one for the spatial coordinates.
    Finally, the function writes the computed forces and other relevant data to an output file.
    """
    outpath="/scratch2/sferrone/simulations/ForceOnOrbit/"+gfieldname+"/"+targetGCname+"/"+targetGCname+"-"+montecarlokey+fnameNote+"-FORCEONORBIT.hdf5"
    # set the initial conditions and such
    Gunit = (u.km/u.s)**2 * u.kpc / u.Msun
    G=const.G.to(Gunit).value
    orbitsamplingskip,timeskip,dtimefactor = 5,10,1
    # get the list of Palomar 5 suspected perturbers
    suspects=np.loadtxt(perturberlistpath,usecols=(0,),dtype=str).tolist()
    # if suspects is a string, convert it to a list
    if isinstance(suspects,str):
        suspects=[suspects]
    # get the sampling times, and the shift from the host to the sampling pints
    timedexes,samplingIndexShift=FOORB.GetTemporalIndexesForOrbitSampling(\
        targetGCname, orbitpath, montecarlokey, orbitsamplingskip, timeskip, dtimefactor)
    # define the data sizes
    Ntimesteps,Nperturbers,Nsamples=len(timedexes),len(suspects),len(samplingIndexShift)
    # estimate file size
    filesize=3*Ntimesteps*Nperturbers*Nsamples*8*u.byte
    filesize=filesize.to(u.GB)
    print("Estimated filesize: ",filesize)
    # grab the positions of Target
    fname = orbitpath+targetGCname+"-orbits.hdf5"
    with h5py.File(fname,'r') as fp:
        tTarg=fp[montecarlokey]["t"][:]
        xTarg=fp[montecarlokey]["xt"][:]
        yTarg=fp[montecarlokey]["yt"][:]
        zTarg=fp[montecarlokey]["zt"][:]
    # get the positions of the perturbers at these times
    (_,\
        perturbersX,perturbersY,perturbersZ,\
        _,_,_,\
        perturberMasses) = FOORB.GetPerturberKinematicsAtSampledTimes(orbitpath,montecarlokey=montecarlokey,\
            perturbers=suspects,myTimeIndexes=timedexes,)
        
    # initialize the arrays to store the data
    AXs = np.zeros((Ntimesteps, Nperturbers, Nsamples))
    AYs,AZs = AXs.copy(),AXs.copy()

    # fill the arrays
    for i in range(Ntimesteps):
        slicearray=np.in1d(np.arange(0,len(tTarg),1),timedexes[i]+samplingIndexShift)
        for jj in range(Nperturbers):
            AXs[i,jj,:],AYs[i,jj,:],AZs[i,jj,:],_ = \
                potentials.potentials.pointmassconfiguration(
                    G,perturberMasses[jj],perturbersX[i,jj],perturbersY[i,jj],perturbersZ[i,jj],
                    xTarg[slicearray],yTarg[slicearray],zTarg[slicearray])

    # save the data
    outfile=h5py.File(outpath,'a')
    outfile.create_dataset("pertrubers", data=np.array(suspects, dtype=h5py.string_dtype()))
    slicearray=np.in1d(np.arange(0,len(tTarg),1),timedexes[:])
    outfile.create_dataset("tSampling",data=tTarg[slicearray])
    outfile.create_dataset("tTarg",data=tTarg)
    outfile.create_dataset("xTarg",data=xTarg)
    outfile.create_dataset("yTarg",data=yTarg)
    outfile.create_dataset("zTarg",data=zTarg)
    outfile.create_dataset("samplingIndexes",data=samplingIndexShift)
    outfile.create_dataset("comTimeIndexes",data=timedexes)
    outfile.create_dataset("AXs",data=AXs)
    outfile.create_dataset("AYs",data=AYs)
    outfile.create_dataset("AZs",data=AZs)
    outfile.create_dataset("masses",data=perturberMasses)
    outfile.create_dataset("montecarlokey",data=montecarlokey)
    outfile.create_dataset("orbitpath",data=orbitpath)
    outfile.create_dataset("targetGCname",data=targetGCname)    
    outfile.close()
    print("Finished computing force on orbit for "+targetGCname+" "+montecarlokey+"\n"+outpath)
    




def storeNGC2808impactInformation(montecarlokey:str):
    """Purpose
    
    NGC is responsible for displacing many stars in the Palomar 5 stream depending on the its orbit and the stream's orbit.
    Let us store the impact parameter, the time of impact, and the velocity of the impact.
    
    """
    perturber="NGC2808"
    
    # open the force file
    basepath="/scratch2/sferrone/simulations/ForceOnOrbit/pouliasis2017pii-GCNBody/Pal5/Pal5-"
    mypath=basepath+montecarlokey+"-NGC2808-only-FORCEONORBIT.hdf5"
    forceFile=h5py.File(mypath,"r")
    
    # open the orbit file of the perturber
    basepath="/scratch2/sferrone/simulations/Orbits/pouliasis2017pii-GCNBody/"
    myOrbit=h5py.File(basepath+"NGC2808-orbits.hdf5","r")
    # open the orbit of the target
    pal5Orbit=h5py.File(basepath+"Pal5-orbits.hdf5","r")
    
    Xs,Ys,magA=FOORB.getAccelArrayFromForceFile(forceFile,perturber)
    TheTimeIndex,theShiftIndx,dX,dV=FOORB.getImpact(forceFile,myOrbit,
                                              pal5Orbit,montecarlokey,
                                              Xs,Ys,magA)
    
    print("The time index is: ",TheTimeIndex)
    print("The shift index is: ",theShiftIndx)
    print("The displacement is: ",np.linalg.norm(dX))
    print("The velocity difference is: ",np.linalg.norm(dV))


    # close the files
    forceFile.close()
    myOrbit.close()
    pal5Orbit.close()
    
    
    
def getAccelArrayFromForceFile(forceFile:h5py._hl.files.File, perturber:str):
    """
    Get the acceleration arrays from a force file for a specific perturber.

    Parameters:
    - forceFile (h5py._hl.files.File): The force file containing the acceleration data.
    - perturber (str): The name of the perturber to extract the acceleration data for.

    Returns:
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
    xs,ys=xs.to(u.Myr).value,ys.to(u.Myr).value
    Xs,Ys=np.meshgrid(xs,ys)
    # get the accelerations
    magA=np.sqrt(forceFile["AXs"][:,i,:]**2 + forceFile["AYs"][:,i,:]**2 + forceFile["AZs"][:,i,:]**2)
    Xs,Ys=Xs.T,Ys.T
    return Xs,Ys,magA

def getImpact(forceFile:h5py._hl.files.File, myOrbit:h5py._hl.files.File, 
              pal5Orbit:h5py._hl.files.File, montecarlokey: str, 
              Xs:np.ndarray, Ys:np.ndarray, magA:np.ndarray,mintime:np.int64=-500):
    """
    Calculate the impact parameters and relative distances and velocities between a perturber and a target orbit.

    Parameters:
        forceFile (numpy.ndarray): Array containing the perturber data.
        myOrbit (dict): Dictionary containing the target orbit data.
        pal5Orbit (dict): Dictionary containing the host orbit data.
        montecarlokey: Key for accessing the specific orbit data.
        perturber (str): Name of the perturber.

    Returns:
        tuple: A tuple containing the following values:
            - TheTimeIndex (int): Time index for the impact.
            - theShiftIndx (int): shift in time ahead of impact time to recover the poisition along the orbit
            - dX (numpy.ndarray): Displacement, perturber - nearest point along orbit
            - dV (numpy.ndarray): difference in velocity, perturber - nearest point along orbit
    """
    # shortening the time axis
    xaxismask=Xs[:,0] > mintime
    Xs,Ys,C=Xs[xaxismask,:],Ys[xaxismask,:],magA[xaxismask,:]
    temp=~xaxismask
    undoshorten=temp.sum()
    
    # Get the maximum value
    peakX,peakY=np.where(C==np.max(C))

    #  the indicies for the impact time
    TheTimeIndex=forceFile["comTimeIndexes"][undoshorten+peakX[0]]
    theShiftIndx=forceFile["samplingIndexes"][peakY[0]]    
    
    # extract the perturber at this point
    xp=myOrbit[montecarlokey]['xt'][TheTimeIndex]
    yp=myOrbit[montecarlokey]['yt'][TheTimeIndex]
    zp=myOrbit[montecarlokey]['zt'][TheTimeIndex]
    vxp=myOrbit[montecarlokey]['vxt'][TheTimeIndex]
    vyp=myOrbit[montecarlokey]['vyt'][TheTimeIndex]
    vzp=myOrbit[montecarlokey]['vzt'][TheTimeIndex]
    
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
    
    return TheTimeIndex,theShiftIndx,dX,dV





    
    
    
if __name__=="__main__":
    montecarlonumber=input("Enter the montecarlo number: ")
    mcarlokey="monte-carlo-"+str(montecarlonumber).zfill(3)
    storeNGC2808impactInformation(mcarlokey)