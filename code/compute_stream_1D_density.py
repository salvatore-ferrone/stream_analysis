import h5py
import numpy as np 
import sys 
import os
import datetime
from astropy import units as u # type: ignore
import StreamOrbitCoords as SOC
import data_extractors as DE
import data_writing as DW # type: ignore
import filters 




def main(mcarlo,startindex=40):
    
    #### Base parameters
    GCname="Pal5"
    nDynTimes=3
    mcarlokey="monte-carlo-"+str(mcarlo).zfill(3)
    NP = int(1e5)
    ### path handling
    path_orbit="/scratch2/sferrone/simulations/Orbits/pouliasis2017pii-GCNBody/Pal5-orbits.hdf5"
    path_stream_orbit = "/scratch2/sferrone/simulations/StreamOrbits/pouliasis2017pii-Pal5-suspects/Pal5/100000/Pal5-"+mcarlokey+"-StreamOrbit.hdf5"
    outpath="/scratch2/sferrone/intermediate-products/stream_density_profiles/" + GCname + "/"
    outname = GCname+"_time_profile_density_"+mcarlokey+".hdf5"
    os.makedirs(outpath, exist_ok=True)
    outfilename=outpath+outname
    
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost = DE.get_orbit(path_orbit,mcarlokey)
    with h5py.File(path_stream_orbit, 'r') as stream:
        t_stamps=DE.extract_time_steps_from_stream_orbit(stream)
    hostorbit = thost,xhost,yhost,zhost,vxhost,vyhost,vzhost
    Nstamps = len(t_stamps)
    out_array = np.zeros((Nstamps,NP))
    DW.initialize_stream_tau_output(GCname,mcarlokey,path_orbit,path_stream_orbit,\
        t_stamps,out_array,outpath=outpath,outname=outname)

    starttime=datetime.datetime.now()

    end_index = Nstamps
    for i in range(startindex,end_index):
        obtain_tau(i,path_stream_orbit,hostorbit,t_stamps,nDynTimes,outfilename)
        if i == startindex:
            print("Done with ",i)
        if i%100==0:
            print("Done with ",i)
    endtime=datetime.datetime.now()
    print("Time taken: ",endtime-starttime)


#####################################################
####################### LOOPS #######################
#####################################################
def obtain_tau(i, path_stream_orbit, hostorbit, t_stamps, period):
    """
    Given a snapshot index, this function projects the stream onto the host orbit 
    and calculates tau. Tau is the difference in time coordinate of a position on 
    the orbit and the current time.

    Parameters
    ----------
    i : int
        The snapshot index.
    path_stream_orbit : str
        The path to the stream orbit file.
    hostorbit : tuple
        A tuple containing the host orbit data (thost, xhost, yhost, zhost, vxhost, vyhost, vzhost).
    t_stamps : array_like
        An array of time stamps.
    period : float
        The period for filtering the orbit.

    Returns
    -------
    tau : ndarray
        The time difference from ahead/behind for each position in the stream.

    """
    # extract stream positions and orbit 
    xp, yp, zp, _, _, _ = DE.extract_stream_from_path(i, path_stream_orbit)
    thost, xhost, yhost, zhost, vxhost, vyhost, vzhost = hostorbit
    orbit = xhost, yhost, zhost, vxhost, vyhost, vzhost
    # filter orbit by dynamical time
    t_index = np.argmin(np.abs(thost - t_stamps[i].value))
    tOrb, xOrb, yOrb, zOrb, _, _, _ = filters.orbit_by_fixed_time(thost, orbit, t_index, period)
    # get closest orbital index of particles to stream
    indexes = SOC.get_closests_orbital_index(xOrb, yOrb, zOrb, xp, yp, zp)
    # calculate time difference from ahead/behind
    tau = tOrb[indexes] - t_stamps[i].value
    return tau
    

##############################################################
######################### COMPUTATIONS #######################
##############################################################
def fourier_strongest_period(t,xt,yt,zt):
    r = np.sqrt(xt**2 + yt**2 + zt**2)
    # Compute the time step
    dt = np.mean(np.diff(t))

    # Compute the Fourier Transform
    R = np.fft.fft(r)

    # Compute the frequencies
    frequencies = np.fft.fftfreq(r.size, dt)

    # Select the half of the arrays that corresponds to the positive frequencies
    positive_frequencies = frequencies[:frequencies.size // 2]
    positive_R = R[:R.size // 2]

    # Compute the magnitude spectrum for the positive frequencies
    magnitude = np.abs(positive_R)

    maxfreq_dex=np.argmax(magnitude[1:]) + 1
    # Find the frequency that corresponds to the maximum value in the magnitude spectrum
    strongest_frequency = positive_frequencies[maxfreq_dex]


    period = 1/strongest_frequency
    return period 





 

if __name__=="__main__":
    mcarlo = int(sys.argv[1])
    main(mcarlo)