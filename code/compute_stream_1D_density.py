import h5py
import numpy as np 
import sys 
import os
import datetime
from astropy import units as u
sys.path.append('/obs/sferrone/gc-tidal-loss/code')
import StreamOrbitCoords as SOC
sys.path.append("/obs/sferrone/mini-reports/GapPredictor/code")
import data_extractors as DE

import multiprocessing as mp




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
    
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost = get_host_orbit(path_orbit,mcarlokey)
    t_stamps=get_stream_time_stamps(path_stream_orbit)
    hostorbit = thost,xhost,yhost,zhost,vxhost,vyhost,vzhost
    Nstamps = len(t_stamps)
    out_array = np.zeros((Nstamps,NP))
    initialize_output_file(GCname,mcarlokey,path_orbit,path_stream_orbit,\
        t_stamps,out_array,outpath=outpath,outname=outname)

    starttime=datetime.datetime.now()

    end_index = Nstamps
    for i in range(startindex,end_index):
        worker(i,path_stream_orbit,hostorbit,t_stamps,nDynTimes,outfilename)
        if i == startindex:
            print("Done with ",i)
        if i%100==0:
            print("Done with ",i)
    endtime=datetime.datetime.now()
    print("Time taken: ",endtime-starttime)
    
    #### IF YOU WANNA DO THIS IN PARALLEL, 
    ####  THEN YOU NEED TO USE TEMP FILES TO FIX THE I/O
    # get the number of processors available
    # num_processes = 4
    # with mp.Pool(num_processes) as pool:
        # pool.starmap(worker, [(i, path_stream_orbit, hostorbit, t_stamps, nDynTimes, outfilename) for i in range(startindex, Nstamps)])
    # endtime=datetime.datetime.now()
    # print("Time taken: ",endtime-starttime)
    
    

def worker(i,path_stream_orbit,hostorbit,t_stamps,period,outfilename):
    # extract stream positions and orbit 
    xp,yp,zp,vxp,vyp,vzp = extract_stream(i,path_stream_orbit)
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost = hostorbit
    # filter orbit by dynamical time
    t_index = np.argmin(np.abs(thost-t_stamps[i].value))
    tOrb,xOrb,yOrb,zOrb,vxOrb,vyOrb,vzOrb=\
        filter_orbit_by_period(\
        thost,xhost,yhost,zhost,vxhost,vyhost,vzhost,\
        t_index,period)
    # get closest orbital index of particles to stream
    indexes=SOC.get_closests_orbital_index(xOrb,yOrb,zOrb,xp,yp,zp)
    # calculate time difference from ahead/behind
    DTS = tOrb[indexes] - t_stamps[i].value
    # append data file
    with h5py.File(outfilename, 'a') as myoutfile:
        myoutfile["density_profile"][i]=DTS
    

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


def filter_orbit_by_period(t,xt,yt,zt,vxt,vyt,vzt,current_index,filter_time):
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


##############################################################
##################### DATA I/O FUNCTIONS #####################
##############################################################
def initialize_output_file(GCname,mcarlokey,path_orbit,path_stream_orbit,\
    time_stamps,out_array,\
    outpath,
    outname):


    print(outpath+outname)
    os.makedirs(outpath, exist_ok=True)
    with h5py.File(outpath+outname, 'a') as myoutfile:
        myoutfile.attrs['author'] = 'sferrone'
        myoutfile.attrs['email'] = 'salvatore.ferrone@uniroma1.it'
        myoutfile.attrs['creation_date'] = str(np.datetime64('now'))
        myoutfile.attrs['note'] = 'This file contains a 1D stream density profile projected onto the Orbit. The unit is in time, i.e. how far ahead of behind the particle is. '
        myoutfile.attrs['unitT'] = str(time_stamps.unit)
        myoutfile.attrs['path_orbit'] = path_orbit
        myoutfile.attrs['GCname'] = GCname
        myoutfile.attrs['mcarlokey'] = mcarlokey
        myoutfile.attrs['path_stream_orbit'] = path_stream_orbit
        if "time_stamps" not in myoutfile:
            myoutfile.create_dataset("time_stamps", data=time_stamps.value)
        if "density_profile" not in myoutfile:
            myoutfile.create_dataset("density_profile", data=out_array)

def get_host_orbit(path_orbit,mcarlokey):
    with h5py.File(path_orbit,'r') as fp:
        thost=fp[mcarlokey]['t'][:]
        xhost=fp[mcarlokey]['xt'][:]
        yhost=fp[mcarlokey]['yt'][:]
        zhost=fp[mcarlokey]['zt'][:]
        vxhost=fp[mcarlokey]['vxt'][:]
        vyhost=fp[mcarlokey]['vyt'][:]
        vzhost=fp[mcarlokey]['vzt'][:]
    return thost,xhost,yhost,zhost,vxhost,vyhost,vzhost

def get_stream_time_stamps(path_stream_orbit):
    with h5py.File(path_stream_orbit, 'r') as stream:
        timestamps=DE.extract_time_steps_from_stream_orbit(stream)
    return timestamps
    
def extract_stream(i,path_stream_orbit):
    with h5py.File(path_stream_orbit, 'r') as stream:
        xp=stream['timestamps'][str(i)][0,:]
        yp=stream['timestamps'][str(i)][1,:]
        zp=stream['timestamps'][str(i)][2,:]
        vxp=stream['timestamps'][str(i)][3,:]
        vyp=stream['timestamps'][str(i)][4,:]
        vzp=stream['timestamps'][str(i)][5,:]
    return xp,yp,zp,vxp,vyp,vzp

 

if __name__=="__main__":
    mcarlo = int(sys.argv[1])
    main(mcarlo)