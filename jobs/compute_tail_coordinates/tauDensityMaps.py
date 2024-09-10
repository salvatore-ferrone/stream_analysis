"""
The tau-data are the 1D profiles of the stream projected onto the orbit. 

At a given simulation time, indexed with *j*, the stream particle *i* is projected onto the orbit.

tau is thus:
    tau = t[i] - t[j]

Where 
    t[j] is the current simulation time and 
    t[i] is the time at which the host globular cluster will be at the same position was projected to the stream particle *i*.

so tau is how far ahead or behind the stream particle is from the host globular cluster in time. 

This code is done in multi processing because it is computational expensive. We have to project 1e5 particles for 5e3 snapshots. which is:
    5e3*1e5 = 5e8 operations.

    
At the time of writing, it takes about 1 second per snapshot. So the estimated computational time is:
    1s * 5e3 / NCPU ~ 5 minutes (for 1)

They will be sasved in 
PROJSIMULATIONS/tauDensityMaps/MWpotential/GCname/NP/internal_dynamics/GCname
"""

import numpy as np
import stream_analysis as sa
import os
import h5py
import multiprocessing as mp
import datetime
import gcs
from gcs import path_handler as ph


def main(dataparams):
    GCname,MWpotential,montecarlokey,NP,internal_dynamics=dataparams

    ndyn=2
    start_index=10



    outdir,outfname = make_outpath_and_filename(GCname,MWpotential,montecarlokey,NP,internal_dynamics)
    os.makedirs(outdir,exist_ok=True)
    # check if file already exists
    if os.path.isfile(outdir+outfname):
        print("Data already exists at: ",outdir+outfname)
        return None



    fname=ph.StreamSnapShots(GCname=GCname,NP=NP,potential_env=MWpotential,internal_dynamics=internal_dynamics,montecarlokey=montecarlokey)

    if os.path.isfile(fname):
        with h5py.File(fname, 'r') as f:
            time_stamps=f['time_stamps'][:] 
    else:
        print("No data found at: ",fname)
        return None
    


    tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost        =   gcs.extractors.GCOrbits.extract_whole_orbit(ph.GC_orbits(MWpotential=MWpotential,GCname=GCname),montecarlokey=montecarlokey)
    hostorbit=(tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost)


    # make bins for tau
    rH          =   np.sqrt(xHost**2+yHost**2+zHost**2)
    vH          =   np.sqrt(vxHost**2+vyHost**2+vzHost**2)
    tdynamical  =   np.median(rH/vH)
    tau_edges   =   np.linspace(-ndyn*tdynamical,ndyn*tdynamical,int(np.ceil(np.sqrt(NP))))
    tau_centers =   0.5*(tau_edges[1:]+tau_edges[:-1])
    tau_counts  =   np.zeros((len(time_stamps),len(tau_centers)))    


    # do this in parallel because its slow
    starttime=datetime.datetime.now()
    ncpu = mp.cpu_count()
    pool = mp.Pool(processes=ncpu)
    results = [pool.apply_async(tau_counts_loop, args=(i,time_stamps,fname,hostorbit,ndyn,tau_edges)) for i in range(start_index,len(time_stamps))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    endtime=datetime.datetime.now()

    print("Time taken: ",endtime-starttime)

    # elimiate the ends because points bunch up due to finite sample size 
    for i in range(start_index,len(time_stamps)):
        tau_counts[i]=output[i-start_index]
        tau_counts[i][0]=0
        tau_counts[i][-1]=0


    # put all the data at the center for the first few snapshots that we skip
    lentau=len(tau_centers)
    tau_counts[0:start_index,lentau//2]=NP

    
    with h5py.File(outdir+outfname, 'w') as f:
        f.create_dataset('tau_centers',data=tau_centers)
        f.create_dataset('tau_counts',data=tau_counts)
        f.create_dataset('time_stamps',data=time_stamps)

    print("Data saved to: ",outdir+outfname)


def make_outpath_and_filename(GCname,MWpotential,montecarlokey,NP,internal_dynamics):
    outdir=ph.paths['simulations'] + "tauDensityMaps/" + MWpotential + "/" + GCname + "/" + str(NP) + "/" + internal_dynamics+"/"
    outfname = GCname+"-"+montecarlokey+"-tauDensityMap.h5"
    return outdir,outfname

def extract_snapshot(i,fname):
    """
    Needs to be it's own function because extracting outside of the loop means copying data and made the computation really slow
    """
    with h5py.File(fname, 'r') as f:
        xp,yp,zp,vxp,vyp,vzp=f['StreamSnapShots'][str(i)][:]
    return xp,yp,zp,vxp,vyp,vzp

def tau_counts_loop(i,time_stamps,fname,hostorbit,ndyn,tau_edges):
    tsampling,xH,yH,zH,vxH,vyH,vzH=hostorbit
    current_time = time_stamps[i] #type: ignore
    xp,yp,zp,_,_,_=extract_snapshot(i,fname)
    TORB, XORB, YORB, ZORB, _, _, _=sa.tailCoordinates.filter_orbit_by_dynamical_time(\
        tsampling,xH,yH,zH,vxH,vyH,vzH,current_time,ndyn)
    indexes = sa.tailCoordinates.get_closests_orbital_index(XORB, YORB, ZORB,xp,yp,zp)
    tau = TORB[indexes] - current_time
    counts,_=np.histogram(tau,bins=tau_edges)
    return counts




if __name__=="__main__":
    internal_dynamics   =   "isotropic-plummer"
    montecarlokey       =   "monte-carlo-001"
    NP                  =   int(1e5)
    MWpotential         =   "pouliasis2017pii-GCNBody"
    GCname              =   "Pal5"

    
    for i in range(50):
        montecarlokey="monte-carlo-"+str(i).zfill(3)
        dataparams = (GCname,MWpotential,montecarlokey,NP,internal_dynamics)
        main(dataparams)