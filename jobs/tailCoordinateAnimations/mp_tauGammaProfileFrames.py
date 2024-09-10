"""
The secrete sauce is to extract the stream particles within each process!
the copying really must've added a lot of overhead. 

"""

import gcs
from gcs import path_handler as ph
import os 
import h5py
import multiprocessing as mp
import tauGammaProfileFrames as TGPF #type: ignore
import datetime

def main(dataparams):

    GCname,MWpotential,montecarlokey,NP,internal_dynamics=dataparams
    outdir=ph.paths['temporary'] + "stream_analysis/tauGammaProfileFrames/" + MWpotential  + GCname + "/" + montecarlokey  + "/"
    os.makedirs(outdir,exist_ok=True)  

    # get the host horbit 
    orbfname=ph.GC_orbits(MWpotential,GCname)
    fullhostorbit=gcs.extractors.GCOrbits.extract_whole_orbit(orbfname,montecarlokey)  

    # get the stream at each snap shot
    streamFname=ph.StreamSnapShots(GCname=GCname,NP=NP,potential_env=MWpotential,internal_dynamics=internal_dynamics,montecarlokey=montecarlokey)
    with h5py.File(streamFname,"r") as myfile:
        timestamps=myfile['time_stamps'][:]
    nFrames = len(timestamps)

    starttime=datetime.datetime.now()


    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print("ncpu:",ncpu)
    results = []
    for i in range(42, nFrames):
        result = pool.apply_async(TGPF.make_frame, args=(outdir, i, timestamps, fullhostorbit, streamFname))
    pool.close()
    pool.join()
    
    # Optionally, handle results or check for errors
    for result in results:
        try:
            result.get()  # This will raise any exceptions that occurred in the worker process
        except Exception as e:
            print(f"Error in worker process: {e}")

    endtime=datetime.datetime.now()
    print("Time taken: ",endtime-starttime)





if __name__=="__main__":
    # data params
    GCname = "Pal5"
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey = "monte-carlo-000"
    NP = int(1e5)
    internal_dynamics = "isotropic-plummer"

    dataparams = (GCname,MWpotential,montecarlokey,NP,internal_dynamics)
    main(dataparams)

    