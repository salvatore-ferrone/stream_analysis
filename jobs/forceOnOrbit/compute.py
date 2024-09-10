"""
Approximate the stream from a globular cluster as the orbit
Compute the force from all other globular clusters at the samepling points
Write the output to a file
"""

import tstrippy
import gcs
from gcs import path_handler as ph
import numpy as np 
import stream_analysis as sa 
import os 



def main(dataparams,hyperparams):
    GCname,MWpotential,montecarlokey = dataparams
    side_length_factor,ntauskip,ntskip = hyperparams

    G = tstrippy.Parsers.potential_parameters.G

    outfilename = ph.ForceOnOrbit(GCname, MWpotential, montecarlokey)
    if os.path.exists(outfilename):
        print(outfilename, "already exists. \n Delete if you want to recompute.")
        return 
    
    ###### LOAD 
    hostorbitfilename           =   ph.GC_orbits(MWpotential, GCname)
    t,x,y,z,vx,vy,vz            =   gcs.extractors.GCOrbits.extract_whole_orbit(hostorbitfilename, montecarlokey) 
    GCnames                     =   get_perturber_names(GCname)
    ts,xs,ys,zs,vxs,vys,vzs     =   get_perturbing_orbits(GCnames, MWpotential, montecarlokey)
    Masses,rs                   =   get_masses_and_radii(GCnames,montecarlokey)


    ####### DOWN SAMPLE ORBIT TO APPROX STREAM
    tdyn                        = sa.forceOnOrbit.median_dynamical_time(x,y,z,vx,vy,vz)
    index_width_dynamical_time  = sa.forceOnOrbit.index_width_of_dynamical_time(t,tdyn,)
    tau_indexing                = sa.forceOnOrbit.tau_index_down_sampling(index_width_dynamical_time,ntauskip,side_length_factor)
    time_indexing               = sa.forceOnOrbit.time_index_down_sampling(ts.shape[0],tau_indexing[0],ntskip)
    # here are the output times
    tau     =   t[tau_indexing + tau_indexing[0]] - t[tau_indexing[0]]
    time    =   ts[time_indexing]

    # initialize output
    ntime,ntau,nGCs     =   len(time),len(tau),len(GCnames)
    ax,ay,az,magnitude  =   sa.forceOnOrbit.initialize_outputs(nGCs,ntime,ntau)

    ########## perform the big computation 
    ax,ay,az,magnitude  =   sa.forceOnOrbit.compute_force_from_all_gcs(G,Masses,[xs,ys,zs],[x,y,z],[time_indexing,tau_indexing],[ax,ay,az,magnitude])

    ## pack the outputs 
    attrs = GCname,MWpotential,montecarlokey,hostorbitfilename,tdyn,side_length_factor,ntauskip,ntskip
    xycoords = time,tau
    xyindexing = time_indexing,tau_indexing
    acceleration = ax,ay,az
    magnitude=magnitude
    perturbers = GCnames,Masses,rs
    sa.forceOnOrbit.write_file(outfilename,attrs,xycoords,xyindexing,acceleration,magnitude,perturbers)
    print(outfilename, "written.")


##### Convience functions
def get_perturber_names(GCname):
    GCdata=tstrippy.Parsers.baumgardtMWGCs()
    myGCnames = GCdata.data['Cluster']
    GCnames = [str(myGCnames[i]) for i in range(len(myGCnames))]
    GCnames.pop(GCnames.index(GCname))
    return GCnames

def get_perturbing_orbits(GCnames, MWpotential, montecarlokey):
    ts,xs,ys,zs,vxs,vys,vzs=gcs.extractors.GCOrbits.extract_orbits_from_all_GCS(GCnames, MWpotential, montecarlokey) # type: ignore
    today_index = np.argmin(np.abs(ts))
    ts=ts[0:today_index]
    xs,ys,zs=xs[:,0:today_index],ys[:,0:today_index],zs[:,0:today_index]
    vxs,vys,vzs=vxs[:,0:today_index],vys[:,0:today_index],vzs[:,0:today_index]
    return ts,xs,ys,zs,vxs,vys,vzs

def get_masses_and_radii(GCnames,montecarlokey):
    Masses,rh_mes,_,_,_,_,_,_=gcs.extractors.MonteCarloObservables.extract_all_GC_observables(GCnames,montecarlokey)
    rs = np.array([gcs.misc.half_mass_to_plummer(x).value for x in rh_mes])
    Masses = np.array([x.value for x in Masses])    
    return Masses,rs



if __name__=="__main__":
    #dataparams
    GCname = "Pal5"
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey = "monte-carlo-000"
    # hyper params
    side_length_factor  =   2
    ntauskip            =   10
    ntskip              =   10
    dataparams = GCname,MWpotential,montecarlokey
    hyperparams = side_length_factor,ntauskip,ntskip
    main(dataparams=dataparams,hyperparams=hyperparams)