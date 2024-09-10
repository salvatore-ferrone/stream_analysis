import stream_analysis as sa
import gcs 
from gcs import path_handler as ph
import os
import h5py
import pandas as pd
import numpy as np 
from scipy import optimize
import datetime



def main(dataparams,hyperparams):

    GCname,NP,MWpotential,montecarlokey,internal_dynamics=dataparams
    targetnumber,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder=hyperparams

    _,fnameSuspects,_ =   load_input_paths(dataparams)
    suspect,targetTime,targetTau                =   pick_target(fnameSuspects,targetnumber)

    outfilename=ph.ImpactGeometry(GCname,MWpotential,montecarlokey,suspect,targetnumber)

    if os.path.exists(outfilename):
        print(outfilename+" already exists. Exiting")
        return
    else:
        print("Computing "+outfilename)
    time_stamps,time_indexes,coefficients_perturber,coefficients_stream,tau,_,particle_filter = load_and_parameterize_trajectory_and_stream(dataparams,hyperparams)

    tauWidth    =   tau[0].max()-tau[0].min()
    t_bounds    =   time_stamps[time_indexes[0]],time_stamps[time_indexes[-1]]
    tau_bounds  =   [targetTau-tauWidth,targetTau+tauWidth]    


    # do the tine tunning
    minimization_method="L-BFGS-B"
    p0      = [targetTime,targetTau]
    args    = (coefficients_stream,coefficients_perturber,time_stamps[time_indexes])
    bounds  = [t_bounds,tau_bounds]
    results=optimize.minimize(sa.parametric_stream_fitting.objective_stream_perturber,p0,args=args,method=minimization_method,bounds=bounds)    
    T,tau = results.x[0],results.x[1]

    # COMPUTE THE ERKAL GEOMETRY
    impact_parameter,w_par,w_per,alpha=sa.parametric_stream_fitting.get_full_impact_geometry_from_parametrization(T,tau,coefficients_stream,coefficients_perturber,time_stamps[time_indexes])
    # GRAB OTHER SUPPLMENTAL INFO 
    Mass,rh_m,_,_,_,_,_,_=gcs.extractors.MonteCarloObservables.extract_GC_observables(ph.MonteCarloObservables(suspect),montecarlokey)
    profile_density=extract_density_at_impact_point(GCname,MWpotential,montecarlokey,NP,internal_dynamics,T,tau)    
    Mass=Mass.value
    rplum=gcs.misc.half_mass_to_plummer(rh_m.value)

    # PREPARE THE OUTPUT
    geometry={"impact_parameter":impact_parameter,"w_par":w_par,"w_per":w_per,"alpha":alpha,"T":T,"tau":tau,"Mass":Mass,"rplum":rplum,"profile_density":profile_density}
    dataparams={"GCname":GCname,"NP":NP,"MWpotential":MWpotential,"montecarlokey":montecarlokey,"internal_dynamics":internal_dynamics}
    hyperparams={"targetnumber":targetnumber,"n_adjacent_points":n_adjacent_points,"nDynTimes":nDynTimes,"width_factor":width_factor,"streampolyorder":streampolyorder,"perturberPolyOrder":perturberPolyOrder}    

    with h5py.File(outfilename,"w") as myoutfile:
        myoutfile.create_group("geometry")
        for key in geometry.keys():
            myoutfile["geometry"].create_dataset(key,data=geometry[key])

        myoutfile.create_group("dataparams")
        for key in dataparams.keys():
            myoutfile["dataparams"].create_dataset(key,data=dataparams[key])

        myoutfile.create_group("hyperparams")
        for key in hyperparams.keys():
            myoutfile["hyperparams"].create_dataset(key,data=hyperparams[key])

        myoutfile.create_dataset("time_stamps",data=time_stamps)
        myoutfile.create_dataset("time_indexes",data=time_indexes)
        myoutfile.create_dataset("coefficients_stream",data=coefficients_stream)
        myoutfile.create_dataset("coefficients_perturber",data=coefficients_perturber)
        myoutfile.create_dataset("particle_filter",data=particle_filter)
        myoutfile.create_dataset("tauBounds",data=tau_bounds)

        myoutfile.create_dataset("suspect",data=suspect)
        
        myoutfile.attrs["date"]=str(datetime.datetime.now())
        myoutfile.attrs["author"]="Salvatore Ferrone"
        myoutfile.attrs["email"]="salvatore.ferrone@uniroma1.it"

    print(outfilename+" written")
    return None

def load_and_parameterize_trajectory_and_stream(dataparams,hyperparams):
    GCname,NP,MWpotential,montecarlokey,internal_dynamics=dataparams
    targetnumber,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder=hyperparams

    fnameSnapShots,fnameSuspects,fnameHostOrbit=load_input_paths(dataparams)

    suspect,targetTime,targetTau=pick_target(fnameSuspects,targetnumber)

    time_stamps     =   get_all_stream_time_stamps(fnameSnapShots)
    time_index      =   np.argmin(np.abs(time_stamps-targetTime))
    time_indexes    =   np.arange(time_index-n_adjacent_points,time_index+n_adjacent_points+1)

    fnamePert   =   ph.GC_orbits(MWpotential,suspect)

    hostOrbit   =   gcs.extractors.GCOrbits.extract_whole_orbit(fnameHostOrbit,montecarlokey=montecarlokey)

    tau,galactocentric_stream,particle_filter   =   load_tau_and_stream(fnameSnapShots,time_stamps,time_indexes,hostOrbit,targetTau,nDynTimes=nDynTimes,width_factor=width_factor)

    coefficients_perturber =   get_perturber_orbit_coeffs(fnamePert=fnamePert,montecarlokey=montecarlokey,tmin=time_stamps[time_indexes[0]],tmax=time_stamps[time_indexes[-1]],perturberPolyOrder=perturberPolyOrder)    

    n_stamps=len(time_indexes)
    coeffs = [ ]
    for i in range(n_stamps):
        result=fit_3D_parametric(tau[i],galactocentric_stream[i,0:3,:],order=streampolyorder,minimization_method='Nelder-Mead')
        coeffs.append(result)
    coefficients_stream=np.array(coeffs)

    return time_stamps,time_indexes,coefficients_perturber,coefficients_stream,tau,galactocentric_stream,particle_filter

def fit_3D_parametric(taus,galcentric_stream,order = 3,minimization_method =   'Nelder-Mead',):
    """
    Dis funzione serve a fittare una parabola 3D a una stream.
    taus: array di parametri per la stream
    galcentric_stream: array di coordinate galattiche della stream
    minimization_method: metodo di minimizzazione. Default: 'Nelder-Mead'
    RETURNS: 
        i coefficienti della parabola fittata (x0,x1,x2,y0,y1,y2,z0,z1,z2) 
            dove i termini corrispondono a x = x0 + x1*tau + x2*tau^2
    """
    assert len(galcentric_stream.shape)==2, "galcentric_stream must have 2 dimensions: (3-coordinates,NP)"
    assert len(taus)==galcentric_stream.shape[1], "Must be as many parametric coordinates as particles"
    assert galcentric_stream.shape[0]==3, "galcentric_stream must have only 3 coordinates: x, y, z"
    objective = sa.parametric_stream_fitting.objective_3D_parametric_line
    args = (taus,galcentric_stream)
    initial_guess = sa.parametric_stream_fitting.initial_guess_3D_parametric_eq(taus,galcentric_stream,order=order)
    results = optimize.minimize(objective,initial_guess,args=args,method=minimization_method)
    if not results.success:
        print("Fit failed. Investigate")
    return results.x


def parameterize_stream_tau(hostOrbit,galactocentric_stream,my_time_stamps,nDynTimes=2):
    # allocate outputs
    assert len(galactocentric_stream.shape)==3, "galactocentric_stream must have 3 dimensions: (n_time_stanps,6-coordinates,NP)"
    assert len(my_time_stamps)==galactocentric_stream.shape[0], "time_stamps and galactocentric_stream must have the same length"
    assert len(hostOrbit)==7, "hostOrbit must have 7 elements: t,x,y,z,vx,vy,vz"
    tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost=hostOrbit
    NP = galactocentric_stream.shape[2]
    n_stamps = galactocentric_stream.shape[0]
    tau = np.zeros((n_stamps,NP))
    for i in range(n_stamps):
        current_time    = my_time_stamps[i]
        xp,yp,zp,_,_,_  = galactocentric_stream[i]
        tOrb,xOrb,yOrb,zOrb,_,_,_ = sa.tailCoordinates.filter_orbit_by_dynamical_time(tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost,current_time,nDynTimes=nDynTimes)
        indexes=sa.tailCoordinates.get_closests_orbital_index(xORB=xOrb,yORB=yOrb,zORB=zOrb,xp=xp,yp=yp,zp=zp)
        tau[i] = tOrb[indexes] - current_time
    return tau


def filter_particles(hostOrbit,stream,current_time,targetTau,nDynTimes=2,width_factor = 10):
    """
    Get particles that are on the target side of the stream, and that are not at the end of the stream
    """
    threshold = 0.5
    xp,yp,zp,vx,vy,vz = stream
    tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost=hostOrbit
    tOrb,xOrb,yOrb,zOrb,vxOrb,vyOrb,vzOrb = sa.tailCoordinates.filter_orbit_by_dynamical_time(tHost,xHost,yHost,zHost,vxHost,vyHost,vzHost,current_time,nDynTimes=nDynTimes)
    # indexes=sa.tailCoordinates.get_closests_orbital_index(xORB=xOrb,yORB=yOrb,zORB=zOrb,xp=xp,yp=yp,zp=zp)
    xprimeP,yprimeP,zprimeP,vxprimeP,vyprimeP,vzprimeP,indexes=sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vx,vy,vz,tOrb,xOrb,yOrb,zOrb,vxOrb,vyOrb,vzOrb,t0=current_time)
    my_tau = tOrb[indexes] - current_time

    # only grab the correct side
    condA = np.sign(targetTau)*my_tau>0
    # reject the tips
    condB = ~(indexes==0)
    condC = ~(indexes==np.max(indexes))
    condD = np.logical_and(condA,condB)
    condE  = np.logical_and(condD,condC)

    # now clip again to get an even smaller portion
    tau_ = my_tau[condE]
    tau_width = tau_.max()-tau_.min()

    under = targetTau-tau_width/width_factor
    over = targetTau+tau_width/width_factor

    condF = my_tau > under
    condG = my_tau < over
    condH = np.logical_and(condF,condG)
    condI = np.logical_and(condE,condH)

    # reject all particles above a certain distance

    R = np.sqrt(yprimeP**2+zprimeP**2)

    condJ = R<threshold

    cond = np.logical_and(condI,condJ)

    return cond


def load_input_paths(dataparams):
    GCname,NP,MWpotential,montecarlokey,internal_dynamics=dataparams
    fnameSnapShots  =   ph.StreamSnapShots(GCname=GCname, NP=NP, potential_env=MWpotential, montecarlokey=montecarlokey, internal_dynamics=internal_dynamics)
    fnameSuspects   =   ph.PerturberSuspects(GCname,MWpotential,montecarlokey)
    fnameHostOrbit  =   ph.GC_orbits(MWpotential,GCname)
    return fnameSnapShots,fnameSuspects,fnameHostOrbit


def pick_target(fnameSuspects,targetnumber):
    suspects = pd.read_csv(fnameSuspects)
    n_targets = len(suspects)
    assert targetnumber<n_targets, "There are only {:d} targets, and you want number {:d}".format(n_targets,targetnumber)

    suspect = suspects["suspects"].values[targetnumber]
    targetTime = suspects['time'].values[targetnumber]
    targetTau = suspects['tau'].values[targetnumber]
    return suspect,targetTime,targetTau


def get_all_stream_time_stamps(fnameSnapShots):
    with h5py.File(fnameSnapShots,'r') as snapsShots:
        time_stamps = snapsShots['time_stamps'][()]
    return time_stamps


# extract the stream, only the good particles
def extract_stream_snapshots(fnameSnapShots,time_indexes,particle_filter):
    NP_filt = np.sum(particle_filter)
    with h5py.File(fnameSnapShots,'r') as snapsShots:
        n_stamps = len(time_indexes)
        galactocentric_stream = np.zeros((n_stamps,6,NP_filt))
        for i in range(n_stamps):
            goodnessgraceious=snapsShots['StreamSnapShots'][str(time_indexes[i])][()]
            galactocentric_stream[i] = goodnessgraceious[:,particle_filter]
    return galactocentric_stream


def get_perturber_orbit_coeffs(fnamePert,montecarlokey,tmin,tmax,perturberPolyOrder):
    perturberTuple  =   gcs.extractors.GCOrbits.extract_whole_orbit(fnamePert,montecarlokey=montecarlokey)
    perturberOrbit  =   sa.parametric_stream_fitting.filter_perturber_orbit(perturberTuple,tmin,tmax)
    xcoeff      =   np.polyfit(perturberOrbit[0],perturberOrbit[1],perturberPolyOrder)
    ycoeff      =   np.polyfit(perturberOrbit[0],perturberOrbit[2],perturberPolyOrder)
    zcoeff      =   np.polyfit(perturberOrbit[0],perturberOrbit[3],perturberPolyOrder)
    coeffs_traj =   np.concatenate((xcoeff,ycoeff,zcoeff))
    return coeffs_traj

def load_tau_and_stream(fnameSnapShots,time_stamps,time_indexes,hostOrbit,targetTau,nDynTimes=2,width_factor=10):
    # build filter to only get the particles that are near the perturber
    middle_index = int(len(time_indexes)/2)
    
    with h5py.File(fnameSnapShots,'r') as snapsShots:
        goodnessgraceious=snapsShots['StreamSnapShots'][str(time_indexes[middle_index])][()]
        particle_filter=filter_particles(hostOrbit,goodnessgraceious,time_stamps[time_indexes[middle_index]],targetTau,nDynTimes=nDynTimes,width_factor=width_factor)    
    
    galactocentric_stream   = extract_stream_snapshots(fnameSnapShots,time_indexes,particle_filter)
    tau                     = parameterize_stream_tau(hostOrbit=hostOrbit,galactocentric_stream=galactocentric_stream,my_time_stamps=time_stamps[time_indexes],nDynTimes=nDynTimes)
    return tau,galactocentric_stream,particle_filter




def extract_density_at_impact_point(GCname,MWpotential,montecarlokey,NP,internal_dynamics,T,tau):
    fileTau=ph.tauDensityMaps(GCname,MWpotential,montecarlokey,NP,internal_dynamics)
    tau_centers,time_stamps,tau_counts = sa.identify_suspects.extract_tau_stream_density(fileTau)
    tau_index = np.argmin(np.abs(tau_centers-tau))
    time_index= np.argmin(np.abs(time_stamps-T))
    profile_density = tau_counts[time_index,tau_index]
    return profile_density



if __name__=="__main__":
    # data params
    GCname = "Pal5"
    NP = int(1e5)
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey="monte-carlo-009"
    internal_dynamics = "isotropic-plummer"

    # hyper params
    targetnumber        = 2
    n_adjacent_points   = 5
    n_stamps            = 2*n_adjacent_points+1
    nDynTimes           = 2
    width_factor        = 10
    streampolyorder     = 2
    perturberPolyOrder  = 2

    dataparams  =   (GCname,NP,MWpotential,montecarlokey,internal_dynamics)
    hyperparams =   (targetnumber,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder)    
    main(dataparams,hyperparams)
    print("done")
