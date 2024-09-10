import tstrippy
import gcs
from gcs import path_handler as ph
import numpy as np 
import os 
import matplotlib.pyplot as plt
import h5py
from matplotlib.cm import ScalarMappable




def main(dataparams):

    start_frame = 0
    GCname,MWpotential,montecarlokey = dataparams

    infile              =   ph.ForceOnOrbit(GCname, MWpotential, montecarlokey, )
    hostorbitfilename   =   ph.GC_orbits(MWpotential, GCname)

    t,x,y,z,vx,vy,vz            =   gcs.extractors.GCOrbits.extract_whole_orbit(hostorbitfilename, montecarlokey) 
    GCnames                     =   get_perturber_names(GCname)
    ts,xs,ys,zs,vxs,vys,vzs     =   get_perturbing_orbits(GCnames, MWpotential, montecarlokey)
    Masses,rs                   =   get_masses_and_radii(GCnames,montecarlokey)    

    ax,ay,az,tau,time,tau_indexing,time_indexing,magnitude = extract_foorb_data(infile)


    rawdata = (xs,ys,zs),(x,y,z),(ax,ay,az)


    # set plot params that don't change
    mycmap = plt.get_cmap('rainbow')
    mynorm = plt.Normalize(vmin=0, vmax=50)
    maxlim=np.max([np.abs(x).max(),np.abs(y).max(),np.abs(z).max()])
    AXIS = {'xlim':(-maxlim,maxlim), 'ylim':(-maxlim,maxlim), 'zlim':(-maxlim,maxlim),"aspect":'equal'}
    qskip = 2 

    outpath=ph.paths["temporary"] + "stream_analysis/FOORB_3D_frames/" 
    os.makedirs(outpath, exist_ok=True)

    plt.style.use('dark_background')
    fig, axis, cbar_axis=initialize_plot()
    add_cbar(cbar_axis, mycmap, mynorm)
    for i in range(start_frame,time_indexing.shape[0]):
        fname       =       "frame-"+ str(i).zfill(4) + ".png"
        axis.cla()
        POS_GCS, Q_POS, COM, STREAM, accels = extract_subsets(i,time_indexing,tau_indexing,rawdata,qskip)
        orbcolors, veccolors                = orbit_vector_colors(mycmap,mynorm,magnitude, i, qskip)
        add_data_to_plot(axis, orbcolors, veccolors, POS_GCS, Q_POS, COM, STREAM, accels, Masses)
        set_axis_properties(axis, AXIS)
        fig.savefig(outpath + fname, dpi=300) # type: ignore
        print("Saved", outpath+fname)




##### PLOT FUNCTIONS
def add_data_to_plot(axis, orbcolors, veccolors, POS_GCS, Q_POS, COM, STREAM, accels, Masses):
    xx,yy,zz = STREAM
    xq,yq,zq = Q_POS
    xCOM,yCOM,zCOM = COM
    axq,ayq,azq = accels
    xxs,yys,zzs = POS_GCS

    axis.scatter(xx, yy, zz, c=orbcolors, alpha=0.5, s=1,)
    axis.quiver(xq, yq, zq, axq, ayq, azq, color=veccolors, alpha=1, length=2, normalize=True)
    axis.scatter(xCOM, yCOM, zCOM, color='green', s=50)
    axis.scatter(xxs, yys, zzs, color='white', s=Masses/np.max(Masses)*100)
    return None


def set_axis_properties(axis, AXIS):
    maxlim=AXIS['xlim'][1]
    axis.set_axis_off()
    axis.plot([-maxlim, maxlim], [0, 0], [0, 0], color='white', alpha=0.8)
    axis.plot([0, 0], [-maxlim, maxlim], [0, 0], color='white', alpha=0.8)
    axis.plot([0, 0], [0, 0], [-maxlim, maxlim], color='white', alpha=0.8)
    axis.set(**AXIS)
    axis.text(maxlim, 0, 0, 'X', color='white', fontsize=8, ha='center', va='center')
    axis.text(0, maxlim, 0, 'Y', color='white', fontsize=8, ha='center', va='center')
    axis.text(0, 0, 10, 'Z [10 kpc]', color='white', fontsize=8, ha='center', va='center')
    return None


def initialize_plot():
    fig = plt.figure(figsize=(10, 12))
    gspec = fig.add_gridspec(1, 2, width_ratios=[1, 1/200], wspace=0.0)
    axis = fig.add_subplot(gspec[0], projection='3d')
    cbar_axis = fig.add_subplot(gspec[1])
    cbar_axis.set_zorder(10)
    cbar_axis.set_position([0.8, 0.375, 0.01, 0.25])  # [left, bottom, width, height]
    return fig, axis, cbar_axis


def add_cbar(cbar_axis, mycmap, mynorm):
    sm = ScalarMappable(cmap=mycmap, norm=mynorm)
    sm.set_array([])  # Dummy array for ScalarMappable
    cbar = plt.colorbar(sm, cax=cbar_axis, orientation='vertical',label=r'$\vec{g}$ $[\text{km}^2/ \text{s}^2 \text{kpc} ]$') 
    return cbar


def orbit_vector_colors(mycmap,mynorm,magnitude, i, qskip):
    orbcolors           =   mycmap(mynorm(magnitude[i]))
    veccolors           =   mycmap(mynorm(magnitude[i][::qskip]))
    return orbcolors, veccolors


def extract_subsets(i,time_indexing,tau_indexing,rawdata,qskip):
    # unpack the inputs
    gcorbits,hostorbit,accelerations=rawdata
    xs,ys,zs            =   gcorbits
    x,y,z               =   hostorbit
    ax,ay,az            =   accelerations
    
    current_index       =   time_indexing[i]
    sampling_indexes    =   tau_indexing + current_index    
    
    POS_GCS =   xs[:,current_index],ys[:,current_index],zs[:,current_index]
    xx,yy,zz  =   x[sampling_indexes],y[sampling_indexes],z[sampling_indexes]
    COM     =   x[current_index],y[current_index],z[current_index]
    Q_POS   =   xx[::qskip],yy[::qskip],zz[::qskip]
    accels  =   ax[i][::qskip],ay[i][::qskip],az[i][::qskip]
    STREAM =   xx,yy,zz
    return POS_GCS, Q_POS, COM, STREAM, accels


########### READING DATA
def extract_foorb_data(infilename):
    with h5py.File(infilename, "r") as foorb:
        ax=foorb['ax'][()]
        ay=foorb['ay'][()]
        az=foorb['az'][()]
        tau=foorb['tau'][()]
        time=foorb['time'][()]
        tau_indexing=foorb['tau_indexing'][()]
        time_indexing=foorb['time_indexing'][()]
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    return ax,ay,az,tau,time,tau_indexing,time_indexing,magnitude


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

if __name__ == "__main__":
    GCname = "Pal5"
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey = "monte-carlo-027"
    dataparams = GCname,MWpotential,montecarlokey
    
    main(dataparams)