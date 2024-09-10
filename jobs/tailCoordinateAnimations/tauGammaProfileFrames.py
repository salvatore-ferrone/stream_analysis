import gcs
from gcs import path_handler as ph
import numpy as np 
import matplotlib.pyplot as plt
import stream_analysis as sa 
import os 
import matplotlib as mpl
import h5py


    

def gspec_tau_gamma_and_profile(FIG={"figsize":(10,5)},GSPEC={
    'width_ratios':[1,0.01],
    'height_ratios':[1,1],
    'hspace':0.001,
    'wspace':0.01
}):
    fig=plt.figure(**FIG)
    gspec=mpl.gridspec.GridSpec(2,2,**GSPEC)
    axis0 = fig.add_subplot(gspec[0,0])
    axis1 = fig.add_subplot(gspec[1,0])
    cbar_ax=fig.add_subplot(gspec[0,1])

    # Adjust the position and size of the colorbar axis to match axis0
    pos0 = axis0.get_position()
    pos1 = axis1.get_position()
    cbar_ax.set_position([pos0.x1 + 0.01, pos0.y0 + pos0.height/4 , 0.01, pos0.height/2 ])

    return fig,axis0,axis1,cbar_ax




def properties(NP,xlims=[-0.1,0.1],ylims=[-0.1,0.1]):
    PROF={'yscale':"log","xlabel":r"$\tau$ [s kpc / km]","ylabel":"counts"}
    AXIS={}
    AXIS['xlim']=xlims
    AXIS['ylim']=ylims
    AXIS['ylabel']=r"$\gamma$ [s kpc / km]"
    AXIS['xlabel']=""
    AXIS['xticks']=[]
    AXIS['aspect']="equal"
    SCAT=sa.plotters.tailcoordinates.density_SCAT_properties(NP)
    return AXIS,PROF,SCAT




def grab_map_and_profile(NP,tau,gamma,xlims,ylims):
    X,Y,H=sa.plotters.binned_density.short_cut(NP,tau,gamma,xlims,ylims)
    H=sa.plotters.binned_density.normalize_density_by_particle_number(H,NP)
    X,Y,H=sa.plotters.binned_density.order_by_density(X,Y,H)
    xedges,yedges   =   sa.plotters.binned_density.get_edges(NP,xlims,ylims)
    XX,YY,HH        =   sa.plotters.binned_density.get_2D_density(tau,gamma,xedges,yedges)
    counts          =   np.sum(HH,axis=0)
    tau_centers     =   (xedges[:-1]+xedges[1:])/2

    profile = (tau_centers,counts)
    scatter=(X,Y,H)
    return profile,scatter


def extract_snapshot(i,fname):
    with h5py.File(fname, 'r') as f:
        xp,yp,zp,vxp,vyp,vzp=f['StreamSnapShots'][str(i)][:]
    return xp,yp,zp,vxp,vyp,vzp

def make_frame(outdir,i,timestamps,fullhostorbit,streamFname,xlims=[-0.1,0.1],ylims=[-0.04,0.04]):
    plt.style.use('dark_background')
    # load orbit and stream data
    tH,xH,yH,zH,vxH,vyH,vzH = fullhostorbit
    xp,yp,zp,vxp,vyp,vzp=extract_snapshot(i,streamFname)
    

    NP = len(xp)
    currenttime=timestamps[i]

    # do coordinate transformation
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB=sa.tailCoordinates.filter_orbit_by_dynamical_time(tH,xH,yH,zH,vxH,vyH,vzH,currenttime,2)
    xT,yT,zT,vxT,vyT,vzT,indexes    =   sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vxp,vyp,vzp,TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB,t0=currenttime)
    tau,gamma                       =   sa.tailCoordinates.tau_gamma(indexes,yT,TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB,currenttime=currenttime)

    # obtain specific data for frame
    (tau_centers,counts),(X,Y,H)=grab_map_and_profile(NP,tau,gamma,xlims,ylims)


    # plot properties
    AXIS,PROF,SCAT=properties(NP,xlims,ylims)

    fig,axis0,axis1,cbar_ax=gspec_tau_gamma_and_profile(FIG={"figsize":(10,10)})

    im=axis0.scatter(X,Y,c=H,**SCAT)
    fig.colorbar(im,cax=cbar_ax,label="density")
    axis1.plot(tau_centers,counts)

    axis0.set(**AXIS)
    axis1.set(**PROF)

    fname = "frame-{:04d}.png".format(i)
    fig.savefig(outdir + fname,dpi=300)
    plt.close(fig)
    print(outdir + fname,"saved")
    return None

if __name__=="__main__":
    # data params
    GCname = "Pal5"
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey = "monte-carlo-000"
    NP = int(1e5)
    internal_dynamics = "isotropic-plummer"