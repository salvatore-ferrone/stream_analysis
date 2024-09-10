import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import h5py
import os
# routines in this folder
import plotting_functions as PF
import gap_results as GR
import sys 
sys.path.append("/obs/sferrone/gc-tidal-loss/code/")
import StreamOrbitCoords as SOC


def main():
    # get the parameters
    NP,internaldynamics,potential,targetGC,xmax,ymax = default_parameters()
    # set the paths
    basepath="/scratch2/sferrone/simulations/Streams/"+potential+"/"+targetGC+"/"+str(NP)+"/"
    orbitpath="/scratch2/sferrone/simulations/Orbits/"+potential+"/"+targetGC+"-orbits.hdf5"
    # make sure the out directory exists
    outdir="./gallery_of_gaps/"
    os.makedirs(outdir,exist_ok=True)
    # get the perturber dictionary
    MC_gaps=GR.GetPerturberDictPouliasis2017pii()
    MCkeys=list(MC_gaps.keys())
    MCkeys = ["monte-carlo-"+str(i).zfill(3) for i in range(0,50)]
    # current time is always 0, b/c zero means today
    currenttime= 0 # started at -5*u.Gyr
    n_dynamic_times=2 # filtering the orbit by dynamical time
    # iterate over the monte-carlo-keys
    
    scat_params={'s':1,'alpha':0.9,'cmap':'rainbow','norm':LogNorm(vmin=1e-5, vmax=5e-4)}
    for montecarlokey in MCkeys:
        # open the stream 
        streamfilename = basepath+targetGC+"-stream-"+montecarlokey+".hdf5"
        myStream = h5py.File(streamfilename, 'r')
        # take a piece of the orbit in order to do the coordinate transform
        with h5py.File(orbitpath,'r') as target:
            tGC =target[montecarlokey]['t'][:]
            xGC =target[montecarlokey]['xt'][:]
            yGC =target[montecarlokey]['yt'][:]
            zGC =target[montecarlokey]['zt'][:]
            vxGC=target[montecarlokey]['vxt'][:]
            vyGC=target[montecarlokey]['vyt'][:]
            vzGC=target[montecarlokey]['vzt'][:]
            
            tGC,xtGC,ytGC,ztGC,vxtGC,vytGC,vztGC=SOC.filter_orbit_by_dynamical_time(tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC,time_of_interest=currenttime,nDynTimes=n_dynamic_times)
            
        # store the particles in more dans les variables qui conviennent plus
        xp=myStream[internaldynamics][str(NP)][montecarlokey]['x']
        yp=myStream[internaldynamics][str(NP)][montecarlokey]['y']
        zp=myStream[internaldynamics][str(NP)][montecarlokey]['z']
        vxp=myStream[internaldynamics][str(NP)][montecarlokey]['vx']
        vyp=myStream[internaldynamics][str(NP)][montecarlokey]['vy']
        vzp=myStream[internaldynamics][str(NP)][montecarlokey]['vz']
        # transform to the orbit frame
        print("Transforming ", montecarlokey, " to orbit frame")
        xf,yf,zf,vxf,vyf,vzf,indexes=SOC.transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vxp,vyp,vzp,tGC,xtGC,ytGC,ztGC,vxtGC,vytGC,vztGC,t0=currenttime)
        # get the denisty of the particles
        xedges,yedges=PF.histogramEdges(NP*10,xmax,ymax)
        X,Y,H=PF.getHistogram2d(xf,yf,xedges,yedges)
        H=H/NP
        H=H/NP
        print(H.max(), H.min())
        # do the plot
        fig,axis,im,cbar= PF.plot2dHist(X,Y,H,scat_params=scat_params)
        axis[0].set_xlim(-xmax,xmax);
        title = "Sampling "+montecarlokey[-3:]
        axis[0].set_title(title)
        cbar.set_label('Normalized density', labelpad=-50)
        # make the output file name
        outname = outdir+targetGC+"-"+montecarlokey+"-xy.png"
        # save the figure
        fig.tight_layout()
        fig.savefig(outname,dpi=300)
        # close the stream
        myStream.close()
        # close the figure
        plt.close(fig)
        print(outname, "saved")
        



def default_parameters(\
    NP=100000,
    internaldynamics = "isotropic-plummer",
    potential="pouliasis2017pii-GCNBody",
    targetGC="Pal5",
    xmax=10,
    ymax=1):
    """
        the default integration parameters
    """
    return NP,internaldynamics,potential,targetGC,xmax,ymax

if __name__=="__main__":
    
    main()

