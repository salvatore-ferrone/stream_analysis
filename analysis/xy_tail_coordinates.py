import plotters 
import h5py
import os
import numpy as np 
from matplotlib import colors
import sys 
sys.path.append("../code/")
import path_handler as PH #type: ignore
import StreamOrbitCoords as SOC
import data_extractors as DE #type: ignore



def main(GCname="Pal5",montecarlokey="monte-carlo-000",potential="pouliasis2017pii-GCNBody",NP=int(1e5)):
    
    ### plot parameters
    xmax,ymax = 12,0.5
    scat_params = {"alpha":0.9, 
                   "s":1.5, 
                   "cmap":'rainbow', 
                   "norm":colors.LogNorm(vmin=1e-5, vmax=3e-4)}
    title="{:s} {:s} {:s}".format(GCname, potential, montecarlokey)
    outfilename = set_outfile_names(GCname,montecarlokey,potential,NP)
    
    ### load in the data
    streamfilename=PH.stream(GCname,montecarlokey,potential,NP)
    xs,ys,zs,vxs,vys,vzs=load_stream(streamfilename,montecarlokey,NP)
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB=load_orbit(GCname,montecarlokey,potential)
    
    ### projcet into tail coordinates
    xprimeP,yprimeP,_,_,_,_,_=SOC.transform_from_galactico_centric_to_tail_coordinates(xs,ys,zs,vxs,vys,vzs,TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB,t0=0)
    
    ### make the density map
    xedges,yedges=plotters.histogramEdges(NP,xmax,ymax)
    X,Y,H=plotters.getHistogram2d(xprimeP,yprimeP,xedges,yedges)
    H=H/NP
    
    ### do and save the plot 
    scat_params = {"alpha":0.9, "s":1.5, "cmap":'rainbow', "norm":colors.LogNorm(vmin=1e-5, vmax=3e-4)}
    fig,axis,_,_=plotters.plot2dHist(X,Y,H,scat_params=scat_params)
    title="{:s} {:s} {:s}".format(GCname, potential, montecarlokey)
    axis[0].set_title(title)    
    fig.tight_layout()
    fig.savefig(outfilename,dpi=300)
    
    
def set_outfile_names(GCname:str,montecarlokey:str,potential:str,NP:int,plottype:str="xy-tail"):
    outpathname=PH.base['manyplots'] + plottype+"/"+potential+"/"+GCname+"/"
    os.makedirs(outpathname,exist_ok=True)
    outname = GCname+"-"+montecarlokey+"-"+potential+"-"+str(NP)+"-"+plottype+".png"
    return outpathname+outname

def load_orbit(GCname,montecarlokey,potential,n_dynamic_time = 2,time_of_interest = 0):
    pathOrbit=PH.orbit(GCname,potential)
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost=DE.get_orbit(pathOrbit,montecarlokey)
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB=DE.filter_orbit_by_dynamical_time(thost,(xhost,yhost,zhost,vxhost,vyhost,vzhost),time_of_interest,n_dynamic_time)
    return TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB

def load_stream(streamfilename,montecarlokey:str,NP:int):
    with h5py.File(streamfilename,'r') as stream:
        x=stream['isotropic-plummer'][str(NP)][montecarlokey]['x'][:]
        y=stream['isotropic-plummer'][str(NP)][montecarlokey]['y'][:]
        z=stream['isotropic-plummer'][str(NP)][montecarlokey]['z'][:]
        vx=stream['isotropic-plummer'][str(NP)][montecarlokey]['vx'][:]
        vy=stream['isotropic-plummer'][str(NP)][montecarlokey]['vy'][:]
        vz=stream['isotropic-plummer'][str(NP)][montecarlokey]['vz'][:]
    return x,y,z,vx,vy,vz


if __name__=="__main__":
    i=sys.argv[1]
    montecarlokey="monte-carlo-"+str(i).zfill(3)
    potential="pouliasis2017pii"
    main(montecarlokey=montecarlokey,potential=potential)