import matplotlib.pyplot as plt
import numpy as np
import h5py
import plotting_functions as PF
import sys 
sys.path.append("/obs/sferrone/gc-tidal-loss/code/")
import StreamOrbitCoords as SOC
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


'''
DECOMPOSITION OF MONTE CARLO 009

'''

############################################################
############################################################
##################### GLOBAL VARIABLES #####################
############################################################
############################################################

#### PATH VARIABLES
streampathbase="/scratch2/sferrone/simulations/Streams/"
orbitpathbase="/scratch2/sferrone/simulations/Orbits/"

##### integration variables 
NP=100000
montecarlokey="monte-carlo-009"
internaldynamics = "isotropic-plummer"
targetGC="Pal5"
### the names of the Pal5 orbits that were used for each stream computation
correspondingOrbits = {
    "pouliasis2017pii": "pouliasis2017pii",
    "pouliasis2017pii-GCNBody": "pouliasis2017pii-GCNBody/",
    "pouliasis2017pii-NGC2808": "pouliasis2017pii-GCNBody/",
    "pouliasis2017pii-NGC104": "pouliasis2017pii-GCNBody/",
    "pouliasis2017pii-NGC7078": "pouliasis2017pii-GCNBody/",
}

###### PARAMS FOR PLOT COMPUTATION ######
xmax,ymax=10,1
currenttime=0
potentials = ["pouliasis2017pii",
              "pouliasis2017pii-GCNBody",
              "DIFF",
              "pouliasis2017pii-NGC2808",
              "pouliasis2017pii-NGC104",
              "pouliasis2017pii-NGC7078"]
Nrows=len(potentials)



###### PARAMS FOR PLOT ASTHETICS
xlim,ylim=10,0.5
diffScatterProperties = {
    "cmap": "coolwarm_r",
    "s": 10,
    "alpha": 0.5,
    "edgecolor": "none",
    "marker": "d",}

normalScatterProperties = {
    "alpha":0.9, 
    "s":1,
    "cmap":'rainbow',
    "norm":LogNorm(vmin=1e-5, vmax=5e-3)
    }

titles = {
    "pouliasis2017pii": "Vanilla (Galaxy & Pal5 only)",
    "pouliasis2017pii-GCNBody": "Vanilla + All GCs ",
    "DIFF": "DIFF [Vanilla - GCs]",
    "pouliasis2017pii-NGC2808": r"Vanilla + $\vec{F}_{\mathrm{NGC2808}}$",
    "pouliasis2017pii-NGC104": r"Vanilla +  $\vec{F}_{\mathrm{NGC104}}$",
    "pouliasis2017pii-NGC7078": r"Vanilla + $\vec{F}_{\mathrm{NGC7078}}$"
}

cbar_labels = {
    "pouliasis2017pii": r"$\sigma/\sigma_{\mathrm{max}}$",
    "pouliasis2017pii-GCNBody": r"$\sigma/\sigma_{\mathrm{max}}$",
    "DIFF": r"$\delta \sigma$ [counts]",
    "pouliasis2017pii-NGC2808": r"$\sigma/\sigma_{\mathrm{max}}$",
    "pouliasis2017pii-NGC104": r"$\sigma/\sigma_{\mathrm{max}}$",
    "pouliasis2017pii-NGC7078": r"$\sigma/\sigma_{\mathrm{max}}$",
}
clabelpad=-50
titleposition = 0.70
ylabel="y' [kpc]"
xlabel="x' [kpc]"
NP_ = NP


dark=True
OUTNAME = "./final_plots/monte-carlo-009-decomposition.png"
if dark:
    plt.style.use('dark_background')
    OUTNAME = "./final_plots/monte-carlo-009-decomposition_dark.png"



def main():
    
    ### Grab the data
    outScatter = {}
    outScatter = grab_all_normal_scatter_density_points(outScatter)
    
    ### Grab the data for the DIFF
    X,Y,H=grab_scatter_density_points_DIFF()
    outScatter["DIFF"] = {}
    outScatter["DIFF"]["X"],outScatter["DIFF"]["Y"],outScatter["DIFF"]["H"]=X,Y,H
    
    ### Add the cbar limits
    std=np.std(H.flatten())
    diffScatterProperties['vmin']=-np.ceil(std)
    diffScatterProperties['vmax']= np.ceil(std)
    
    ## Start the plot
    fig,axis=set_up_plot()
    
    ## fill the plot 
    for i in range(len(potentials)):
        if potentials[i]=="DIFF":
            outScatter["DIFF"]
            X=outScatter["DIFF"]["X"]
            Y=outScatter["DIFF"]["Y"]
            H=outScatter["DIFF"]["H"]
            im=axis[0][i].scatter(X,Y,c=H,**diffScatterProperties)
        else:
            X=outScatter[potentials[i]]["X"]
            Y=outScatter[potentials[i]]["Y"]
            H=outScatter[potentials[i]]["H"]
            im=axis[0][i].scatter(X,Y,c=H,**normalScatterProperties)
        
        cbar=fig.colorbar(im,cax=axis[1][i])
        cbar.set_label(cbar_labels[potentials[i]], labelpad=clabelpad)

    ## add the info and labels
    for i in range(len(potentials)):
        axis[0][i].set_title(titles[potentials[i]],y=titleposition)
        axis[0][i].set_xlim(-xlim,xlim)
        axis[0][i].set_ylim(-ylim,ylim)
        axis[0][i].set_ylabel(ylabel)
        if i < len(potentials)-1:
            axis[0][i].set_xticklabels([])
    axis[0][Nrows-1].set_xlabel(xlabel)
    
    ## save the plot
    fig.tight_layout()
    fig.savefig(OUTNAME)
    plt.close(fig)


def set_up_plot():
    fig=plt.figure(figsize=(10,1*Nrows + 1))
    gs=gridspec.GridSpec(Nrows,2, width_ratios=[1,0.01])
    axis=[[],[]]
    for i in range(Nrows):
        axis[0].append(fig.add_subplot(gs[i,0]))
        axis[1].append(fig.add_subplot(gs[i,1]))
    return fig,axis

def grab_scatter_density_points(streamFileName,orbitPath,currenttime=0):
    myStream = h5py.File(streamFileName, 'r')

    with h5py.File(orbitPath,'r') as target:
        tGC =target[montecarlokey]['t'][:]
        xGC =target[montecarlokey]['xt'][:]
        yGC =target[montecarlokey]['yt'][:]
        zGC =target[montecarlokey]['zt'][:]
        vxGC=target[montecarlokey]['vxt'][:]
        vyGC=target[montecarlokey]['vyt'][:]
        vzGC=target[montecarlokey]['vzt'][:]
        tGC,xtGC,ytGC,ztGC,vxtGC,vytGC,vztGC=SOC.filter_orbit_by_dynamical_time(tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC,time_of_interest=currenttime,nDynTimes=2)
    # extract
    xp=myStream[internaldynamics][str(NP)][montecarlokey]['x']
    yp=myStream[internaldynamics][str(NP)][montecarlokey]['y']
    zp=myStream[internaldynamics][str(NP)][montecarlokey]['z']
    vxp=myStream[internaldynamics][str(NP)][montecarlokey]['vx']
    vyp=myStream[internaldynamics][str(NP)][montecarlokey]['vy']
    vzp=myStream[internaldynamics][str(NP)][montecarlokey]['vz']
    xf,yf,zf,vxf,vyf,vzf,indexes=SOC.transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vxp,vyp,vzp,tGC,xtGC,ytGC,ztGC,vxtGC,vytGC,vztGC,t0=currenttime)
    xedges,yedges=PF.histogramEdges(NP,xmax,ymax)
    X,Y,H=PF.getHistogram2d(xf,yf,xedges,yedges)
    H=H/NP
    myStream.close()
    return X,Y,H

def grab_scatter_density_points_DIFF():
    """
    NP_ could be the number of particles, which is used to scale the bin size. 
        But also this can be changed to something larger or smaller to see how the
        histogram looks
    
    """
    
    orbitPath1=orbitpathbase+"pouliasis2017pii"+"/"+targetGC+"-orbits.hdf5"
    orbitPath2=orbitpathbase+"pouliasis2017pii-GCNBody"+"/"+targetGC+"-orbits.hdf5"
    streamPath1=streampathbase+"pouliasis2017pii"+"/"+targetGC+"/"+str(NP)+"/"
    streamPath2=streampathbase+"pouliasis2017pii-GCNBody"+"/"+targetGC+"/"+str(NP)+"/"
    streamFileName1=streamPath1+targetGC+"-streams-"+montecarlokey+".hdf5"
    streamFileName2=streamPath2+targetGC+"-stream-"+montecarlokey+".hdf5"
    
    myStream1 = h5py.File(streamFileName1, 'r')
    myStream2 = h5py.File(streamFileName2, 'r')


    with h5py.File(orbitPath1,'r') as target:
        tGC1 =target[montecarlokey]['t'][:]
        xGC1 =target[montecarlokey]['xt'][:]
        yGC1 =target[montecarlokey]['yt'][:]
        zGC1 =target[montecarlokey]['zt'][:]
        vxGC1=target[montecarlokey]['vxt'][:]
        vyGC1=target[montecarlokey]['vyt'][:]
        vzGC1=target[montecarlokey]['vzt'][:]

        tGC1,xtGC1,ytGC1,ztGC1,vxtGC1,vytGC1,vztGC1=SOC.filter_orbit_by_dynamical_time(tGC1,xGC1,yGC1,zGC1,vxGC1,vyGC1,vzGC1,time_of_interest=currenttime,nDynTimes=2)

    with h5py.File(orbitPath2,'r') as target:
        tGC2 =target[montecarlokey]['t'][:]
        xGC2 =target[montecarlokey]['xt'][:]
        yGC2 =target[montecarlokey]['yt'][:]
        zGC2 =target[montecarlokey]['zt'][:]
        vxGC2=target[montecarlokey]['vxt'][:]
        vyGC2=target[montecarlokey]['vyt'][:]
        vzGC2=target[montecarlokey]['vzt'][:]
        tGC2,xtGC2,ytGC2,ztGC2,vxtGC2,vytGC2,vztGC2=SOC.filter_orbit_by_dynamical_time(tGC2,xGC2,yGC2,zGC2,vxGC2,vyGC2,vzGC2,time_of_interest=currenttime,nDynTimes=2)
    # extract
    xp1=myStream1[internaldynamics][str(NP)][montecarlokey]['x']
    yp1=myStream1[internaldynamics][str(NP)][montecarlokey]['y']
    zp1=myStream1[internaldynamics][str(NP)][montecarlokey]['z']
    vxp1=myStream1[internaldynamics][str(NP)][montecarlokey]['vx']
    vyp1=myStream1[internaldynamics][str(NP)][montecarlokey]['vy']
    vzp1=myStream1[internaldynamics][str(NP)][montecarlokey]['vz']
    # extract
    xp2=myStream2[internaldynamics][str(NP)][montecarlokey]['x']
    yp2=myStream2[internaldynamics][str(NP)][montecarlokey]['y']
    zp2=myStream2[internaldynamics][str(NP)][montecarlokey]['z']
    vxp2=myStream2[internaldynamics][str(NP)][montecarlokey]['vx']
    vyp2=myStream2[internaldynamics][str(NP)][montecarlokey]['vy']
    vzp2=myStream2[internaldynamics][str(NP)][montecarlokey]['vz']

    xf1,yf1,zf1,vxf1,vyf1,vzf1,indexes1=SOC.transform_from_galactico_centric_to_tail_coordinates(xp1,yp1,zp1,vxp1,vyp1,vzp1,tGC1,xtGC1,ytGC1,ztGC1,vxtGC1,vytGC1,vztGC1,t0=currenttime)
    xf2,yf2,zf2,vxf2,vyf2,vzf2,indexes2=SOC.transform_from_galactico_centric_to_tail_coordinates(xp2,yp2,zp2,vxp2,vyp2,vzp2,tGC2,xtGC2,ytGC2,ztGC2,vxtGC2,vytGC2,vztGC2,t0=currenttime)
    xedges,yedges=PF.histogramEdges(NP_,xmax,ymax)
    X,Y,H=PF.getHistogram2dDIFF(xf1,yf1,xf2,yf2,xedges,yedges)
    myStream1.close()
    myStream2.close()
    return X,Y,H

def grab_all_normal_scatter_density_points(
    outScatter:dict,
    currenttime:int=0):

    for i in range(len(potentials)):
        if potentials[i]!="DIFF":
            streamPath=streampathbase+potentials[i]+"/"+targetGC+"/"+str(NP)+"/"
            streamFileName=streamPath+targetGC+"-stream-"+montecarlokey+".hdf5"
            if  potentials[i]=="pouliasis2017pii":
                streamFileName=streamFileName.replace("-stream-","-streams-")

            orbitPath=orbitpathbase+correspondingOrbits[potentials[i]]+"/"+targetGC+"-orbits.hdf5"

            X,Y,H=grab_scatter_density_points(streamFileName,orbitPath,currenttime=currenttime)
            # store the output in a dictionary
            outScatter[potentials[i]] = {}
            outScatter[potentials[i]]["X"]=X
            outScatter[potentials[i]]["Y"]=Y
            outScatter[potentials[i]]["H"]=H
        else:
            continue
    return outScatter


if __name__ == "__main__":
    main()