"""
Use the results from:
    FORCE ON ORBIT
    STREAM LENGTH

in order to identify the moments the are most likely responsible for the gaps.

Save the top 5 moments in some kind of file

"""


# imports
import stream_analysis as sa
from gcs import path_handler as ph 
import h5py
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.ndimage import maximum_filter, label, find_objects
from scipy import ndimage




def main(dataparams,hyperparams):
    internal_dynamics,montecarlokey,NP,MWpotential,GCname = dataparams
    threshold,NKEEPS,fname = hyperparams

    fileFOOB = ph.ForceOnOrbit(GCname,MWpotential,montecarlokey)
    fileTau  = ph.tauDensityMaps(GCname=GCname,MWpotential=MWpotential,montecarlokey=montecarlokey,NP=NP,internal_dynamics=internal_dynamics)


    time_foob,tau_foob,mag_total,Perturbers,GCmags = sa.identify_suspects.extract_FOOB(fileFOOB)
    tau_centers,time_stamps,tau_counts = sa.identify_suspects.extract_tau_stream_density(fileTau)

    # get the envlopes
    leftindexes,rightindexes=sa.streamLength.get_envelop_indexes(tau_counts,threshold)
    tau_left,tau_right=sa.streamLength.tau_envelopes(tau_centers,leftindexes,rightindexes)


    mask=sa.identify_suspects.build_mask_on_FOOB_from_density((time_stamps,tau_left,tau_right),(time_foob,tau_foob,mag_total))
    masked_data = mag_total[mask]

    kernel = np.ones(4) / 4  # 5-point moving average
    convolved_data = ndimage.convolve1d(mag_total, kernel, axis=0, mode='reflect')
    convolved_mask = np.ma.masked_where(~mask,convolved_data)


    coordinates = sa.identify_suspects.get_peaks(convolved_mask)

    time,tau,mag,convolved,suspects=sa.identify_suspects.build_output_data(coordinates,tau_foob,time_foob,mag_total,convolved_data,GCmags,Perturbers,NKEEPS)
    outfilename=ph.PerturberSuspects(GCname,MWpotential,montecarlokey)
    dframe =pd.DataFrame({'time':time,'tau':tau,'mag':mag,'convolved':convolved,'suspects':suspects})
    dframe.to_csv(outfilename,index=False)
    print("Saved to ",outfilename)


    fig,axis1,axis2,caxisFOOB,caxisTau = sa.identify_suspects.getfigure()
    foobNORM,tauNORM,AXIS1,AXIS2,TEXT = sa.identify_suspects.properties(time_foob,tau_left,tau_right,NP,montecarlokey)
    FIGSTUFF=[fig,axis1,axis2,caxisFOOB,caxisTau]
    DATASTUFF=time_foob,tau_foob,convolved_mask,suspects,coordinates,tau_counts,tau_centers,time_stamps,tau_left,tau_right
    PROPERTIESSTUFF=foobNORM,tauNORM,AXIS1,AXIS2,TEXT
    sa.identify_suspects.doplot(FIGSTUFF,DATASTUFF,PROPERTIESSTUFF)
    fig.savefig(fname,dpi=300) 
    print("Saved to ",fname)
    plt.close(fig)


## i/o
def build_output_data(coordinates,tau_foob,time_foob,mag_total,convolved_data,GCmags,Perturbers,NKEEPS):
    suspects,tau,time,mag = [],[],[],[]
    convolved = []
    for i in range(NKEEPS):
        mags_all = GCmags[:,coordinates[i][0],coordinates[i][1]] - mag_total[coordinates[i][0],coordinates[i][1]]
        suspects.append(str(Perturbers[np.argmin(np.abs(mags_all))].decode('utf-8')))
        tau.append(tau_foob[coordinates[i][1]])
        time.append(time_foob[coordinates[i][0]])
        mag.append(mag_total[coordinates[i][0],coordinates[i][1]])
        convolved.append(convolved_data[coordinates[i][0],coordinates[i][1]])
    return time,tau,mag,convolved,suspects

## plotting 

def getfigure():
    fig=plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1/100])
    axis1=fig.add_subplot(gs[0,0])
    axis2=fig.add_subplot(gs[1,0])
    caxisFOOB=fig.add_subplot(gs[0,1])
    caxisTau=fig.add_subplot(gs[1,1])
    return fig,axis1,axis2,caxisFOOB,caxisTau

def properties(time_foob,tau_left,tau_right,NP,montecarlokey):
    foobpcolor    =   {"cmap":"gray","norm":mpl.colors.Normalize(vmin=0,vmax=50)}
    taupcolor     =   {"cmap":"inferno","norm":mpl.colors.LogNorm(vmin=1,vmax=NP)}
    ylimits     =   np.max([np.abs(tau_left),np.abs(tau_right)])
    AXIS1       =   {"xlim":[time_foob[0],0],"ylim":[-ylimits,ylimits],"ylabel":"$\\tau$ [s kpc / km]","xlabel":"","title":montecarlokey}
    AXIS2       =   {"xlim":[time_foob[0],0],"ylim":[-ylimits,ylimits],"ylabel":"$\\tau$ [s kpc / km]","xlabel":"$t$ [s kpc / km]",}
    TEXT        =   {"color":'red',"ha":'left',"va":'top'}
    
    return foobpcolor,taupcolor,AXIS1,AXIS2,TEXT

def doplot(FIGSTUFF,DATASTUFF,PROPERTIESSTUFF,x_text_shift=-0.1,y_text_shift=0.005):
    fig,axis1,axis2,caxisFOOB,caxisTau=FIGSTUFF
    foobpcolor,taupcolor,AXIS1,AXIS2,TEXT=PROPERTIESSTUFF
    time_foob,tau_foob,convolved_mask,suspects,coordinates,tau_counts,tau_centers,time_stamps,tau_left,tau_right=DATASTUFF
    
    
    im=axis1.pcolorfast(time_foob,tau_foob,convolved_mask.T,**foobpcolor)
    cbarfoob=fig.colorbar(im,cax=caxisFOOB,label="Force on Orbit")
    for i in range(len(coordinates)):
        x,y=coordinates[i]
        axis1.scatter(time_foob[x], tau_foob[y], marker="o", s=75,edgecolor="red",facecolors='none')
        axis1.text(x=time_foob[x]+x_text_shift,y=tau_foob[y]+y_text_shift,s=str(i)+" "+suspects[i],**TEXT)
    
    im2=axis2.pcolorfast(time_stamps,tau_centers,tau_counts.T,**taupcolor)
    cbartau=fig.colorbar(im2,cax=caxisTau,label="Counts")
    axis2.plot(time_stamps,tau_left,'r--')
    axis2.plot(time_stamps,tau_right,'r--')
    
    axis1.set(**AXIS1)
    axis2.set(**AXIS2)
    return None

## COMPUTATIONS 

def build_mask_on_FOOB_from_density(STREAM_DENSITY,FOOB):
    """Use the stream density to mask the FOOB data
        They are not sampled on the same time grid, so we do an interpolation and then build the mask 
    """
    time_stamps,tau_left,tau_right = STREAM_DENSITY
    time_foob,tau_foob,input_data = FOOB
    mask = np.ones_like(input_data, dtype=bool)
    for i in range(time_foob.shape[0]):
        lower = np.interp(time_foob[i], time_stamps, tau_left)
        upper = np.interp(time_foob[i], time_stamps, tau_right)
        cond0 = tau_foob > lower
        cond1 = tau_foob < upper
        mask[i] = np.logical_and(cond0, cond1)
    return mask


# use this wonderful code from co-pilot for finding the local maxima
def get_peaks(grid):
    # Apply maximum filter
    neighborhood_size = 10
    local_max = maximum_filter(grid, size=neighborhood_size) == grid

    # Label the local maxima
    labeled, num_objects = label(local_max)

    # Extract the coordinates of the local maxima
    slices = find_objects(labeled)
    coordinates = [(int((dy.start + dy.stop - 1) / 2), int((dx.start + dx.stop - 1) / 2)) for dy, dx in slices]

    # sort for the largest maximuma
    massimi = np.zeros(len(coordinates))
    for i,coord in enumerate(coordinates):
        massimi[i] = grid[coord[0],coord[1]]
    sortdexes=np.argsort(massimi)[::-1]
    mycoordinates=[coordinates[i] for i in sortdexes]
    return mycoordinates



# EXTACT DATA

def extract_FOOB(fileFOOB):
    with h5py.File(fileFOOB,'r') as FOOB:
        time_foob       =       np.array(FOOB['time'][:],dtype=float)
        tau_foob        =       FOOB['tau'][:]
        mag_total       =       np.sqrt(FOOB["ax"][:]**2 + FOOB["ay"][:]**2 + FOOB["az"][:]**2)
        Perturbers      =       FOOB['Perturbers'][:]
        GCmags          =       FOOB['magnitude'][:]
    return time_foob,tau_foob,mag_total,Perturbers,GCmags


def extract_tau_stream_density(fileTau):
    with h5py.File(fileTau,'r') as TauDensity:
        time_stamps=TauDensity['time_stamps'][:]
        tau_centers=TauDensity['tau_centers'][:]
        tau_counts=TauDensity['tau_counts'][:]
    return tau_centers,time_stamps,tau_counts



if __name__=="__main__":
    internal_dynamics   =   "isotropic-plummer"
    montecarlokey       =   "monte-carlo-003"
    NP                  =   int(1e5)
    MWpotential         =   "pouliasis2017pii-GCNBody"
    GCname              =   "Pal5"    
    
    # hyper params
    threshold=50
    NKEEPS = 5
    plotdir="/home/sferrone/plots/stream_analysis/identify_suspects/"
    fname = plotdir+GCname+"_"+MWpotential+"_"+montecarlokey+"_suspects.png"

    dataparams = (internal_dynamics,montecarlokey,NP,MWpotential,GCname)
    hyperparams = (threshold,NKEEPS,fname)
    main(dataparams,hyperparams)