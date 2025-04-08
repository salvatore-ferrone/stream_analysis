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
import os 




def main(dataparams,hyperparams):
    internal_dynamics,montecarlokey,NP,MWpotential,GCname,fnameTau = dataparams
    threshold,NKEEPS,fname = hyperparams

    fileFOOB = ph.ForceOnOrbit(GCname,MWpotential,montecarlokey)
    pathTau     =   ph._tauDensityMaps(GCname=GCname,MWpotential=MWpotential,NP=NP,internal_dynamics=internal_dynamics)
    fileTau     =   os.path.join(pathTau,fnameTau)



    time_foob,tau_foob,mag_total,Perturbers,GCmags = extract_FOOB(fileFOOB)
    tau_centers,time_stamps,tau_counts = extract_tau_stream_density(fileTau)

    # get the envlopes
    leftindexes,rightindexes=get_envelop_indexes(tau_counts,threshold)
    tau_left,tau_right=tau_envelopes(tau_centers,leftindexes,rightindexes)


    mask=build_mask_on_FOOB_from_density((time_stamps,tau_left,tau_right),(time_foob,tau_foob,mag_total))
    masked_data = mag_total[mask]

    kernel = np.ones(4) / 4  # 5-point moving average
    convolved_data = ndimage.convolve1d(mag_total, kernel, axis=0, mode='reflect')
    convolved_mask = np.ma.masked_where(~mask,convolved_data)


    coordinates = get_peaks(convolved_mask)

    time,tau,mag,convolved,suspects=build_output_data(coordinates,tau_foob,time_foob,mag_total,convolved_data,GCmags,Perturbers,NKEEPS)
    suspect_x,suspect_y= time,tau
    outfilename=ph.PerturberSuspects(GCname,MWpotential,montecarlokey)
    dframe =pd.DataFrame({'time':time,'tau':tau,'mag':mag,'convolved':convolved,'suspects':suspects})
    dframe.to_csv(outfilename,index=False)
    print("Saved to ",outfilename)


    fig,axis1,axis2,caxisFOOB,caxisTau = getfigure()
    foobNORM,tauNORM,AXIS1,AXIS2,TEXT = properties(time_foob,tau_left,tau_right,NP,montecarlokey)
    FIGSTUFF=[fig,axis1,axis2,caxisFOOB,caxisTau]
    DATASTUFF=time_foob,tau_foob,convolved_mask,suspect_x,suspect_y,suspects,tau_counts,tau_centers,time_stamps,tau_left,tau_right
    PROPERTIESSTUFF=foobNORM,tauNORM,AXIS1,AXIS2,TEXT
    doplot(FIGSTUFF,DATASTUFF,PROPERTIESSTUFF)
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
    foobpcolor    =   {"cmap":"gray","norm":mpl.colors.Normalize(vmin=0,vmax=.05)}
    taupcolor     =   {"cmap":"inferno","norm":mpl.colors.LogNorm(vmin=1,vmax=NP)}
    ylimits     =   np.max([np.abs(tau_left),np.abs(tau_right)])
    AXIS1       =   {"xlim":[time_foob[0],0],"ylim":[-ylimits,ylimits],"ylabel":"$\\tau$ [s kpc / km]","xlabel":"","title":montecarlokey}
    AXIS2       =   {"xlim":[time_foob[0],0],"ylim":[-ylimits,ylimits],"ylabel":"$\\tau$ [s kpc / km]","xlabel":"$t$ [s kpc / km]",}
    TEXT        =   {"color":'red',"ha":'left',"va":'center'}
    
    return foobpcolor,taupcolor,AXIS1,AXIS2,TEXT

def doplot(FIGSTUFF,DATASTUFF,PROPERTIESSTUFF,x_text_shift=-0.1,y_text_shift=0.005):
    fig,axis1,axis2,caxisFOOB,caxisTau=FIGSTUFF
    foobpcolor,taupcolor,AXIS1,AXIS2,TEXT=PROPERTIESSTUFF
    time_foob,tau_foob,convolved_mask,suspect_x,suspect_y,suspects,tau_counts,tau_centers,time_stamps,tau_left,tau_right=DATASTUFF
    
    
    im=axis1.pcolorfast(time_foob,tau_foob,convolved_mask[1:,1:].T,**foobpcolor)
    cbarfoob=fig.colorbar(im,cax=caxisFOOB,label="Force on Orbit")

    print("len(time_foob)",time_foob.shape)
    print("len(tau_foob)",tau_foob.shape)
    print("len(suspects)",len(suspects))
    for i in range(len(suspect_x)):
        axis1.scatter(suspect_x[i], suspect_y[i], marker="o", s=75,edgecolor="red",facecolors='none')
        axis1.text(x=suspect_x[i]+x_text_shift,y=suspect_y[i]+y_text_shift,s=str(i)+" "+suspects[i],**TEXT)
    
    im2=axis2.pcolorfast(time_stamps,tau_centers,tau_counts[1:,1:].T,**taupcolor)
    cbartau=fig.colorbar(im2,cax=caxisTau,label="Counts")
    axis2.plot(time_stamps,tau_left,'k--')
    axis2.plot(time_stamps,tau_right,'k--')
    for i in range(len(suspect_x)):
        axis2.scatter(suspect_x[i],suspect_y[i], marker="o", s=75,edgecolor="red",facecolors='none')
        axis2.text(x=suspect_x[i]+x_text_shift,y=suspect_y[i]+y_text_shift,s=str(i),**TEXT)
    
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
        tau_counts=TauDensity['counts'][:]
    return tau_centers,time_stamps,tau_counts


def get_envelop_indexes(density_array, density_min):
    """
    Finds the limits of a stream based on a density array and a minimum density threshold.

    The first axis of the density array is the simulation time
    The second axis the array is the 1D profile of the stream on the orbit 

    Parameters:
    - density_array (ndarray): A 2D array representing the density values.
    - density_min (float): The minimum density threshold.

    Returns:
    - index_from_left (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the left for each row of the density array.
    - index_from_right (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the right for each row of the density array.
    """
    
    Nstamps, _ =density_array.shape
    
    index_from_left, index_from_right = np.zeros(Nstamps), np.zeros(Nstamps)
    for i in range(Nstamps):
        array = density_array[i]
        # Find the first element that surpasses THRESHOLD when scanning from the left
        index_from_left[i] = np.argmax(array > density_min)
        # Find the first element that surpasses THRESHOLD when scanning from the right
        index_from_right[i] = len(array) - np.argmax(array[::-1] > density_min) - 1

    index_from_left = index_from_left.astype(int)
    index_from_right = index_from_right.astype(int)
    return index_from_left, index_from_right


def tau_envelopes(tau, index_from_left, index_from_right):
    """
    we want \tau(t), tau as a function of time...
    """
    tau_left = [tau[xx] for xx in index_from_left]
    tau_right = [tau[xx] for xx in index_from_right]
    return np.array(tau_left), np.array(tau_right)



def sig_clip(quantity, Nstdflag, Nstdclip, sides = 100, trial_max = 10000):

    # some signal processing in case there are problems with the end points
    std,mean    = np.std(quantity),np.mean(quantity)
    flag,clip   = mean+Nstdflag*std,mean+Nstdclip*std
    abs_diff = np.abs(quantity-mean)
    bad_dexes = np.where(abs_diff>flag)[0]
    sig_clipped=quantity.copy()
    cc = 0 
    conditions = cc < trial_max and len(bad_dexes)>0
    while conditions:
        # replace the bad dexes with the average of the two neighbors
        for bd in bad_dexes:
            uplim,lowlim=bd+sides,bd-sides
            # make sure the limits are within the array
            if uplim>len(sig_clipped):
                uplim=len(sig_clipped)
            if lowlim<0:
                lowlim=0
            sig_clipped[bd] = np.mean(sig_clipped[bd-sides:bd+sides-1])
            if sig_clipped[bd]<clip:
                bad_dexes = np.delete(bad_dexes,np.where(bad_dexes==bd))
        cc+=1
        
        if cc == trial_max:
            print('Max number of iterations reached')
        conditions = cc < trial_max and len(bad_dexes)>0
    return sig_clipped


if __name__=="__main__":
    GCname="Pal5"
    MWpotential="pouliasis2017pii-GCNBody"
    montecarlokey="monte-carlo-009"
    NP=int(103292)
    internal_dynamics = "isotropic-plummer_mass_radius_grid"
    fnameTau       =   "Pal5-tauDensity-monte-carlo-009_mass_10000_radius_029.hdf5"
    
    # hyper params
    threshold=10
    NKEEPS = 8

    plotdir="/scratch2/sferrone/plots/stream_analysis/identify_suspects/"
    os.makedirs(plotdir,exist_ok=True)
    fname = plotdir+GCname+"_"+MWpotential+"_"+montecarlokey+"_suspects.png"
    dataparams = (internal_dynamics,montecarlokey,NP,MWpotential,GCname,fnameTau)
    hyperparams = (threshold,NKEEPS,fname)
    main(dataparams,hyperparams)


