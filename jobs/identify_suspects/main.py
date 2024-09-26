"""
NOTE.

You cannot use the (tstrippy) conda environment for this because we need scipy which necessitates numpy>1.22. 
    Numpy 1.23 and above is incompatible with the current version of the tstrippy environment because they no longer support the fortran compiler


Use the results from:
    FORCE ON ORBIT
    STREAM LENGTH

in order to identify the moments the are most likely responsible for the gaps.

Save the top 5 moments in some kind of file

"""


# imports
import stream_analysis as sa
from gcs import path_handler as ph 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

import os 




def main(dataparams,hyperparams):
    internal_dynamics,montecarlokey,NP,MWpotential,GCname = dataparams
    threshold,NKEEPS,fname = hyperparams

    fileFOOB = ph.ForceOnOrbit(GCname,MWpotential,montecarlokey)
    fileTau  = ph.tauDensityMaps(GCname=GCname,MWpotential=MWpotential,montecarlokey=montecarlokey,NP=NP,internal_dynamics=internal_dynamics)

    outfilename=ph.PerturberSuspects(GCname,MWpotential,montecarlokey)

    if not os.path.exists(fileFOOB):
        print(fileFOOB, " not found")
        return None 
    
    if not os.path.exists(fileTau):
        print(fileFOOB, " not found")
        return None

    if os.path.exists(outfilename):
        print(outfilename, " already exists, overwriting")
        # return None 


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
    coordinates = [coordinates[i] for i in range(NKEEPS) ]

    time,tau,mag,convolved,suspects=sa.identify_suspects.build_output_data(coordinates,tau_foob,time_foob,mag_total,convolved_data,GCmags,Perturbers,NKEEPS)
    dframe =pd.DataFrame({'time':time,'tau':tau,'mag':mag,'convolved':convolved,'suspects':suspects})
    dframe.to_csv(outfilename,index=False)
    print("Saved to ",outfilename)

    doplot=False
    if doplot:
        fig,axis1,axis2,caxisFOOB,caxisTau = sa.identify_suspects.getfigure()
        foobNORM,tauNORM,AXIS1,AXIS2,TEXT = sa.identify_suspects.properties(time_foob,tau_left,tau_right,NP,montecarlokey)
        FIGSTUFF=[fig,axis1,axis2,caxisFOOB,caxisTau]
        DATASTUFF=time_foob,tau_foob,convolved_mask,suspects,coordinates,tau_counts,tau_centers,time_stamps,tau_left,tau_right
        PROPERTIESSTUFF=foobNORM,tauNORM,AXIS1,AXIS2,TEXT
        sa.identify_suspects.doplot(FIGSTUFF,DATASTUFF,PROPERTIESSTUFF)
        fig.savefig(fname,dpi=300) 
        print("Saved to ",fname)
        plt.close(fig)


if __name__=="__main__":
    internal_dynamics   =   "isotropic-plummer"
    montecarlokey       =   "monte-carlo-003"
    NP                  =   int(1e5)
    MWpotential         =   "pouliasis2017pii-GCNBody"
    GCname              =   "Pal5"    
    
    # hyper params
    threshold=50
    NKEEPS = 8
    plotdir="/home/sferrone/plots/stream_analysis/identify_suspects/"


    for i in range(50):
        montecarlokey = "monte-carlo-"+str(i).zfill(3)
        fname = plotdir+GCname+"_"+MWpotential+"_"+montecarlokey+"_suspects.png"
        dataparams = (internal_dynamics,montecarlokey,NP,MWpotential,GCname)
        hyperparams = (threshold,NKEEPS,fname)
        main(dataparams,hyperparams)