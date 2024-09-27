"""
Takes the impact geometry that was saved in a bunch of HDF5 files and assembles them into a single CSV file.
"""

from gcs import path_handler as ph
import pandas as pd
import h5py
import numpy as np
import os 


if __name__ == '__main__':
    # Load the data
    GCname = "Pal5"
    NP = int(1e5)
    MWpotential = "pouliasis2017pii-GCNBody"
    internal_dynamics = "isotropic-plummer"


    outfilename = GCname + "-perturber-suspects.csv"

    outpath=ph.paths['simulations'] + "ImpactGeometry/" + MWpotential + "/" 

    row_list = []
    for j in range(50):
        montecarlokey = "monte-carlo-"+str(j).zfill(3)
        fnameSuspects   =   ph.PerturberSuspects(GCname,MWpotential,montecarlokey)
        suspects = pd.read_csv(fnameSuspects)
        for i in range(len(suspects['suspects'])):
            target_number = i
            suspect=suspects['suspects'][target_number]
            infile=ph.ImpactGeometry(GCname,MWpotential,montecarlokey,suspect,target_number)
            if not os.path.exists(infile):
                print("File not found: ",infile)
                continue
            with h5py.File(infile, 'r') as myfile:
                impact_parameter=myfile['geometry']['impact_parameter'][()]
                w_par=np.linalg.norm(myfile['geometry']['w_par'][()])
                w_per=np.linalg.norm(myfile['geometry']['w_per'][()])
                v_rel = np.sqrt(w_par**2 + w_per**2)
                alpha=myfile['geometry']['alpha'][()]
                Mass=myfile['geometry']['Mass'][()]
                rplum=myfile['geometry']['rplum'][()]
                T=myfile['geometry']['T'][()]
                tau=myfile['geometry']['tau'][()]
                gap_flag = False
                profile_density = myfile['geometry']['profile_density'][()]
                data_slice = {'impact_parameter':impact_parameter,'w_par':w_par,'w_per':w_per,'v_rel':v_rel,'alpha':alpha,'Mass':Mass,'rplum':rplum,'T':T,'tau':tau,"suspect":suspect,"montecarlokey":montecarlokey,"target_number":target_number,"gap_flag":gap_flag,"profile_density":profile_density}
            row_list.append(data_slice)
    df = pd.DataFrame(row_list)
    df.to_csv(outpath + outfilename,index=False)
    print("Saved to ",outpath + outfilename)