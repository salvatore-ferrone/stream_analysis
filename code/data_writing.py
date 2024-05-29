import os 
import h5py 
import numpy as np


author = 'Salvatore Ferrone'
email = "salvatore.ferrone@uniroma1.it"


##############################################################
##################### DATA I/O FUNCTIONS #####################
##############################################################
def initialize_stream_tau_output(GCname,mcarlokey,path_orbit,path_stream_orbit,\
    time_stamps,out_array,outpath,outname):

    note = "This file contains a 1D stream density profile projected onto the Orbit. The unit is in time, i.e. how far ahead of behind the particle is. "

    print(outpath+outname)
    os.makedirs(outpath, exist_ok=True)
    with h5py.File(outpath+outname, 'a') as myoutfile:
        myoutfile.attrs['author'] = author
        myoutfile.attrs['email'] = email
        myoutfile.attrs['creation_date'] = str(np.datetime64('now'))
        myoutfile.attrs['note'] = note
        myoutfile.attrs['unitT'] = str(time_stamps.unit)
        myoutfile.attrs['path_orbit'] = path_orbit
        myoutfile.attrs['GCname'] = GCname
        myoutfile.attrs['mcarlokey'] = mcarlokey
        myoutfile.attrs['path_stream_orbit'] = path_stream_orbit
        
        if "time_stamps" not in myoutfile:
            myoutfile.create_dataset("time_stamps", data=time_stamps.value)
        if "density_profile" not in myoutfile:
            myoutfile.create_dataset("density_profile", data=out_array)
