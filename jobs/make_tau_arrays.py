import numpy as np
import h5py
import datetime
import multiprocessing as mp

import sys 
sys.path.append("../code/")
import data_extractors as DE # type: ignore

import compute_stream_1D_density as CSD # type: ignore
import path_handler as PH # type: ignore

### Global variables 
author              =    'Salvatore Ferrone'
email               =    "salvatore.ferrone@uniroma1.it"
DOmultiprocessing   =    True

def base_parameters(mcarlo,):
    GCname                  =   "Pal5"
    potential_orbit         =   "pouliasis2017pii-GCNBody"
    potential_stream_orbit  =   "pouliasis2017pii-Pal5-suspects"
    NP                      =   int(1e5)
    mcarlokey               =   "monte-carlo-"+str(mcarlo).zfill(3)
    NCPU                    =   10
    return GCname,potential_orbit,potential_stream_orbit,NP,mcarlokey,NCPU


def main(mcarlo,startindex=40):
    

    #########################
    #### BASE PARAMETERS ####
    #########################
    GCname,potential_orbit,potential_stream_orbit,NP,mcarlokey,NCPU=base_parameters(mcarlo)
    

    path_orbit          =   PH.orbit(GCname,potential_orbit)
    path_stream_orbit   =   PH.stream_orbit(GCname,mcarlokey,potential_stream_orbit,NP)
    temporary_path      =   PH.temporary_tau_coordiantes_folder(GCname,mcarlokey,potential_stream_orbit)
    outfilename         =   PH.tau_coordinates(GCname,mcarlokey,potential_stream_orbit)


    #####   ##### #####     #####   ##### ##
    ##### OPENING NECESSARY DATA FILES #####
    #####   ##### #####     #####   ##### ##

    hostorbit = DE.get_orbit(path_orbit,mcarlokey)

    with h5py.File(path_stream_orbit, 'r') as stream:
        t_stamps=DE.extract_time_steps_from_stream_orbit(stream)
    
    ####### ####### ####### ####### #######
    ####### DERIVE OTHER PARAMETERS ####### 
    ## ####### ####### ####### ####### ###
    Nstamps     =   len(t_stamps)
    end_index   =   Nstamps
    period      =   CSD.fourier_strongest_period(hostorbit[0],hostorbit[1],hostorbit[2],hostorbit[3])
    

    attrs = {
        'unitT':str(t_stamps.unit),
        'path_orbit':path_orbit,
        'GCname':GCname,
        'mcarlokey':mcarlokey,
        'path_stream_orbit':path_stream_orbit,
        "potential_orbit":potential_orbit,
        'period':period,}
    
    #################### ############
    #### PERFORM THE COMPUTATION ####
    ############ ####################
    starttime=datetime.datetime.now()
    if DOmultiprocessing:
        pool = mp.Pool(NCPU)
        for i in range(startindex,end_index):
            pool.apply_async(worker_and_save, args=(i,path_stream_orbit,hostorbit,t_stamps,period,temporary_path))
        pool.close()
        pool.join()
    else:
        for i in range(startindex,end_index):
            worker_and_save(i,path_stream_orbit,hostorbit,t_stamps,period,temporary_path)

    endtime=datetime.datetime.now()
    print("Computation duration: ",endtime-starttime)
    
    ######## ######## ## ######## ######## ######## ## 
    ########## EXTRACT THE DATA FROM THE TEMP FILES ##
    ######## AND PUT THEM INTO THE FINAL FILE ########
    ## ######## ######## ## ######## ######## ######## 
    starttime=datetime.datetime.now()
    create_outfile_from_temp_files(\
        outfilename,\
        temporary_path,
        t_stamps,\
        NP,attrs=attrs)
    endtime=datetime.datetime.now()
    print("File update duration: ",endtime-starttime)
    
       
def worker_and_save(i,path_stream_orbit,hostorbit,t_stamps,period,temporary_path):
    """ Worker function that computes the tau array and saves it to a temporary file."""
    tau     =   CSD.obtain_tau(i,path_stream_orbit,hostorbit,t_stamps,period)
    outname =   temporary_path+"tau_"+str(i).zfill(4)+".npy"
    np.save(outname,tau)

    
##############################################################
##################### DATA I/O FUNCTIONS #####################
##############################################################
def create_outfile_from_temp_files(\
    outname,\
    tempdir,
    time_stamps,
    NP,
    attrs):
    """
    Iterates over the temporary files and creates the final file.
    """

    note = "This file contains particle's projected onto the Orbit. The unit is in time, i.e. how far ahead of behind the particle is. "

    
    # Get the file names of the tempfiles
    files = PH.os.listdir(tempdir)
    # extract the numerical indexes of each file
    index = [f.split('_')[1] for f in files]
    index = [int(i.split('.')[0]) for i in index]

    # check if the file exists
    if PH.os.path.exists(outname):
        new_file = h5py.File(outname, 'r+')
    else:
        new_file = h5py.File(outname, 'a')
    
    # update the attribtues
    new_file.attrs['author'] = author
    new_file.attrs['email'] = email 
    new_file.attrs['creation_date'] = str(np.datetime64('now'))
    new_file.attrs['note'] = note
    for attr in attrs:
        new_file.attrs[attr] = attrs[attr]
        
    # if it doesn't exist, create it
    if not ("time_stamps" in new_file):
        new_file.create_dataset("time_stamps", data=time_stamps)

    if not ("tau" in new_file):
        new_file.create_dataset("tau", (len(time_stamps), NP))

    for i in range(len(files)):
        tau=np.load(tempdir+files[i])
        new_file["tau"][index[i]]=tau
    new_file.close()


if __name__ == "__main__":
    marclo=int(sys.argv[1])
    main(marclo)