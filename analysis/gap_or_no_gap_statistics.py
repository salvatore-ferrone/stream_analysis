"""
Purpose:
Analyize all of the files in the folder /impact-geometry-results/ 
These files were created with fine_tune_impact_geometry.py 

After job_fine_tune_impact_geometry.sh was run for all globular clusters that 
have imparted at least one gap on the stream, we can now compare the statistics.
If a perturbing GC imparted at least one gap on a stream, then it will have 49 
cases without a gap. 

"""
import json
import os
import pandas as pd
import h5py
import numpy as np 
import sys 
sys.path.append("../code/")
import path_handler as PH #type: ignore

def extract_all_statistics_into_data_frame(GCname,potential):
    

    path_to_geom_files=PH.base['impact-geometry-results']+potential+"/"+GCname+"/"
    filenames=os.listdir(path_to_geom_files)
    
    hole_punchers=obtain_perturbers_per_monte_carlo()
    column_names=make_erkal_column_names_for_pandas_DF(\
        path_to_geom_files,filenames)
    data_frame=initialize_data_frame(column_names)

    GCnames = []
    for montecarlokey in hole_punchers.keys():
        for GCname in hole_punchers[montecarlokey]:
            GCnames.append(GCname)
    uniquenames=np.unique(GCnames)    
    
    keep_filenames = []
    for uniquename in uniquenames:
        for filename in filenames:
            if uniquename in filename:
                keep_filenames.append(filename)
        
    keep_filenames    
    for filename in keep_filenames:
        geometryfile=h5py.File(path_to_geom_files+filename,'r')
        montecarlokeys=list(geometryfile.keys())
        perturber=filename.split("-")[1]

        for montecarlokey in montecarlokeys:
            dataDict    =   extract_erkal_params_with_gap_flag(\
                geometryfile,hole_punchers,perturber,montecarlokey)
            data_frame  =   append_data_frame(data_frame, dataDict)
    return data_frame

def append_data_frame(dataFrame, dataDict):
    """
    Given a dictionary, append it to the data frame.
    """
    dataFrame=pd.concat([dataFrame,pd.DataFrame([dataDict])],ignore_index=True)
    return dataFrame

def extract_erkal_params_with_gap_flag(geometryfile,hole_punchers,perturber,montecarlokey):
    """
    Given the name of a file, append the Erkal parameters to the data frame.
    """
    outdict={}
    outdict['perturber']=perturber
    outdict['monte_carlo']=montecarlokey
    outdict['gap_flag']=0
    # check if the perturber is a hole puncher
    if montecarlokey in hole_punchers.keys():
        if perturber in hole_punchers[montecarlokey]:
            outdict['gap_flag']=1
    # add the Erkal parameters to the dictionary    
    for key in geometryfile[montecarlokey]['erkal_2015_params'].keys():
        if key=="w_per":
            w_per = np.linalg.norm(geometryfile[montecarlokey]['erkal_2015_params'][key][()])
            outdict[key]=w_per
        elif key=="w_par":
            w_par = np.linalg.norm(geometryfile[montecarlokey]['erkal_2015_params'][key][()])
            outdict[key]=w_par
        else:
            outdict[key]=geometryfile[montecarlokey]['erkal_2015_params'][key][()]
    return outdict

def initialize_data_frame(column_names):
    """
    Initialize a data frame with the columns that will be used to store the results of the gap statistics.
    """
    dataframe=pd.DataFrame(columns=column_names)
    return dataframe

# get column names for the data_frame
def make_erkal_column_names_for_pandas_DF(path_to_geom_files,gap_geometry_filenames):
    """
    to compile the results of the gap statistics
    """
    
    
    myfile=h5py.File(path_to_geom_files+gap_geometry_filenames[0], 'r')
    montecarlokeys=list(myfile.keys())
    erkal_keys=list(myfile[montecarlokeys[0]]['erkal_2015_params'].keys())
    column_names=[]
    column_names.append("monte_carlo")
    column_names.append("perturber")
    column_names.append("gap_flag")
    temp_keys=erkal_keys.copy()
    for key in temp_keys:
        column_names.append(key)
    return column_names



# a function that opens the json file with gap results
def obtain_perturbers_per_monte_carlo():
    """
    Returns a dictionary with the perturbers responsible for each observed gap in the simulation.
    """
    fname=PH.base['minidata']+"monte_carlo_perturbers.json"
    with open(fname) as f:
        hole_punchers = json.load(f)
    return hole_punchers




