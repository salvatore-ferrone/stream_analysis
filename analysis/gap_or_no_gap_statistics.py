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

def extract_all_statistics_into_data_frame():
    
    filenames=list_all_perturber_file_names()
    paths=obtain_base_paths()
    hole_punchers=obtain_perturbers_per_monte_carlo()
    column_names=make_erkal_column_names_for_pandas_DF()
    data_frame=initialize_data_frame(column_names)
    for filename in filenames:
        montecarlokey="monte-carlo-"+filename.split('monte-carlo-')[1].split('-')[0]
        geometryfile=h5py.File(paths['path_to_geometry']+filename,'r')
        GCnames=list(geometryfile.keys())
        for GCname in GCnames:
            dataDict=extract_erkal_params_with_gap_flag(geometryfile,hole_punchers,montecarlokey,GCname)
            data_frame=append_data_frame(data_frame, dataDict)
    return data_frame

def append_data_frame(dataFrame, dataDict):
    """
    Given a dictionary, append it to the data frame.
    """
    dataFrame=pd.concat([dataFrame,pd.DataFrame([dataDict])],ignore_index=True)
    return dataFrame

def extract_erkal_params_with_gap_flag(geometryfile,hole_punchers,montecarlokey,GCname):
    """
    Given the name of a file, append the Erkal parameters to the data frame.
    """
    outdict={}
    outdict['monte_carlo']=montecarlokey
    outdict['perturber']=GCname
    outdict['gap_flag']=0
    # check if the perturber is a hole puncher
    if montecarlokey in hole_punchers.keys():
        if GCname in hole_punchers[montecarlokey]:
            outdict['gap_flag']=1
    # add the Erkal parameters to the dictionary    
    for key in geometryfile[GCname]['erkal_2015_params'].keys():
        if key=="w_per":
            w_per = np.linalg.norm(geometryfile[GCname]['erkal_2015_params'][key][()])
            outdict[key]=w_per
        elif key=="w_par":
            w_par = np.linalg.norm(geometryfile[GCname]['erkal_2015_params'][key][()])
            outdict[key]=w_par
        else:
            outdict[key]=geometryfile[GCname]['erkal_2015_params'][key][()]
    return outdict

def initialize_data_frame(column_names):
    """
    Initialize a data frame with the columns that will be used to store the results of the gap statistics.
    """
    dataframe=pd.DataFrame(columns=column_names)
    return dataframe

# get column names for the data_frame
def make_erkal_column_names_for_pandas_DF():
    """
    to compile the results of the gap statistics
    """
    paths=obtain_base_paths()
    gap_geometry_filenames=list_all_perturber_file_names()
    myfile=h5py.File(paths['path_to_geometry']+gap_geometry_filenames[0], 'r')
    GCnames=list(myfile.keys())
    erkal_keys=list(myfile[GCnames[0]]['erkal_2015_params'].keys())
    column_names=[]
    column_names.append("monte_carlo")
    column_names.append("perturber")
    column_names.append("gap_flag")
    temp_keys=erkal_keys.copy()
    for key in temp_keys:
        column_names.append(key)
    return column_names

# a function that lists all perturber results
def list_all_perturber_file_names():
    paths=obtain_base_paths()
    path_to_geometry=paths["path_to_geometry"]
    # list the valid extension 
    valid_extension = ".hdf5"
    # list all the files in the directory
    all_files = os.listdir(path_to_geometry)
    # take only those that are valid
    valid_files = [f for f in all_files if f.endswith(valid_extension)]
    return valid_files

# a function that opens the json file with gap results
def obtain_perturbers_per_monte_carlo():
    """
    Returns a dictionary with the perturbers responsible for each observed gap in the simulation.
    """
    paths = obtain_base_paths()
    with open(paths['json_guilty_perturbers']) as f:
        hole_punchers = json.load(f)
    return hole_punchers

# a function that returns the paths 
def obtain_base_paths():
    """
    Returns the base paths for the json files that contain the gap statistics.
    """
    
    paths={}
    base_GapPredictorPath="/obs/sferrone/mini-reports/GapPredictor/"
    json_guilty_perturbers = base_GapPredictorPath + "input_data/monte_carlo_perturbers.json"
    path_to_geometry = base_GapPredictorPath + "impact-geometry-results/"
    
    paths["json_guilty_perturbers"]=json_guilty_perturbers
    paths["path_to_geometry"]=path_to_geometry
    return paths


