"""
PURPOSE

To handle the paths of the files and directories in the project.

This code will be used to obtain base paths and also derived paths to output files. 

"""


import json
import os 

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to paths.json relative to the script directory
paths_file = os.path.join(script_dir, '../paths.json')

base = json.load(open('../paths.json'))

def temporary_impact_geometry(GCname:str,montecarlokey:str,potential:str,perturber:str):
    path = base['temporary'] + "temporary_impact_geometry/" + potential + "/" + GCname + "/" 
    name = perturber+"-"+montecarlokey+".hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def temporary_tau_coordiantes_folder(GCname:str, montecarlokey:str,potential_stream_orbit:str):
    path=base["temporary"] +"stream-1D-density/"+potential_stream_orbit+"/"+GCname+"/"+montecarlokey+"/"
    os.makedirs(path,exist_ok=True)
    return path

def tau_coordinates(GCname:str,montecarlokey:str,potential_stream_orbit:str):
    path = base['intermediate-products'] + "stream_tau_projections/" + potential_stream_orbit +"/"+ GCname + "/"
    name = GCname+"_tau_projections_"+montecarlokey+".hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def force_on_orbit(GCname:str,montecarlokey:str,potential:str):
    """ The potential is for what the orbit was integrated in, not the particles """
    path = base['ForceOnOrbit'] + potential +"/"+ GCname + "/"
    name = GCname+"-"+montecarlokey+"-suspects-FORCEONORBIT.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def stream_orbit(GCname:str,montecarlokey:str,potential:str,NP:int):
    # "/scratch2/sferrone/simulations/StreamOrbits/pouliasis2017pii-Pal5-suspects/Pal5/100000"
    path = base['StreamOrbits'] +potential +"/"+ GCname + "/"+ str(NP) + "/"
    name = GCname+"-"+montecarlokey+"-StreamOrbit.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def orbit(GCname:str,potential:str):
    path = base['Orbits'] + potential +"/"
    name = GCname + "-orbits.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def impact_geometry_results(GCname:str,montecarlokey:str,potential:str):
    path = base['impact-geometry-results'] + potential +"/"+ GCname + "/"
    name = GCname+"-"+montecarlokey+"-erkal-impact-geometry.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name
