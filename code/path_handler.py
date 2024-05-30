"""
PURPOSE

To handle the paths of the files and directories in the project.

This code will be used to obtain base paths and also derived paths to output files. 

"""


import json
import os 

base = json.load(open('../paths.json'))


def temporary_tau_coordiantes_folder(GCname:str, montecarlokey:str,potential_stream_orbit:str):
    path=base["temporary"] +"stream-1D-density/"+potential_stream_orbit+"/"+GCname+"/"+montecarlokey+"/"
    os.makedirs(path,exist_ok=True)
    return path

def tau_coordinates(GCname:str,montecarlokey:str,potential_stream_orbit:str):
    path = base['intermediate-products'] + "stream_tau_projections/" + potential_stream_orbit +"/"+ GCname + "/"
    name = GCname+"_tau_projections_"+montecarlokey+".hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def force_on_orbit(GCname:str,montecarlokey:str,orbit_potential_name:str):
    path = base['ForceOnOrbit'] + orbit_potential_name +"/"+ GCname + "/"
    name = GCname+"-"+montecarlokey+"-suspects-FORCEONORBIT.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def stream_orbit(GCname:str,montecarlokey:str,potential_name:str,NP:int):
    path = base['StreamOrbits'] +potential_name +"/"+ GCname + "/"+ str(NP) + "/"
    name = GCname+"-"+montecarlokey+"-StreamOrbit.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

def orbit(GCname:str,potential_name:str):
    path = base['Orbits'] + potential_name +"/"
    name = GCname + "-orbits.hdf5"
    os.makedirs(path,exist_ok=True)
    return path+name

