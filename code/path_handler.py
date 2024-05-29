"""
PURPOSE

To handle the paths of the files and directories in the project.

This code will be used to obtain base paths and also derived paths to output files. 

"""


import json
import os 


base = json.load(open('../paths.json'))


def tau_coordinates(GCname:str,montecarlokey:str):
    outpath="/scratch2/sferrone/intermediate-products/stream_density_profiles/" + GCname + "/"
    return outpath + GCname+"_time_profile_density_"+montecarlokey+".hdf5"

def stream_orbit(gcname:str,montecarlokey:str,potential_name:str,NP:int):
    return base['stream_orbit_path'] +potential_name +"/"+ gcname + "/"+ str(NP) + "/Pal5-"+montecarlokey+"-StreamOrbit.hdf5"

def orbit(montecarlokey:str,gcname:str,potential_name:str):
    return base['orbit_path'] + potential_name +"/"+ gcname + "-orbits.hdf5"

def stream_density_profiles_path(gcname:str,montecarlokey:str):
    return base['intermediate-products'] +"stream_density_profiles/"+gcname + "/"+ gcname + "_time_profile_density_" + montecarlokey + ".hdf5"