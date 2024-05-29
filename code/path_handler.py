"""
PURPOSE

To handle the paths of the files and directories in the project.

This code will be used to obtain base paths and also derived paths to output files. 

"""


import json
import os 


base = json.load(open('../paths.json'))



def orbit_path(montecarlokey:str,gcname:str,potential_name:str):
    "scratch2/sferrone/simulations/Orbits/pouliasis2017pii-GCNBody/Pal5-orbits.hdf5"
    return base['orbit_path']