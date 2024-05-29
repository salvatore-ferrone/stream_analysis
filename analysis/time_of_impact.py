import gap_or_no_gap_statistics as GONGS
import os
import pandas as pd
import h5py
import numpy as np 


paths=GONGS.obtain_base_paths()
gap_geometry_filenames=GONGS.list_all_perturber_file_names()
myfile=h5py.File(paths['path_to_geometry']+gap_geometry_filenames[0], 'r')
GCnames=list(myfile.keys())
perturbers=GONGS.obtain_perturbers_per_monte_carlo()


