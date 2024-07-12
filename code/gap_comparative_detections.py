"""
THIS CODE DETECTS GAPS by comparing a model that could produce gaps against a control model.

"""


import numpy as np 
import h5py 
from scipy.stats import zscore
import datetime
import sys 
import os 
sys.path.append('../code/')
import path_handler as PH # type: ignore
import data_extractors as DE # type: ignore




def compute_tau_density_profile(config):
    """
    This function takes two different simulations and compares their density profiles.
    The are projected onto their respective orbits. Then, they are binned onto a common histogram.
    
    """
    
    GCname              =   config["GCname"]
    montecarlokey       =   config["montecarlokey"]  
    control_potential   =   config["control_potential"]
    compare_potential   =   config["compare_potential"]
    NP                  =   config["NP"]
    time_of_interest    =   config["time_of_interest"]
    xmin,xmax           =   config["xlims"]

    

    control_stream,_    =   DE.get_stream(PH.stream(GCname,montecarlokey,control_potential,NP),montecarlokey)
    compare_stream,_    =   DE.get_stream(PH.stream(GCname,montecarlokey,compare_potential,NP),montecarlokey)
    
    # load orbit
    control_orbit       =   packorbits(GCname,control_potential,montecarlokey)
    compare_orbit       =   packorbits(GCname,compare_potential,montecarlokey)
    
    # put in tail coordinates
    s_control,_         =   DE.convert_instant_to_tail_coordinates(control_stream,control_orbit,time_of_interest)
    s_compare,_         =   DE.convert_instant_to_tail_coordinates(compare_stream,compare_orbit,time_of_interest)
    
    ## prepare histogram binning 
    nbins               =   int(np.ceil(np.sqrt(NP)))
    edges               =   np.linspace(xmin,xmax,nbins)
    centers,counts_control,counts_compare = build_density_profile(edges,s_control,s_compare)
    return centers,counts_control,counts_compare

    
    
    

###############################################################################################
######################################## I/O FUNCTIONS ########################################
###############################################################################################
def packorbits(GCname,potentialname,montecarlokey):
    orbit=DE.get_orbit(PH.orbit(GCname,potentialname),montecarlokey)
    t=orbit[0]
    orbit=orbit[1:]
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB = DE.filter_orbit_by_dynamical_time(t,orbit,0,2.5)
    orbit=np.array([TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB])    
    return orbit



def make_out_filename(config):
    mid_path = "gap_profile_detections/" + config["compare_potential"] + "-against-" + config["control_potential"] + "/" + config["GCname"] + "/density_profiles/" 
    fileoutname = config["GCname"] + "-" + config["montecarlokey"] + "-density_profiles.h5" 
    os.makedirs(config["data_output_path"]+mid_path,exist_ok=True)
    fullname = config["data_output_path"]+mid_path+fileoutname
    return fullname


  
def save_density_profles(fullname,config,centers,counts_control,counts_compare):
    """ SAVE THE DENSITY PROFILES TO A FILE"""
    with h5py.File(fullname,"w") as myoutfile:
        ## make the file attributes
        for key in config.keys():
            myoutfile.attrs[key] = config[key]
        myoutfile.attrs["author"]  = "Salvatore Ferrone"
        myoutfile.attrs["Creation date"] = datetime.datetime.now().isoformat()
        myoutfile.attrs['Contact'] = "salvatore.ferrone.1996@uniroma1.it"
        ## write out the data
        myoutfile["centers"] = centers
        myoutfile["compare"] = counts_compare
        myoutfile["control"] = counts_control
        # myoutfile["z_scores"] = z_scores
        # myoutfile["N_Gaps"] = len(result)
        # for i in range(len(result)):
            # myoutfile["Gap_{:d}".format(i)] = result[i]
    print("Results saved to: ",fullname)
    




###############################################################################################
####################################### DENSITY PROFILE #######################################
###############################################################################################
def build_density_profile(edges,s_control,s_compare):
    """
    Purpose:
    --------
    Given the edges, the control and compare streams, and the method, this function
    will find the places where compare is less than control. 
    
    "difference" only subtracts
    "normalized difference" subtracts and divides by the sum of the two arrays
    """
    # make the edges
    centers =   (edges[1:]+edges[:-1])/2
    # make the histograms 
    counts_control = np.histogram(s_control,bins=edges)[0]
    counts_compare = np.histogram(s_compare,bins=edges)[0]
    
    return centers,counts_control,counts_compare




#######################################################################################
####################################### GAP  :o #######################################
#######################################################################################
def find_contiguous_sublists(indices):
    """
    A gap is going to be a continguous set of under-densites, not just one
    """    
    
    # List to hold the result
    result = []
    # Temporary list to hold the current sublist of contiguous elements
    current_sublist = []
    
    for i, index in enumerate(indices):
        # If it's the first element or is contiguous with the previous one, add to the current sublist
        if i == 0 or index == indices[i - 1] + 1:
            current_sublist.append(int(index))
        else:
            # If not contiguous, add the current sublist to the result and start a new one
            result.append(current_sublist)
            current_sublist = [int(index)]
    # Don't forget to add the last sublist to the result
    result.append(current_sublist)
    
    return result

   
def compute_z_scores(method,counts_control,counts_compare):
    difference=(counts_compare-counts_control)
    if method=="Difference":
        quantity = difference
    elif method=="Normalized Difference":
        lasum = counts_control+counts_compare
        quantity = difference/lasum
    else:
        raise ValueError("Method not recognized")
    z_scores = zscore(quantity)
    
    return z_scores
      
    


if __name__=="__main__":
    config = {
        "GCname":"Pal5",
        "montecarlokey":"monte-carlo-000",
        "NP":int(1e5),
        "internal":"isotropic-plummer",
        "control_potential":"pouliasis2017pii",
        "compare_potential":"pouliasis2017pii-GCNBody",
        "time_of_interest": 0,
        "xlims": [-0.1,0.1],
        "sigmathreshold": 2,
        "method": "Normalized Difference",
        "x-coordinate": "tau",
        "x-unit": "s kpc / km",
        "data_output_path": "/scratch2/sferrone/stream_analysis/"}
