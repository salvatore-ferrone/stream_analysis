"""
This code is used to generate a summary plot of the number of gaps detected in each simulation.

The gap detectiosn are made my projecting the particles onto the orbit to view a 1D density profile

The 1D density profiles of a control and compare potential are used. Places where the control potential is less dense than the compare potential are considered underdensities.
These are densities are followed starting from sigmin (intended to be 1 sigma) until sigmax (intended to be 4 sigma).

In the end, we plot how many gaps are detected in each of the 50 monte carlo simulations at each threshold.



"""


import numpy as np 
import matplotlib.pyplot as plt
import os 
import sys 
import matplotlib as mpl
sys.path.append('../code/')
import gap_comparative_detections as GCD
sys.path.append("../analysis/")
import gap_profile_detections as GPD


def main(config):
    
    mystyle = "dark_background"
    plt.style.use(mystyle)
    GCname                      =   config['GCname']
    method                      =   config['method']
    montecarlokey               =   config['montecarlokey']
    baseoutpath                 =   config['base_plots'] + "/gap_profile_detections/"+ GCname + "/gap_detection_summary/"
    sig_min,sig_max,sig_step    =   config['sig_min'],config['sig_max'],config['sig_step']
    threshold_steps             =   np.arange(sig_min,sig_max+sig_step,sig_step)
    N_montecarlo                =   50
    N_thresholds                =   len(threshold_steps)
    N_gaps_array                      = np.zeros((N_montecarlo,N_thresholds))


    all_gaps    =   extract_all_gap_data(config)

    ## fill in the number of gaps per simulation at each threshold

    for ii in range(N_montecarlo):
        montecarlokey = "monte-carlo-"+str(ii).zfill(3)
        for jj in range(N_thresholds):
            threshold = threshold_steps[jj]
            for key in all_gaps[montecarlokey].keys():
                above_thres=np.where(all_gaps[montecarlokey][key]['zscores'] < -threshold)[0]
                if len (above_thres) > 0:
                    N_gaps_array[ii,jj] +=1 
                    
                    
    os.makedirs(baseoutpath,exist_ok=True)
    if method=="Log_Difference":
        outname = "{:s}-noise_thresshold_{:s}-average-number-of-gaps-per-simulation.png".format(config["method"],str(config["noise_threshold"]))
        title = "{:s} - Noise Threshold {:s}".format(config["method"],str(config["noise_threshold"]))
    else:
        outname = "{:s}-average-number-of-gaps-per-simulation.png".format(config["method"])
        title = "{:s}".format(config["method"])
    
    fig,axis = plot_summary(N_gaps_array,threshold_steps,config,sig_step)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(baseoutpath+outname,dpi=300)
    plt.close(fig)
    print("Results saved to: ",baseoutpath+outname)
        
        



def plot_summary(N_gaps_array,threshold_steps,config,sig_step):

    AXIS = {
        "ylabel": "Num. Gaps per Simulation",
        "xlabel": r"$\sigma$ threshold",
        "title": config["method"]}

    fig,axis = plt.subplots(1,1,figsize=(6,4))
    n_gaps = np.sum(N_gaps_array,axis=0)
    for ii in range(threshold_steps.size):
        axis.bar(threshold_steps[ii],n_gaps[ii]/50,width=sig_step,color="gray",zorder=10)

    axis.grid(True,alpha=0.4,zorder=-1);
    axis.set(**AXIS);
    fig.tight_layout()
    return fig,axis

    
def extract_all_gap_data(config,N_montecarlo=50):
    method  =   config['method']
    sig_min =  config['sig_min']
    centeres_min_filter=config["centeres_min_filter"]
    all_gaps        =   {}
    for jj in range(N_montecarlo):
        montecarlokey   =   "monte-carlo-"+str(jj).zfill(3)
        config["montecarlokey"]     =   montecarlokey
        fullname                    =   GCD.make_out_filename(config)

        centers,compare,control,control_potential,compare_potential=GPD.extract_density_profiles(fullname)
        if method=="Difference":
            myZscores=GPD.linear_difference_z_scores(compare,control)
        elif method=="Normalized_Difference":
            myZscores=GPD.normalized_linear_difference_z_scores(compare,control)
        elif method=="Log_Difference":
            noise_threshold = config['noise_threshold']
            myZscores,centers,compare,control,err,differences=GPD.log_difference_z_scores(centers,compare,control,noise_threshold)
        else:
            print("Method not implemented")
            raise NotImplementedError

        # myZscores                   =   GPD.linear_difference_z_scores(compare,control)
        valid_under_densty_indicies =   GPD.significant_underdensities(myZscores,sig_min,centers,centeres_min_filter=centeres_min_filter)
        gaps                        =   GCD.find_contiguous_sublists(valid_under_densty_indicies)
        all_gaps[montecarlokey] = {}
        for i in range(len(gaps)):
            key = "Candidate "+str(i)
            all_gaps[montecarlokey][key]={}
            all_gaps[montecarlokey][key]['indicies']    =   gaps[i]
            all_gaps[montecarlokey][key]['zscores']     =   myZscores[gaps[i]]
            all_gaps[montecarlokey][key]['centers']     =   centers[gaps[i]]
            all_gaps[montecarlokey][key]['color']       =   mpl.color_sequences['tab10'][np.mod(i,10)]    
    return all_gaps












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
        "sig_min": 1,
        "sig_max": 4,
        "sig_step": 0.5,
        "method": "Log_Difference",
        "noise_threshold": 500,
        "centeres_min_filter":0.01,
        "x-unit": "s kpc / km",
        "base_plots": "/scratch2/sferrone/plots/stream_analysis/analysis/",
        "data_output_path": "/scratch2/sferrone/stream_analysis/"}
    main(config)