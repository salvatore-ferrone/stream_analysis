import numpy as np 
import matplotlib.pyplot as plt
import h5py 
from scipy.stats import zscore
import os 

import sys 
sys.path.append('../code/')
import gap_comparative_detections as GCD

import matplotlib as mpl


def main(config):
    mystyle = "dark_background"
    plt.style.use(mystyle)
    fullname                    =   GCD.make_out_filename(config)
    GCname                      =   config['GCname']
    method                      =   config['method']
    montecarlokey               =   config['montecarlokey']
    centeres_min_filter         =   0.01
    sig_min,sig_max,sig_step    =   1,4,0.5
    threshold_steps             =   np.arange(sig_min,sig_max+sig_step,sig_step)
    
    centers,compare,control,control_potential,compare_potential=extract_density_profiles(fullname)


    if method=="Difference":
        myZscores=linear_difference_z_scores(compare,control)
    elif method=="Normalized_Difference":
        myZscores=normalized_linear_difference_z_scores(compare,control)
    elif method=="Log_Difference":
        noise_threshold = config['noise_threshold']
        myZscores,centers,compare,control,err,differences=log_difference_z_scores(centers,compare,control,noise_threshold)
    else:
        print("Method not implemented")
        raise NotImplementedError
    
    outdir  =   config['base_plots'] + GCname + "/" + method + "/"
    os.makedirs(outdir,exist_ok=True)

    # store the gap candidates in a dictionary
    valid_under_densty_indicies =   significant_underdensities(myZscores,sig_min,centers,centeres_min_filter=centeres_min_filter)
    gaps                        =   GCD.find_contiguous_sublists(valid_under_densty_indicies)
    my_gaps = { }
    for i in range(len(gaps)):
        key = "Candidate "+str(i)
        my_gaps[key]={}
        my_gaps[key]['indicies']=gaps[i]
        my_gaps[key]['zscores']=myZscores[gaps[i]]
        my_gaps[key]['color']=mpl.color_sequences['tab10'][np.mod(i,10)]
    for i in range(threshold_steps.shape[0]):
        threshold       =   threshold_steps[i]
        fname   =   montecarlokey + "-" + method + "-" + control_potential + "-" + compare_potential + "-sigma-{:d}.png".format(int(1000*threshold))
        figtitle = "{:s} {:s} Significant underdensities at {:d} milisigma".format(method, montecarlokey,int(1000*threshold_steps[i]))
        fig,axis     =   profile_plots(centers,control,compare,my_gaps,control_potential,compare_potential,threshold)
        fig.suptitle(figtitle)
        fig.tight_layout()
        fig.savefig(outdir+fname)
        plt.close(fig)
        print("Done with ", outdir+fname)





def profile_plots(centers,control,compare,my_gaps,control_potential,compare_potential,threshold,
                  AXIS = {\
                    "xlabel": "Tau [s kpc / km ]",
                    "ylabel": "dN/dTau",
                    "yscale": "log"}):

    fig,axis = plt.subplots(figsize=(15,5))
    axis.plot(centers,control,label=control_potential)
    axis.plot(centers,compare,label=compare_potential)
    for keys in my_gaps.keys():
        above_thres=np.where(my_gaps[keys]['zscores'] < -threshold)[0]
        valid_indexes = np.array(my_gaps[keys]['indicies'])[above_thres]
        if len(valid_indexes)==0:
            continue
        axis.scatter(centers[valid_indexes],compare[valid_indexes],color=my_gaps[keys]['color'],label=keys,zorder=10)
    axis.legend()
    axis.set(**AXIS)
    return fig,axis



def significant_underdensities(zscores,threshold,centers,centeres_min_filter=0.01):
    """ Reject candidates outside of the center of mass of the cluster"""
    under_densities = np.where(zscores < -threshold)[0]
    valid_under_densty_indicies = []
    for i in range(len(under_densities)):
        if np.abs(centers[under_densities[i]]) > centeres_min_filter:
            valid_under_densty_indicies.append(under_densities[i])
    return valid_under_densty_indicies


def log_difference_z_scores(centers,compare,control,noise_threshold):
    """ Compute the log difference between compare and control and return the zscored values"""
    cond1,cond2=control>0,compare>0
    cond = cond1 & cond2
    centers_,compare_,control_=centers[cond],compare[cond],control[cond]
    ### compute the noise only based on the control 
    err_ = (1/np.log(10))*(1/np.sqrt(control_))
    differences = np.log10(compare_) - np.log10(control_)
    cond_filter = control_ > noise_threshold*err_
    myZscores = zscore(differences[cond_filter])
    return myZscores,centers_[cond_filter],compare_[cond_filter],control_[cond_filter],err_,differences[cond_filter]


def normalized_linear_difference_z_scores(compare,control):
    differences = compare - control
    thesum = (compare+control)/2
    normalized_differences = differences / thesum
    myZscores = zscore(normalized_differences)
    return myZscores


def linear_difference_z_scores(compare,control):
    differences = compare - control
    myZscores = zscore(differences)
    return myZscores



def extract_density_profiles(fullname):
    with h5py.File(fullname,"r") as myresults:
        centers = myresults['centers'][()]
        compare = myresults['compare'][()]
        control = myresults['control'][()]
        control_potential = myresults.attrs['control_potential']
        compare_potential = myresults.attrs['compare_potential']    
    return centers,compare,control,control_potential,compare_potential



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
    "method": "Log_Difference",
    "noise_threshold": 10,
    "x-coordinate": "tau",
    "x-unit": "s kpc / km",
    "base_plots": "/scratch2/sferrone/plots/stream_analysis/analysis/gap_profile_detections/",
    "base_output": "/scratch2/sferrone/stream_analysis/"}

    for i in range(50):
        config['montecarlokey'] = "monte-carlo-" + str(i).zfill(3)
        main(config)