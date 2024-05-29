import numpy as np 
import os 
import h5py 
from astropy import units as u
import matplotlib.pyplot as plt





####################################################################
########### DEFAULT PARAMETERS AND GLOBAL VARIABLES ################
####################################################################
def get_global_parameters(\
    tau_max=0.240*u.kpc/(u.km/u.s), # in integration units
    density_min=1e-5, 
    density_max=1e0, 
    density_threshold = 3e-4,
    outpath_frames="/scratch2/sferrone/intermediate-products/frames/denity_tau/",\
    initial_frame = 40,
):
    return tau_max,density_min,density_max,density_threshold,outpath_frames,initial_frame


def profile_plot_params(unitTau=u.Myr):
    plot_p = {"linewidth":3,"color":"w"}
    stem_p = {
        "markerfmt": 'r',
        "linefmt": '--r'
    }
    hlines_p={
        "color":'r',
        "linestyles":'dashed'
    }
    
    tau_max,density_min,density_max,_,_,_=get_global_parameters()
    tau_max=tau_max.to(unitTau).value
    
    axis_p = {"xlabel": r"$\tau$ ["+str(unitTau)+"]",
            "ylabel": r"$\rho(\tau)$",
            "yscale": 'log',
            "ylim": (density_min,density_max),
            "xlim": (-tau_max,tau_max),}
    stem_markersize=2
    return plot_p, stem_p, hlines_p, axis_p, stem_markersize


#######################
######## LOOPS ########
#######################
def make_profile_frames(GCname,mcarlokey):
    plt.style.use('dark_background')
    ##### Get global parameters
    _,_,_,density_threshold,outpath_frames,initial_frame=get_global_parameters()
    
    ######## OBTAIN DENSITY MAP ########
    _,_,density_array,time_stamps,tau = \
        construct_2D_density_map(GCname,mcarlokey)
    
    index_from_left, index_from_right = find_stream_limits(density_array, density_threshold)
    
    unitT=u.kpc/(u.km/u.s)
    tau_Myr = unitless_convert_unit(tau,unitT,u.Myr).value
    tsamps_Gyr = unitless_convert_unit(time_stamps,unitT,u.Gyr)
    
    Nstamps, _ = density_array.shape
    sim_number = int(mcarlokey.split("-")[-1])
    for i in range(initial_frame, Nstamps):
        fig,axis=plot_1D_density_profile_frame(tau_Myr,density_array[i],density_threshold,index_from_left[i],index_from_right[i])
        current_time=tsamps_Gyr[i]
        title = "{:s} Stream -- Simulation {:d} -- Time: {:.2f} {:s}".format(GCname,sim_number,current_time.value,current_time.unit)
        axis.set_title(title)
        fig.tight_layout()
        outname = "frame-"+str(i).zfill(5)+".png"
        outfilename = outpath_frames+outname
        fig.savefig(outfilename)
        plt.close(fig)
        
        if i==initial_frame:
            print("Frame {:d} completed".format(i))
            print("Output file: ",outfilename)
        if i%100==0:
            print("Frame {:d} completed".format(i))
            print("Output file: ",outfilename)
        

########################################
########## PLOTTING FUNCTIONS ##########
########################################
def plot_1D_density_profile_frame(tau,density,threshold,left_index,right_index):

    fig,axis=plt.subplots(1,1,figsize=(10,3))
    plot_p, stem_p, hlines_p, axis_p, stem_markersize = profile_plot_params()
    axis.plot(tau,density, **plot_p)

    axis.hlines(threshold,tau[left_index],tau[right_index],**hlines_p)
    markerline, _, _ = axis.stem(tau[left_index], threshold, **stem_p)
    markerline.set_marker('o')
    markerline.set_markersize(stem_markersize)
    markerline, _, _ = axis.stem(tau[right_index], threshold, **stem_p)
    markerline.set_marker('o')
    markerline.set_markersize(stem_markersize)

    axis.set(**axis_p)
    return fig,axis


############################################################
#### COMPOSITE FUNCTIONS FOR OPENING DATA AND COMPUTING #### 
############################################################




######################################################
###################### DATA I/O ######################
######################################################
def get_input_path(GCname,mcarlokey,):
    outpath="/scratch2/sferrone/intermediate-products/stream_density_profiles/Pal5/"
    outname = GCname+"_time_profile_density_"+mcarlokey+".hdf5"
    filename=outpath+outname
    return filename


##############################################################
###################### LITTLE FUNCTIONS ######################
##############################################################
def unitless_convert_unit(quantity,inUnit,outUnit):
    quantity = quantity*inUnit
    return quantity.to(outUnit)



if __name__=="__main__":
    GCname="Pal5"
    mcarlokey="monte-carlo-009"
    make_profile_frames(GCname,mcarlokey)