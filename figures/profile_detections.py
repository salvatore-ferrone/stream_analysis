import numpy as np 
import stream_analysis as sa
import gcs
from gcs import path_handler as ph 
import matplotlib.pyplot as plt

outpath = "/home/sferrone/plots/stream_analysis/makeplots/profile_detections/"
def main(dataparams,hyperparams):
    
    # get data and properties for the plot 
    data        =   assemble_all_data_for_plot(dataparams,hyperparams)
    properties  =   assemle_plot_properties(dataparams,hyperparams)
    
    # get meta data
    CENTERS     = data["append1Dprofiles"][0]
    xsize       = np.mean(np.diff(CENTERS))
    BOXLEN      = hyperparams["box_length"]*xsize
    figtitle    = make_title(dataparams["montecarlokey"],hyperparams["sigma"],hyperparams["noise_factor"],BOXLEN)
    outname     = make_outname(dataparams,hyperparams)
    
    handles = gspec_double_xy_profile()
    fig,ax0,ax1,ax2,cbar_ax = handles
    fig = doplot(handles,data,properties)
    ax2.legend()
    fig.suptitle(figtitle);
    fig.savefig(outpath+outname,dpi=300)
    print("saved to ",outpath+outname)
    plt.close(fig)
    return None



def gspec_double_xy_profile():
    fig=plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1,1,3],width_ratios=[1, 1/100],wspace=0.05)
    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[1,0])
    ax2=fig.add_subplot(gs[2,0])
    cbar_ax = fig.add_subplot(gs[0:2,1])
    return fig,ax0,ax1,ax2,cbar_ax


def doplot(handles,data,properties):
    # unpack each 
    data_append2Dmaps                   = data['append2Dmaps']
    data_append1Dprofiles               = data['append1Dprofiles']
    data_add_gap_candidates_to_profiles = data['add_gap_candidates_to_profiles']
    data_add_gap_candidates_to_2D_plot  = data['add_gap_candidates_to_2D_plot']
    
    # unpack the properties
    properties_append2Dmaps                     = properties['append2Dmaps']
    properties_append1Dprofiles                 = properties['append1Dprofiles']
    properties_add_gap_candidates_to_profiles   = properties['add_gap_candidates_to_profiles']
    properties_add_gap_candidates_to_2D_plot    = properties['add_gap_candidates_to_2D_plot']
    
    
    # initialize the plot 
    fig,ax0,ax1,ax2,cbar_ax = handles
    
    append2Dmaps([fig,ax0,ax1,cbar_ax],data_append2Dmaps,properties_append2Dmaps)
    append1Dprofiles((ax2),data_append1Dprofiles,properties_append1Dprofiles)
    add_gap_candidates_to_profiles((ax2),data_add_gap_candidates_to_profiles,properties_add_gap_candidates_to_profiles)
    add_gap_candidates_to_2D_plot((ax1),data_add_gap_candidates_to_2D_plot,properties_add_gap_candidates_to_2D_plot)
    return fig


def assemble_all_data_for_plot(dataparams,hyperparams):
    

    sigma,xlims,ylims,noise_factor,box_length,do_cross_correlation,N_APPlY = unpack_hyperparams(hyperparams)        
    
    
    _,distributions,raw_profiles=extract_raw_data(dataparams,hyperparams)
    
    # unpack 
    dist2D0,dist2D1 = distributions
    x0,c0,c1 = raw_profiles
    
    # clean up the profiles
    profiles                =   sa.density_profile_gap_detections.clean_up_stream_profiles(x0,c0,c1,box_length,N_APPlY,do_cross_correlation)
    CENTERS,CONTROL,COMPARE =   profiles
    
    # DETECTION CANDIDATES 
    noise_filtered_profiles,noiselevel,candidates=sa.density_profile_gap_detections.log_difference_detection(profiles,noise_factor,sigma)
    # unpack
    centers,compare,control = noise_filtered_profiles    
    data = {}
    data["append2Dmaps"]                    = (dist2D0,dist2D1)
    data["append1Dprofiles"]                = (CENTERS,CONTROL,COMPARE,noiselevel)
    data["add_gap_candidates_to_profiles"]  = (centers,compare,control,candidates)
    data["add_gap_candidates_to_2D_plot"]   = (centers,candidates)
    return data


def assemle_plot_properties(dataparams,hyperparams):
    montecarlokey,GCname,NP,MWpotential0,MWpotential1,internal_dynamics = unpack_data_params(dataparams)
    sigma,xlims,ylims,noise_factor,box_length,do_cross_correlation,N_APPlY = unpack_hyperparams(hyperparams)    
    
    SCAT    =   sa.plotters.tailcoordinates.density_SCAT_properties(NP)
    AXIS    =   sa.plotters.tailcoordinates.xy_AXIS_properties(xlims,ylims)
    AXIS['xticks'] = []
    AXIS['xlabel'] = ''
    AXIS_p  =   {'xlim':xlims,'yscale':'log','xlabel':'x [kpc]','ylabel':'Density'}

    AX0,AX1 = AXIS.copy(),AXIS.copy()
    AX0['title'] = MWpotential0
    AX1['title'] = MWpotential1
    
    
    PROF0 = {'label':MWpotential0,'color':'k'}
    PROF1 = {'label':MWpotential1,'color':'cyan'}
    PROF2 = {'label':'Noise Threshold','color':'red'}
    properties_append1Dprofiles = (AXIS_p,PROF0,PROF1,PROF2)
    lineproperties = {'label':'Gap Candidate','zorder':0,'alpha':0.5,"color":'blue'}
    properties_add_gap_candidates_to_profiles = (lineproperties)
    vlineproperties = {'color':'blue','linestyle':'-','alpha':0.2,'zorder':0}
    properties_add_gap_candidates_to_2D_plot=(vlineproperties)
    
    properties = {}
    properties["append2Dmaps"]                  = (SCAT,AX0,AX1,{'orientation':'vertical'})
    properties["append1Dprofiles"]              = properties_append1Dprofiles
    properties["add_gap_candidates_to_profiles"] = properties_add_gap_candidates_to_profiles
    properties["add_gap_candidates_to_2D_plot"]  = properties_add_gap_candidates_to_2D_plot
    return properties


def extract_raw_data(dataparams,hyperparams):
    
    
    montecarlokey,GCname,NP,MWpotential0,MWpotential1,internal_dynamics = unpack_data_params(dataparams)
    sigma,xlims,ylims,noise_factor,box_length,do_cross_correlation,N_APPlY = unpack_hyperparams(hyperparams)
    
    tail0,tail1 = convert_both_to_tail_coordinates(dataparams)
    # unpack
    xT0,yT0,zY0,vxT0,vyT0,vzT0 = tail0
    xT1,yT1,zY1,vxT1,vyT1,vzT1 = tail1
    
    # make 2D density maps
    dist2D0                 =   make2Dhist(NP,xT0,yT0,xlims,ylims)
    dist2D1                 =   make2Dhist(NP,xT1,yT1,xlims,ylims)    
    
    # get raw 1D profiles of each 
    x0,c0                   =   sa.density_profile_gap_detections.get_profile(xT0,yT0,NP,xlims,ylims)
    x1,c1                   =   sa.density_profile_gap_detections.get_profile(xT1,yT1,NP,xlims,ylims)        
    
    # pack the data
    tails                   = (tail0,tail1)
    distributions           = (dist2D0,dist2D1)
    raw_profiles            = (x0,c0,c1)
    return tails,distributions,raw_profiles

########################################
############ little helpers ############
########################################
def make2Dhist(NP,xT,yT,xlims,ylims):
    # get the 2D profiles
    XX,YY,H  =   sa.plotters.binned_density.short_cut(NP,xT,yT,xlims,ylims)
    H        =   sa.plotters.binned_density.normalize_density_by_particle_number(H,NP)
    XX,YY,H  =   sa.plotters.binned_density.order_by_density(XX,YY,H)
    return XX,YY,H



### UNPACK THE ARGUMENTS 
def unpack_data_params(dataparams):
    montecarlokey=dataparams["montecarlokey"]
    GCname=dataparams["GCname"]
    NP=dataparams["NP"]
    MWpotential0=dataparams["MWpotential0"]
    MWpotential1=dataparams["MWpotential1"]
    internal_dynamics=dataparams["internal_dynamics"]
    return montecarlokey,GCname,NP,MWpotential0,MWpotential1,internal_dynamics


def unpack_hyperparams(hyperparams):
    sigma=hyperparams["sigma"]
    xlims=hyperparams["xlims"]
    ylims=hyperparams["ylims"]
    noise_factor=hyperparams["noise_factor"]
    box_length=hyperparams["box_length"]
    do_cross_correlation=hyperparams["do_cross_correlation"]
    N_APPlY=hyperparams["N_APPlY"]
    return sigma,xlims,ylims,noise_factor,box_length,do_cross_correlation,N_APPlY



################# LOAD THE DATA #################
def data_params_to_tailcoords(MWpotential, GCname, montecarlokey, NP, internal_dynamics):
    """Convienence function to convert a configuration to tail coordinates.

    Parameters
    ----------
    MWpotential : str
        the name of the potential
    GCname : str
        the name of the globular
    montecarlokey : str 
        the montecarlo key
    NP : int
        number of particles
    internal_dynamics : str
        key for the internal dynamics

    Returns
    -------
    (np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray)
        the tail coordinates of the stream
    """
    orbitspath=ph.GC_orbits(MWpotential=MWpotential, GCname=GCname)
    streampath=ph.old_streams(MWpotential,GCname,montecarlokey,NP)
    tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC              = gcs.extractors.GCOrbits.extract_whole_orbit(orbitspath,montecarlokey)
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB = sa.tailCoordinates.filter_orbit_by_dynamical_time(tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC,0,2)
    tesc,xp,yp,zp,vx,vy,vz = gcs.extractors.Stream.extract_old_streams(streampath,internal_dynamics,montecarlokey,NP)
    xT,yT,zY,vxT,vyT,vzT,indexes=sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vx,vy,vz,TORB,XORB,YORB,ZORB,VXORB,VYORB,VZORB)
    return xT,yT,zY,vxT,vyT,vzT


def convert_both_to_tail_coordinates(dataparams):
    """The bottle neck of the code. Convience functions for data_params_to_tailcoords

    Parameters
    ----------
    dataparams : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    montecarlokey,GCname,NP,MWpotential0,MWpotential1,internal_dynamics = unpack_data_params(dataparams)
    xT0,yT0,zY0,vxT0,vyT0,vzT0=data_params_to_tailcoords(MWpotential0, GCname, montecarlokey, NP, internal_dynamics)
    xT1,yT1,zY1,vxT1,vyT1,vzT1=data_params_to_tailcoords(MWpotential1, GCname, montecarlokey, NP, internal_dynamics)    
    tail0=(xT0,yT0,zY0,vxT0,vyT0,vzT0)
    tail1=(xT1,yT1,zY1,vxT1,vyT1,vzT1)
    return tail0,tail1


################# PLOTTING #################
def make_outname(dataparams,hyperparams):
    montecarlokey,GCname,NP,MWpotential0,MWpotential1,internal_dynamics = unpack_data_params(dataparams)
    sigma,xlims,ylims,noise_factor,box_length,do_cross_correlation,N_APPlY = unpack_hyperparams(hyperparams)        
    outname = "{:s}-{:s}-{:d}-milisigma-{:d}-noisefactor-{:d}-boxcarindexlength-shifted-{:d}.png".format(montecarlokey,MWpotential1,int(1e3*sigma),noise_factor,box_length,do_cross_correlation)
    return outname


def make_title(montecarlokey,sigma,noise_factor,BOXLEN):
    return "{:s}: {:d}-sigma dections - {:d} noise factor - {:d} pc smoothing length".format(montecarlokey,sigma,noise_factor,int(1000*BOXLEN))


def append2Dmaps(handles,data,properties):
    # unpack the main arguments
    fig,ax0,ax1,cbar_ax = handles
    dist2D0,dist2D1=data
    SCAT,AX0,AX1,cbarstuff=properties
    # unpack the data
    X0,Y0,H0=dist2D0
    X1,Y1,H1=dist2D1
    # append the 2D maps
    ax0.scatter(X0,Y0,c=H0,**SCAT)
    im=ax1.scatter(X1,Y1,c=H1,**SCAT)
    cbar=fig.colorbar(im,cax=cbar_ax,**cbarstuff)
    ax0.set(**AX0),ax1.set(**AX1);
    return None


def append1Dprofiles(handles,data,properties):
    # unpack the main arguments
    ax2 = handles
    CENTERS,CONTROL,COMPARE,noiselev = data
    AXIS_p,PROF0,PROF1,PROF2 = properties
    ax2.plot(CENTERS,CONTROL,**PROF0)
    ax2.plot(CENTERS,COMPARE,**PROF1)
    ax2.plot(CENTERS,noiselev,**PROF2)
    ax2.set(**AXIS_p);
    return None


def add_gap_candidates_to_profiles(handles,data,properties):
    axis = handles
    centers,compare,control,candidate_indexes = data
    lineproperties = properties
    axis.vlines(centers[candidate_indexes],compare[candidate_indexes],control[candidate_indexes],**lineproperties)
    return None


def add_gap_candidates_to_2D_plot(handles,data,properties):
    ax = handles
    ymin,ymax=ax.get_ylim()
    centeres,candidate_indexes = data
    ax.vlines(centeres[candidate_indexes],ymin,ymax,**properties)
    return None


if __name__ == "__main__":
    montecarlokey="monte-carlo-010"
    dataparams = {
        "montecarlokey":montecarlokey,
        "GCname":"Pal5",
        "NP":int(1e5),
        "MWpotential0":"pouliasis2017pii",
        "MWpotential1":"pouliasis2017pii-GCNBody",
        "internal_dynamics":"isotropic-plummer"
    }
    ## HYPER PARAMETERS

    hyperparams = {
        "xlims":[-10,10],
        "ylims":[-0.5,0.5],
        "noise_factor":50,
        "sigma":2,
        "box_length":20,
        "do_cross_correlation":False,
        "N_APPlY":2}
    main(dataparams,hyperparams)