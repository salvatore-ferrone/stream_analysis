import numpy as np
import h5py
from astropy import units as u
import os 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors 

plt.style.use("dark_background")

def main(config):

    # load data
    stream_path     = "/scratch2/sferrone/simulations/Streams/"+config["MWpotential"]+"/"+config["GCname"]+"/"+str(config["NP"])+"/"
    stream_filename = config["GCname"] + "-streams-" + config['montecarlokey'] + ".hdf5"
    tesc,x,y,z,vx,vy,vz = load_stream(stream_path+stream_filename,config)
    
    ### PREPARE THE PLOT     
    nbins = int(np.ceil(np.sqrt(config["NP"])))
    tesc_escape_only = tesc[tesc <=0]
    norm = colors.Normalize(vmin=tesc.min(), vmax=0)
    cmap = plt.cm.rainbow
    
    outdir=config["outdir"]+config['GCname']+"/"+config['montecarlokey']+"/"
    os.makedirs(outdir,exist_ok=True)
    ## PICK OUR CURRENT THRESHOLD 
    for i in range(config['factor_min'],config['factor_max'],config["nskip"]):
        threshold=threshold_factor(tesc, i)
        title = "{:s} {:s} {:s} {:d}".format(config['GCname'],config["MWpotential"],config['montecarlokey'],config['NP'])
        outname = "{:s}-xy-{:.4f}".format(config["GCname"],threshold)
    
        fig, ax0, ax1, ax2, ax3, ax4 = set_up_gridspec()
        fig, ax0, ax1, ax2, ax3, ax4 = fill_in_plot(fig,ax0, ax1, ax2, ax3, ax4, cmap, norm, tesc_escape_only, tesc,  x, y, threshold, nbins)
        fig.suptitle(title)
        fig.tight_layout()  
        fig.savefig(outdir+outname+".png",dpi=300)
        plt.close(fig)
        
        
        print("Saved: ",outdir+outname+".png")
    
def fill_in_plot(fig,ax0, ax1, ax2, ax3, ax4, cmap, norm, tesc_escape_only, tesc,  x, y, threshold, nbins):
    
    cond = tesc <= threshold
    ax1.hist(tesc_escape_only,bins=nbins)
    ax1.set_xlim(tesc.min(),tesc.min()//2)
    ylims=ax1.get_ylim()
    ax1.vlines(threshold,ylims[0],ylims[1],colors='r',)
    words = "{:.4f} s kpc / km".format(threshold)
    ax1.text(threshold,ylims[1]//2,words)

    ax0.scatter(x[cond],y[cond],s=1,color="r",alpha=0.5)
    ax0.scatter(x[~cond],y[~cond],s=1,color="w",alpha=0.5)

    xlims,ylims=ax0.get_xlim(),ax0.get_ylim()

    ## the middle plots 
    sc0 = ax2.scatter(x[~cond],y[~cond],c=tesc[~cond],cmap=cmap,norm=norm,s=1,alpha=0.5)
    sc1 = ax3.scatter(x[cond],y[cond],c=tesc[cond],cmap=cmap,norm=norm,s=1,alpha=0.5)
    cbar = fig.colorbar(sc0, cax=ax4)
    cbar.set_label('Escape time [s kpc / km]')
    ax1.set_xlabel('Escape time [s kpc / km]')




    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims)
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)

    ax2.set_ylabel('y [kpc]')
    ax2.set_xlabel('x [kpc]')
    ax3.set_xlabel('x [kpc]')
    fig.tight_layout()
    return fig, ax0, ax1, ax2, ax3, ax4
    

def set_up_gridspec():
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    # Define GridSpec layout
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])
    # Top row, one long plot
    ax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    # Bottom row, two axes of similar size
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    # Third column, skinny plot for colorbar
    ax4 = fig.add_subplot(gs[:, 2]) 
    return fig, ax0, ax1, ax2, ax3, ax4   

def threshold_factor(tesc, factor,dt = 1e4*u.yr):
    """define the threshold by a little ∆t after the minimum escape time"""
    dt=dt.to(u.s*u.kpc/u.km).value
    threshold = tesc.min() + factor*dt
    return threshold

def load_stream(filepath,config):
    with h5py.File(filepath,'r') as mystream:
        tesc=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['tesc'][:]
        x=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['x'][:]
        y=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['y'][:]
        z=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['z'][:]
        vx=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['vx'][:]
        vy=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['vy'][:]
        vz=mystream['isotropic-plummer'][str(config['NP'])][config['montecarlokey']]['vz'][:]
        ## Change the non escaped particles to positive times
        tesc[tesc==tesc.min()]=999    
    return tesc,x,y,z,vx,vy,vz

if __name__=="__main__":
    config = {
    "montecarlokey": "monte-carlo-000",
    "GCname": "NGC3201",
    "MWpotential": "pouliasis2017pii",
    "NP": int(1e5),
    "outdir":"/scratch2/sferrone/plots/stream_analysis/questions/initial-escapers-distributions/",
    "factor_min": 1,
    "factor_max":30000,
    "nskip": 1000
}
    os.makedirs(config["outdir"],exist_ok=True)
    main(config)    
