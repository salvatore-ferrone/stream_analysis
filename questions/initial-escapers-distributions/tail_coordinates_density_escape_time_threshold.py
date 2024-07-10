import escape_time_threshold as ETT
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import sys 
sys.path.append("../../code/")
import StreamOrbitCoords as SOC
import data_extractors as DE
sys.path.append("../../analysis/")
import plotters



def main(config):
    NP              = config["NP"]
    # paths
    stream_path     = "/scratch2/sferrone/simulations/Streams/"+config["MWpotential"]+"/"+config["GCname"]+"/"+str(config["NP"])+"/"
    stream_filename = config["GCname"] + "-streams-" + config['montecarlokey'] + ".hdf5"
    orbit_path      = "/scratch2/sferrone/simulations/Orbits/"+config["MWpotential"]+"/"+config["GCname"]+"-orbits.hdf5"    
    
    ### RC PARAMS!
    bigFont = 18
    smallfont=14
    plt.rcParams['axes.titlesize'] = bigFont  # For the title
    plt.rcParams['axes.labelsize'] = bigFont  # For the x and y labels
    plt.rcParams['xtick.labelsize'] = smallfont  # For the x tick labels
    plt.rcParams['ytick.labelsize'] = smallfont  # For the y tick labels
    plt.rcParams['legend.fontsize'] = smallfont  # For the legend    
    plt.style.use("dark_background")
    
    
    # LOAD ORBIT 
    time_of_interest=0
    filtertime=0.3
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost=DE.get_orbit(orbit_path,config['montecarlokey'])
    current_index = np.argmin(np.abs(thost-time_of_interest))
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost=DE.filter_orbit_by_fixed_time(thost,(xhost,yhost,zhost,vxhost,vyhost,vzhost),current_index,filtertime)    
    
    # LOAD STREAM
    tesc,x,y,z,vx,vy,vz = ETT.load_stream(stream_path+stream_filename,config)    
    
    # convert to tail coordiantes
    galactic_coordinates=(x,y,z,vx,vy,vz)
    orbit_galactic_coordinates=(thost,xhost,yhost,zhost,vxhost,vyhost,vzhost)
    time_coordinate,tail_coordinates=DE.convert_instant_to_tail_coordinates(galactic_coordinates,orbit_galactic_coordinates,time_of_interest)    
    XX = tail_coordinates[0]
    YY = tail_coordinates[1]
    
    ## PREPARE THE PLOT 
    # Set parameters for the density plot
    xmax,ymax=10,0.5
    xedges,yedges=plotters.histogramEdges(NP//1,xmax,ymax)
    norm_density = colors.LogNorm(vmin=1e-5, vmax=4e-4)
    cmap_density = plt.cm.rainbow
    
    # prepare the bins for the histogram
    tesc_escape_only = tesc[tesc <=0]
    nbins = int(np.ceil(np.sqrt(NP)))
    counts,edges = np.histogram(tesc_escape_only,bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2    
    ourdir = config["outdir"]+config['MWpotential']+"/"+config['GCname']+"/"+config['montecarlokey']+"/"
    for i in range(config['factor_min'],config['factor_max'],config["nskip"]):
        threshold=ETT.threshold_factor(tesc, i)
        cond = tesc <= threshold
        # make the new density distributions
        X_new,Y_new,H_new=plotters.getHistogram2d(tail_coordinates[0][cond],tail_coordinates[1][cond],xedges,yedges)
        X_old,Y_old,H_old=plotters.getHistogram2d(tail_coordinates[0][~cond],tail_coordinates[1][~cond],xedges,yedges)
        H_old,H_new=H_old/NP,H_new/NP
        hist2D_old = (X_old,Y_old,H_old)
        hist2D_new = (X_new,Y_new,H_new)
        
        
        title = "{:s} {:s} {:s} {:d}".format(config['GCname'],config["MWpotential"],config['montecarlokey'],config['NP'])
        outname = "TC-density-{:s}-{:.4f}".format(config["GCname"],threshold)
        fig, ax0, ax1, ax2, ax3, caxis = set_up_gridspec()
        fig, ax0, ax1, ax2, ax3, caxis = fill_in_figure(fig, ax0, ax1, ax2, ax3, caxis, threshold, XX, YY, cond, tesc_escape_only, counts, edges, centers, hist2D_old, hist2D_new, cmap_density, norm_density, bigFont)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(config["outdir"]+outname+".png")
        print("Saved: ",config["outdir"]+outname+".png")
        plt.close(fig)
        
    

def set_up_gridspec():
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 0.01], height_ratios=[1,1,1,1])
    ax0= plt.subplot(gs[0, 0])
    ax1= plt.subplot(gs[1, 0])
    ax2= plt.subplot(gs[2, 0])
    ax3= plt.subplot(gs[3, 0])
    caxis = plt.subplot(gs[2:4, 1])

    return fig, ax0, ax1, ax2, ax3, caxis




def fill_in_figure(fig, ax0, ax1, ax2, ax3, caxis, threshold, XX, YY, cond, tesc_escape_only, counts, edges, centers, hist2D_old, hist2D_new, cmap_density, norm_density, cbar_label_fontsize):
    ax0.scatter(XX[cond],YY[cond], c="r", s=1, alpha=0.5,);
    ax0.scatter(XX[~cond],YY[~cond], c="w", s=1,alpha=0.5);
    # Create custom markers
    custom_markers = [Line2D([0], [0], color='r', marker='o', linestyle='', markersize=10),  # Adjust markersize as needed
                    Line2D([0], [0], color='w', marker='o', linestyle='', markersize=10)]  # Adjust markersize as needed

    # Create the legend with custom markers
    ax0.legend(custom_markers, ["t_esc < {:.2f}".format(threshold), "t_esc > {:.2f}".format(threshold)])


    for i in range(len(counts)):
        if centers[i] < threshold:
            color = "r"
        else:
            color = "w"
        ax1.bar(centers[i], counts[i], width=(edges[i+1] - edges[i]), color=color, edgecolor='black', )
    ax1.set_xlabel('Escape Time [s kpc / km]');
    ax1.set_ylabel('N');
    ax1.set_yscale('log');
    ax1.set_xlim(tesc_escape_only.min(), 0);
    ylims = ax1.get_ylim()
    ax1.vlines(threshold, ylims[0], ylims[1], color='r', linestyle='--', label='Threshold');
    ax1.set_ylim(ylims);
    word = "Threshold: {:.2f}".format(threshold)
    ax1.text(threshold, 0.6*ylims[1], word, ha='left', va='top', color='r',size=15);



    for ax in [ax0,ax2,ax3]:
        ax.set_xlim(-10,10);
        ax.set_ylim(-0.5,0.5);
    ax0.set_ylabel("y' [kpc]");
    ax2.set_ylabel("y' [kpc]");
    ax3.set_ylabel("y' [kpc]");
    ax3.set_xlabel("x' [kpc]");
    ax0.set_xlabel("x' [kpc]");
    X_old,Y_old,H_old = hist2D_old
    X_new,Y_new,H_new = hist2D_new

    sc0=ax2.scatter(X_old,Y_old,c=H_old, s=1, cmap=cmap_density, norm=norm_density);
    ax3.scatter(X_new,Y_new,c=H_new, s=1, cmap=cmap_density, norm=norm_density);

    cbar = fig.colorbar(sc0, cax=caxis)
    cbar.set_label(r'$\Sigma$ [# kpc$^{-2}$]',fontsize=cbar_label_fontsize)
    
    fig.tight_layout()
    return fig, ax0, ax1, ax2, ax3, caxis


if __name__=="__main__":
    config = {
        "montecarlokey": "monte-carlo-027",
        "GCname": "Pal5",
        "MWpotential": "pouliasis2017pii-GCNBody",
        "NP": int(1e5),
        "outdir":"/scratch2/sferrone/plots/stream_analysis/questions/initial-escapers-distributions/",
        "factor_min": 1,
        "factor_max":30000,
        "nskip": 1000
    }
    main(config)