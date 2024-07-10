import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors 
from matplotlib.lines import Line2D

import escape_time_threshold as ETT
import sys 
sys.path.append("../../code/")
import data_extractors as DE




def main(config):
    plt.style.use("dark_background")
    
    stream_path     = "/scratch2/sferrone/simulations/Streams/"+config["MWpotential"]+"/"+config["GCname"]+"/"+str(config["NP"])+"/"
    stream_filename = config["GCname"] + "-streams-" + config['montecarlokey'] + ".hdf5"
    orbit_path      = "/scratch2/sferrone/simulations/Orbits/"+config["MWpotential"]+"/"+config["GCname"]+"-orbits.hdf5"    


    # load orbit 
    time_of_interest=0
    filtertime=0.3
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost=DE.get_orbit(orbit_path,config['montecarlokey'])
    current_index = np.argmin(np.abs(thost-time_of_interest))
    thost,xhost,yhost,zhost,vxhost,vyhost,vzhost=DE.filter_orbit_by_fixed_time(thost,(xhost,yhost,zhost,vxhost,vyhost,vzhost),current_index,filtertime)
    
    # load stream
    tesc,x,y,z,vx,vy,vz = ETT.load_stream(stream_path+stream_filename,config)
    
    # convert to tail coordiantes
    galactic_coordinates=(x,y,z,vx,vy,vz)
    orbit_galactic_coordinates=(thost,xhost,yhost,zhost,vxhost,vyhost,vzhost)
    time_coordinate,tail_coordinates=DE.convert_instant_to_tail_coordinates(galactic_coordinates,orbit_galactic_coordinates,time_of_interest)
    
    
    ### PREPARE THE PLOT     
    norm = colors.Normalize(vmin=tesc.min(), vmax=0)
    cmap = plt.cm.rainbow
    ### PREPARE PLOT 
    tesc_escape_only = tesc[tesc <=0]
    nbins = int(np.ceil(np.sqrt(config["NP"])))
    counts,edges = np.histogram(tesc_escape_only,bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    bincolors = cmap(norm(centers))
    
    outdir=config["outdir"]+config['MWpotential']+"/"+config['GCname']+"/"+config['montecarlokey']+"/"
    os.makedirs(outdir,exist_ok=True)    
    for i in range(config['factor_min'],config['factor_max'],config["nskip"]):
        threshold=ETT.threshold_factor(tesc, i)
        title = "{:s} {:s} {:s} {:d}".format(config['GCname'],config["MWpotential"],config['montecarlokey'],config['NP'])
        outname = "TC-{:s}-{:.4f}".format(config["GCname"],threshold)
    
        fig, ax0, ax1, ax2, ax3, caxis = set_up_gridspec()
        fig, ax0, ax1, ax2, ax3, caxis = fill_in_figure(fig, ax0, ax1, ax2, ax3, caxis, tail_coordinates, tesc_escape_only, tesc, threshold, cmap, norm, counts, edges, centers, bincolors)
        fig.suptitle(title)
        fig.tight_layout()  
        fig.savefig(outdir+outname+".png",dpi=300)
        plt.close(fig)
        print("Saved: ",outdir+outname+".png")

def fill_in_figure(fig, ax0, ax1, ax2, ax3, caxis, tail_coordinates, tesc_escape_only, tesc, threshold, cmap, norm, counts, edges, centers, bincolors):
    cond = tesc <= threshold
    ax0.scatter(tail_coordinates[0][cond],tail_coordinates[1][cond], c="r", s=1, alpha=0.5,);
    ax0.scatter(tail_coordinates[0][~cond],tail_coordinates[1][~cond], c="w", s=1,alpha=0.5);

    for i in range(len(counts)):
        ax1.bar(centers[i], counts[i], width=(edges[i+1] - edges[i]), color=bincolors[i], edgecolor='black', )
    ax0.set_xlim(tesc_escape_only.min(), -3);

    sc0=ax2.scatter(tail_coordinates[0][~cond],tail_coordinates[1][~cond], c=tesc[~cond], s=1, cmap=cmap, norm=norm);
    ax3.scatter(tail_coordinates[0][cond],tail_coordinates[1][cond], c=tesc[cond], s=1, cmap=cmap, norm=norm);

    cbar = fig.colorbar(sc0, cax=caxis)
    cbar.set_label('Escape time [s kpc / km]')


    # Create custom markers
    custom_markers = [Line2D([0], [0], color='r', marker='o', linestyle='', markersize=10),  # Adjust markersize as needed
                    Line2D([0], [0], color='w', marker='o', linestyle='', markersize=10)]  # Adjust markersize as needed

    # Create the legend with custom markers
    legend = ax0.legend(custom_markers, ["t_esc < {:.2f}".format(threshold), "t_esc > {:.2f}".format(threshold)])


    ax1.set_xlabel('Escape Time [s kpc / km]');
    ax1.set_ylabel('N');
    ax1.set_yscale('log');
    ax1.set_xlim(tesc_escape_only.min(), 0);
    ylims = ax1.get_ylim()
    ax1.vlines(threshold, ylims[0], ylims[1], color='r', linestyle='--', label='Threshold');
    ax1.set_ylim(ylims);
    word = "Threshold: {:.2f}".format(threshold)
    ax1.text(threshold, ylims[1], word, ha='left', va='top', color='r',size=15);
    for ax in [ax0,ax2,ax3]:
        ax.set_xlim(-10,10);
        ax.set_ylim(-0.5,0.5);
    ax0.set_ylabel("y' [kpc]");
    ax2.set_ylabel("y' [kpc]");
    ax3.set_ylabel("y' [kpc]");
    ax3.set_xlabel('x [kpc]');
        
    for ax in [ax2]:
        ax.set_xticklabels([]);
    
    return fig, ax0, ax1, ax2, ax3, caxis
    
def set_up_gridspec():
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 0.01], height_ratios=[1,1,1,1])
    ax0= plt.subplot(gs[0, 0])
    ax1= plt.subplot(gs[1, 0])
    ax2= plt.subplot(gs[2, 0])
    ax3= plt.subplot(gs[3, 0])
    caxis = plt.subplot(gs[1:4, 1])

    return fig, ax0, ax1, ax2, ax3,caxis


if __name__=="__main__":
    config = {
        "montecarlokey": "monte-carlo-009",
        "GCname": "Pal5",
        "MWpotential": "pouliasis2017pii-GCNBody",
        "NP": int(1e5),
        "outdir":"/scratch2/sferrone/plots/stream_analysis/questions/initial-escapers-distributions/",
        "factor_min": 1,
        "factor_max":30000,
        "nskip": 1000
    }
    main(config)