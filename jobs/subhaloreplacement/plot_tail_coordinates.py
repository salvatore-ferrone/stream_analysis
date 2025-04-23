# %% [markdown]
# ## Subhalo instead of GC?
# 
# - 150 experiments
#     - 6 different "subhalo" like properties in terms of mass and radius
#     - 5 different raddii for internal dynamics
#     - 5 different masses for the internal dynamics
# 
# in what cases do we see the gap? in what cases do we not percieve the gap?

import stream_analysis as sa
import gcs 
import matplotlib.pyplot as plt
import numpy as np
import yaml 
import os 
import h5py
import matplotlib as mpl
from scipy import signal
import tstrippy
import sys

# plotting params
# set the plotting params
xtailrange=[-17,17]
ytailrange=[-1,1]
counts_range=[1,1e4]
cmap_pericenter_passages=mpl.cm.hsv
basedir="/scratch2/sferrone/plots/stream_analysis/subhaloreplacement/" 

plt.rcParams.update({
    'font.size': 12,          # Basic font size
    'axes.labelsize': 12,     # Label font size
    'axes.titlesize': 12,     # Title font size
    'xtick.labelsize': 10,    # X tick font size
    'ytick.labelsize': 10,    # Y tick font size
    'legend.fontsize': 10,    # Legend font size
    'font.family': 'serif',   # Font family, similar to LaTeX default
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'serif'],
    'text.usetex': False,     # Not using LaTeX directly for compatibility
    'axes.linewidth': 1.0,    # Width of the axes lines
    'lines.linewidth': 1.5,   # Width of plotted lines
    'lines.markersize': 6,    # Size of markers
})


def get_pericenter_passages(torbit,xorbit,yorbit,zorbit):
    """
    get the characteristic time by the value of the acceleration at pericenter passages 
    this function finds when the galcentric radius is minimum
    and then it grabs two indicies about that time
    the size of the indeces is determined by the magnitude of the acceleration at that time
    """

    # prepare the escape time 
    rorbit = np.sqrt(xorbit**2 + yorbit**2 + zorbit**2)
    minima_indices, _ = signal.find_peaks(-rorbit)

    # Compute the characteristic time of the shocks
    pouliasis2017pii = tstrippy.Parsers.pouliasis2017pii()
    ax, ay, az, _ = tstrippy.potentials.pouliasis2017pii(pouliasis2017pii, xorbit[minima_indices], yorbit[minima_indices], zorbit[minima_indices])
    accel = np.sqrt(ax**2 + ay**2 + az**2)
    chartimes = np.sqrt(rorbit[minima_indices] / accel)

    # Find the upper and lower bounds of the time of interest
    pericenter_passages_indices = []
    for i in range(len(minima_indices) - 1):
        downTime = torbit[minima_indices[i]] - chartimes[i]
        upTime = torbit[minima_indices[i]] + chartimes[i]
        minDex = np.argmin(np.abs(torbit - downTime))
        upDex = np.argmin(np.abs(torbit - upTime))
        pericenter_passages_indices.append((minDex, upDex))
    return pericenter_passages_indices

def group_star_particles_by_percenter_passages(pericenter_passages_indices, tesc, torbit):
    """
    group the stars by the pericenter passages.
    The ungroupped are the escaped stars that leak out the sides 
    """
    groups = []
    cond = np.zeros(len(tesc), dtype=bool)
    for ii in range(len(pericenter_passages_indices)):
        minDex, upDex = pericenter_passages_indices[ii]
        condA = tesc > torbit[minDex]
        condB = tesc < torbit[upDex]
        condC = np.logical_and(condA, condB)
        groups.append(condC)
        cond = np.logical_or(cond, condC)
    ungroupped = ~cond
    return groups, ungroupped

def create_histograms(particles, groups, limits=[-20,20]):
    """
    Create histograms for each group of particles in the specified limits.
    Parameters:
    particles : array_like
        The data to be binned.
    groups : list of array_like BOOLEANS
        The groups of particles to be binned.
    limits : list
        The limits for the histogram bins.
    Returns:
    counts : list of array_like
        The counts for each group.
    bin_centers : array_like
        The centers of the bins.
    """
    nbins = int(np.ceil(np.sqrt(particles.shape[0])))
    bmin= limits[0]
    bmax= limits[1]
    bin_edges = np.linspace(bmin, bmax, nbins+1)
    counts = []
    for i in range(len(groups)):
        group = groups[i]
        hist, _ = np.histogram(particles[group], bins=bin_edges)
        counts.append(hist)
    return counts, bin_edges


def create_simulation_paths(perturberMassRadius, hostMassRadius, NPs, dataparams, montecarlokey):
    """
    Create simulation paths for the given parameters.

    it checks to see if the file exists at the highest resolution and then goes down
    this is because some simulations didn't finish yet 

    Parameters:
    perturberMassRadius : array_like
        Array of perturber mass and radius.
    hostMassRadius : array_like
        Array of host mass and radius.
    NPs : list
        List of particle numbers.
    dataparams : dict
        Dictionary of data parameters.
    montecarlokey : str
        Monte Carlo key.
    Returns:
    datadict : dict
        Dictionary containing simulation paths and parameters.
    """
    fileexists=False
    total=len(NPs)
    counter=0
    datadict = {}
    for i in range(perturberMassRadius.shape[0]):
        datadict[i] = {}
        datadict[i]["mass"] = perturberMassRadius[i,0]
        datadict[i]["radius"] = perturberMassRadius[i,1]
        PerturberMass = int(perturberMassRadius[i,0])
        PerturberRadius = int(perturberMassRadius[i,1])
        for k in range(hostMassRadius.shape[0]):
            datadict[i][k] = {}
            datadict[i][k]["mass"] = hostMassRadius[k,0]
            datadict[i][k]["radius"] = hostMassRadius[k,1]
            HostMass = int(hostMassRadius[k,0])
            HostRadius = int(1000*hostMassRadius[k,1])
            # try to find the simulation at high res, if it doesn't exist then try lower tes 
            counter=0
            fileexists=False
            while not fileexists and counter < total:
                NP=NPs[counter]
                simname=gcs.path_handler.StreamMassRadiusVaryPerturber(GCname=dataparams["GCname"],
                                    NP=NP,
                                    internal_dynamics=dataparams["internal_dynamics"],
                                    montecarlokey=montecarlokey,
                                    potential_env=dataparams["stream_potential"],
                                    HostMass=HostMass,
                                    HostRadius=HostRadius,
                                    PerturberName=dataparams["PerturberName"],
                                    PerturberMass=PerturberMass,
                                    PerturberRadius=PerturberRadius,)
                fileexists=os.path.exists(simname)
                counter+=1
                if fileexists:
                    datadict[i][k]["simname"] = simname
                    datadict[i][k]["NP"] = NP
            if not fileexists and counter == total:
                print("No simulation found for ",i,k)
                datadict[i][k]["simname"]=None
                datadict[i][k]["NP"]=None
    return datadict


def make_out_dir(stream_potential, GCname, internal_dynamics, montecarlokey):
    # THINKING: OF AN OUT DIR to SAVE THE PLOTS
    
    plotdir=os.path.join(basedir, stream_potential,GCname,internal_dynamics, montecarlokey)
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    return plotdir


def make_plot_name(GCname,
                   PerturberName,
                   PerturberMass,
                   PerturberRadius,
                   HostMass,
                   HostRadius):
    """
    Create a plot name based on the parameters.
    """
    return "tailPlot_{:s}_Perturber_by_{:s}_M{:d}_R{:d}_Host_M{:d}_R{:d}.png".format(
        GCname,
        PerturberName,
        int(PerturberMass),
        int(PerturberRadius),
        int(HostMass),
        int(HostRadius))



def main(perturberIndex, hostIndex):
    """
    Main function to process and plot the stream simulation data.
    Parameters:
    perturberIndex : int
        Index of the perturber in the data dictionary.
    hostIndex : int
        Index of the host in the data dictionary.
    """
    # load the host params
    hostMassRadius=np.loadtxt("hostMassRadius.txt",delimiter=",")
    # load the subhalo params
    perturberMassRadius=np.loadtxt("sub_halo_mass_radius.txt")

    # load the other data simulation params 
    with open("dataparams.yaml", "r") as f:
        dataparams = yaml.safe_load(f)
    montecarlokey="monte-carlo-{:s}".format(str(dataparams["montecarloindex"]).zfill(3))
    # not all of the simulations hae finished,
    # the "complete sims" have 99993, if one or two batches are missing then it has less
    NPs = [99993, 83326,83325, 66658]
    # make sure that its in descending order
    NPs=np.sort(NPs)[::-1]
    NPs=NPs.tolist()

    # THINKING: OF AN OUT DIR to SAVE THE PLOTS
    plotdir=make_out_dir(dataparams["stream_potential"],dataparams["GCname"],dataparams["internal_dynamics"],montecarlokey)
    plotname=make_plot_name(
        dataparams["GCname"],
        dataparams["PerturberName"],
        perturberMassRadius[perturberIndex,0],
        perturberMassRadius[perturberIndex,1],
        hostMassRadius[hostIndex,0],
        1000*hostMassRadius[hostIndex,1]
    )
    outname=os.path.join(plotdir,plotname)
    print("plot dir:\n",outname)

    datadict=create_simulation_paths(perturberMassRadius, hostMassRadius, NPs, dataparams, montecarlokey)
    


    # grab the host orbit 
    pathGCorbit=gcs.path_handler.GC_orbits(MWpotential=dataparams['GC_potential'],GCname=dataparams['GCname'])
    tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC=gcs.extractors.GCOrbits.extract_whole_orbit(pathGCorbit,montecarlokey)
    torbit=tGC
    xorbit=xGC
    yorbit=yGC
    zorbit=zGC
    tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC=sa.tailCoordinates.filter_orbit_by_dynamical_time(tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC,0,3)

    # load the data of choice
    myfile=h5py.File(datadict[perturberIndex][hostIndex]["simname"],"r")
    phase_space=myfile['phase_space']
    tesc=myfile['tesc']
    escaped=tesc>np.min(tesc)
    NP=datadict[perturberIndex][hostIndex]['NP']     
    PerturberMass=int(datadict[perturberIndex]['mass'])
    PerturberRadius=int(datadict[perturberIndex]['radius'])
    HostMass=int(datadict[perturberIndex][hostIndex]['mass'])
    HostRadius=int(datadict[perturberIndex][hostIndex]['radius']) 

    # convert to tail coordinates
    xp,yp,zp,vxp,vyp,vzp,=myfile['phase_space'][0,escaped],myfile['phase_space'][1,escaped],myfile['phase_space'][2,escaped],myfile['phase_space'][3,escaped],myfile['phase_space'][4,escaped],myfile['phase_space'][5,escaped]
    tesc=tesc[escaped]
    # xp,yp,zp,vxp,vyp,vzp,=phase_space[0],phase_space[1],phase_space[2],phase_space[3],phase_space[4],phase_space[5]
    # only do the escaped stars to make the calc faster
    xp,yp,zp,vxp,vyp,vzp,_=sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(
        xp,yp,zp,vxp,vyp,vzp,
        tORB=tGC,
        xORB=xGC,
        yORB=yGC,
        zORB=zGC,
        vxORB=vxGC,
        vyORB=vyGC,
        vzORB=vzGC,
        t0=0
    )


    # prepare the 2D map 
    X,Y,H=sa.plotters.binned_density.short_cut(NP,xp,yp,xtailrange,ytailrange)
    X,Y,H=sa.plotters.binned_density.order_by_density(X,Y,H)


    # Prepare the 1D profile for all data
    pericenter_passages_indices=get_pericenter_passages(torbit,xorbit,yorbit,zorbit)
    # coordinate with the escaped stars 
    groups, ungroupped = group_star_particles_by_percenter_passages(pericenter_passages_indices, tesc, torbit)
    # make the histograms
    counts_groups,bin_edges=create_histograms(xp,groups,limits=xtailrange)
    counts_leak,_ = np.histogram(xp[ungroupped], bins=bin_edges)
    counts_total,_= np.histogram(xp, bins=bin_edges)
    bin_centers=0.5*(bin_edges[1:]+bin_edges[:-1])
    # get the colors for each group 
    cmap_pericenter_passages=mpl.cm.hsv
    percenter_colors = cmap_pericenter_passages(np.linspace(0, 1, len(counts_groups)))



    # set the text for the plot 
    xpos=[0.025]
    ypos=[0.95]
    deltay=0.075
    text=[r"$N_p={:,d}$".format(NP),
        r"$M_{{Perturber}}={:,d}~M_\odot$".format(PerturberMass),
        r"$R_{{Perturber}}={:.0f}~pc$".format(PerturberRadius),
        r"$M_{{Host}}={:,d}~M_\odot$".format(HostMass),
        r"$R_{{Host}}={:.0f}~pc$".format(1000*HostRadius)]
    for i in range(1,len(text)):
        xpos.append(xpos[i-1])
        ypos.append(ypos[i-1]-deltay)


    norm=mpl.colors.LogNorm(vmin=1,vmax=2e2)
    SCAT = {
        "cmap":"rainbow",
        "norm":norm,
        "s":1,
        "alpha":1,
    }
    AXIS_xy = {
        "yscale":"linear",
        "xscale":"linear",
        "ylabel":"y' (kpc)",
        "xlim":xtailrange,
        "ylim":ytailrange,
        "xticks":[]
    }
    AXIS_prof = {
        "yscale":"log",
        "xscale":"linear",
        "ylabel":"Counts",
        "xlabel":"x' [kpc]",
        "xlim":xtailrange,
        "ylim":counts_range,
        "yticks":[1e1,1e2,1e3]
    }

    # begin the plot 

    # set up the figure
    fig=plt.figure(figsize=(11.7-2,(8.4-2)))
    gs=fig.add_gridspec(4, 2, height_ratios=[1, 0.5, 0.5,0.5], width_ratios=[1, 1/50], hspace=0, wspace=0)
    axis_xy=fig.add_subplot(gs[0, 0])
    axis_profile=fig.add_subplot(gs[1:4, 0])
    cax=fig.add_subplot(gs[0, 1])
    cax_shocks=fig.add_subplot(gs[2:4, 1])

    # plot the 2D density map 
    im=axis_xy.scatter(X,Y,c=H, **SCAT)
    # put the cbar on the cax
    cbar=fig.colorbar(im, cax=cax)
    # plot the profile 
    cbar.set_label("Counts")
    axis_profile.plot(bin_centers, counts_total, drawstyle='steps-mid', color='k',label='Total')
    axis_profile.plot(bin_centers, counts_leak, drawstyle='steps-mid', color='k', alpha=0.5,label="Leaked outflow")

    # Add shaded areas for each group
    floor = 0.1
    npericenterpassages=len(counts_groups)
    for i in range(npericenterpassages):
        alpha = (i/npericenterpassages + floor) / (1+floor)
        y = counts_groups[i]
        color = percenter_colors[i]
        # Set label for last group
        label = "Shocked outflow" if i == npericenterpassages-1 else None
        # Create shaded area
        axis_profile.fill_between(bin_centers, y, 0, 
                            color=color, 
                            alpha=alpha, 
                            zorder=i+1,
                            label=label)
        axis_profile.plot(bin_centers, y, color=color, alpha=1, zorder=i+1)

    # Create a ScalarMappable without plotting anything, just to add the tescape
    sm = plt.cm.ScalarMappable(
        cmap= cmap_pericenter_passages,
        norm=plt.Normalize(vmin=tesc.min(), vmax=0)
    )
    sm.set_array([])  # Needs to be set but we're not using it
    cbar_shock = fig.colorbar(sm, cax=cax_shocks)
    cbar_shock.set_label(r"$t_{esc}$ [Gyr]", fontsize=8)

    axis_xy.set(**AXIS_xy)
    axis_profile.set(**AXIS_prof)
    axis_profile.legend(loc='upper right', fontsize=8)
    for i in range(len(xpos)):
        axis_profile.text(xpos[i],ypos[i],text[i], transform=axis_profile.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left')
        
    fig.tight_layout()
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print("saved plot to ",outname)


if __name__ == "__main__":
    perturber_index = int(sys.argv[1])
    host_index = int(sys.argv[2])
    main(perturber_index,host_index)
    



