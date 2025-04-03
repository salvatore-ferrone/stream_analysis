"""
This module with make the frames for the Christmas tree animation.
It will color coat the tree according to when pericenter passages

it will display the 1d profile of the tree beneath it for each each group

"""

import stream_analysis as sa
import gcs 
import tstrippy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import os
from scipy import signal
import sys 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool
import re


# MASS GRID AND RADIUS GRID
SIZE_GRID = 5
N_MASS_SAMPLING = SIZE_GRID
N_RADIUS_SAMPLING = SIZE_GRID # square grid
MASS_GRID = np.logspace(4,4.8, N_MASS_SAMPLING) # in Msun
RADIUS_GRID = np.logspace(np.log10(2),np.log10(30),N_RADIUS_SAMPLING)/1000 # in kpc


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


def get_streamSnapShotsFileName(GCname, NP, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype='physical'):
    """
    Get the filename for the StreamSnapShots file based on the GC name, Monte Carlo key, mass index, and radius index.
    
    Parameters:
    ----------
    GCname : str
        Name of the globular cluster.
    montecarlokey : str
        Key for the Monte Carlo simulation.
    MASS_INDEX : int
        Index for the mass.
    RADIUS : int
        Index for the radius.
    Returns:
    -------
    str
        The formatted filename.
    """ 
    valid_types = ['index', "physical"]
    if mytype not in valid_types:
        raise ValueError("type must be one of the following: {:s}".format(str(valid_types)))
    if mytype == 'index':
        assert isinstance(MASS, int), "MASS must be an integer but was {:}".format(type(MASS))
        assert isinstance(RADIUS, int), "RADIUS must be an integer but was {:}".format(type(RADIUS))
        path=gcs.path_handler._StreamSnapShots(GCname=GCname,NP=NP,potential_env=potential_env,internal_dynamics=internal_dynamics)
        fname = "{:s}-StreamSnapShots-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS).zfill(3), str(RADIUS).zfill(3))
        fname = path + fname
    elif mytype == 'physical':
        assert isinstance(MASS, int), "MASS must be a int IN SOLAR MASSES but was {:}".format(type(MASS))
        assert isinstance(RADIUS, int), "RADIUS must be a int in PARSECS but was  {:}".format(type(RADIUS))
        fname = gcs.path_handler.StreamSnapShotsMassRadius(GCname, NP, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS)
    return fname


def grab_valid_fnames(GCname, NPs, potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype='physical'):

    """
    iterate over the NPs and check if the file exists
    Parameters
    ----------
    GCname : str
        Name of the globular cluster.
    NPs : list
        List of number of particles.
    potential_env : str
        Name of the potential environment.          
    internal_dynamics : str
        Name of the internal dynamics.
    montecarlokey : str
        Key for the Monte Carlo simulation.
    MASS : int
        Mass of the globular cluster in solar masses.
    RADIUS : int
        Half Mass Radius of the cluster in parsecs.
    mytype : str
        Type of the file name, either 'index' or 'physical'.
    Returns
    -------
    fnames : list
        List of valid file names.
    valid_NPs : list
        List of valid number of particles corresponding to the file names.
    """

    fnames = []
    valid_NPs=[]
    
    for i in range(len(NPs)):
        # get the file name
        fpath=get_streamSnapShotsFileName(GCname, NPs[i], potential_env, internal_dynamics, montecarlokey, MASS, RADIUS, mytype=mytype)
        # check if the file exists
        if os.path.exists(fpath):
            fnames.append(fpath)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fpath)
    valid_NPs=np.array(valid_NPs)
    return fnames, valid_NPs


def stack_phase_space(fnames,NPs,time_of_interest=0):
    """ assumes all fnames are valid files of the same format 
    """
    # set the indicies
    cummulative_NPs = np.cumsum(NPs)
    cummulative_NPs = np.insert(cummulative_NPs, 0, 0)
    # initiate the output arrays 
    phase_space = np.zeros((6,NPs.sum()))
    tesc=np.zeros(NPs.sum())
    snapshottime=np.zeros(len(fnames))
    for i in range(len(fnames)):
        with h5py.File(fnames[i],"r") as f:
            target_index = np.argmin(np.abs(f['time_stamps'][:]-time_of_interest))
            phase_space[:,cummulative_NPs[i]:cummulative_NPs[i+1]] = f["StreamSnapShots"][str(target_index)][:]
            tesc[cummulative_NPs[i]:cummulative_NPs[i+1]] = f['tesc'][:]
            snapshottime[i]=f['time_stamps'][target_index]
    return phase_space, tesc, snapshottime



# make the histogram on the same grid by binning subsets
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




def create_christmas_tree_plot(xT, tesc_escaped, groups, colors, bin_centers, counts_total, 
                              counts_groups, counts_leak, XORB, YORB, phase_space, escaped,
                              time_of_interest, limits=[-20, 20], figsize=(9.7, 6)):
    """
    Create a Christmas Tree plot with stream data.
    
    Parameters:
    -----------
    xT : array
        X positions in tail coordinates
    tesc_escaped : array
        Escape times for stars
    groups : list of arrays
        Boolean arrays for stars in different pericenter passage groups
    colors : array
        Color array for different groups
    bin_centers : array
        Centers of histogram bins
    counts_total : array
        Counts for all stars in histogram
    counts_groups : list of arrays
        Histogram counts for each group
    counts_leak : array
        Histogram counts for ungrouped/"leaked" stars
    XORB, YORB : arrays
        Orbital X and Y positions
    phase_space : array
        Full phase space information for stars
    escaped : array
        Boolean array of escaped stars
    time_of_interest : float
        Current time in simulation
    limits : list, optional
        X-axis limits for plots
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig, axis : matplotlib figure and axes
        The created figure and axes
    """
    # Create figure and axes
    fig, axis = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot escape time vs position
    npericenterpassages = len(groups)
    axis[0].scatter(xT, tesc_escaped, s=1, c='black', alpha=1)
    for ii in range(npericenterpassages):
        axis[0].scatter(xT[groups[ii]], tesc_escaped[groups[ii]], s=1, color=colors[ii], alpha=1)
    
    # Plot histograms
    axis[1].plot(bin_centers, counts_total, color='black', alpha=1, zorder=0, label="All stars")
    
    # Add shaded areas for each group
    floor = 0.1
    for i in range(npericenterpassages):
        alpha = (i/npericenterpassages + floor) / (1+floor)
        y = counts_groups[i]
        color = colors[i]
        
        # Set label for last group
        label = "shocked outflow" if i == npericenterpassages-1 else None
        
        # Create shaded area
        axis[1].fill_between(bin_centers, y, 0, 
                            color=color, 
                            alpha=alpha, 
                            zorder=i+1,
                            label=label)
        axis[1].plot(bin_centers, y, color=color, alpha=1, zorder=i+1)
    
    # Add inset with stream and orbit
    inset_ax = inset_axes(axis[0], 
                         width="25%",
                         height="50%",
                         loc='upper right',
                         bbox_to_anchor=(0, 0, 1, 1),
                         bbox_transform=axis[0].transAxes)
    
    inset_ax.plot(XORB, YORB, color='red', alpha=1)
    inset_ax.scatter(phase_space[0, escaped], phase_space[1, escaped], s=1, c='black', alpha=0.5)
    inset_ax.set_xlim(-20, 20)
    inset_ax.set_ylim(-20, 20)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel("X [kpc]")
    inset_ax.set_ylabel("Y [kpc]")
    
    # Add time annotation
    axis[0].text(0.05, 0.95, f"t = {time_of_interest:.2f} s kpc / km",
                transform=axis[0].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Add ungrouped stars
    axis[1].fill_between(bin_centers, counts_leak, 0, 
                        color='black', 
                        alpha=0.5, 
                        zorder=0,
                        label="Leaked stars")
    
    # Set axis properties
    axis[1].legend(frameon=False)
    axis[0].set_ylabel("Escape time [s kpc / km]")
    axis[1].set_yscale('log')
    axis[1].set_ylim(1e0, 5e3)
    axis[1].set_xlim(limits)
    axis[1].set_xlabel("X' [kpc]")
    axis[1].set_ylabel("Counts")
    
    return fig, axis




def generate_frame(i, valid_time_stamps, fnames, valid_NPs, torbit, xorbit, yorbit, zorbit, vxorbit, vyorbit, vzorbit, 
                   pericenter_passages_indices, colors, limits, baseout, title):
    """
    Generate a single frame for the Christmas tree animation.
    """
    time_of_interest = valid_time_stamps[i]
    phase_space, tesc, snapshottime = stack_phase_space(fnames, valid_NPs, time_of_interest=time_of_interest)
    
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB = sa.tailCoordinates.filter_orbit_by_dynamical_time(
        torbit,
        xorbit, yorbit, zorbit,
        vxorbit, vyorbit, vzorbit,
        time_of_interest=time_of_interest,
        nDynTimes=2,
    )

    # Grab those that escaped
    escaped = tesc > -990
    tesc_escaped = tesc[escaped]    

    # Convert to tail coordinates
    xT, yT, zT, vxT, vyT, vzT, indexes = sa.tailCoordinates.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0, escaped], phase_space[1, escaped], phase_space[2, escaped],
        phase_space[3, escaped], phase_space[4, escaped], phase_space[5, escaped],
        TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB,
        time_of_interest,
    )       

    # Obtain the groups of escaped stars that are not from the same pericenter passage
    groups = []
    cond = np.zeros(len(tesc_escaped), dtype=bool)
    for ii in range(len(pericenter_passages_indices)):
        minDex, upDex = pericenter_passages_indices[ii]
        condA = tesc_escaped > torbit[minDex]
        condB = tesc_escaped < torbit[upDex]
        condC = np.logical_and(condA, condB)
        groups.append(condC)
        cond = np.logical_or(cond, condC)
    ungroupped = ~cond

    # Create histograms
    counts_groups, bin_edges = create_histograms(xT, groups, limits)
    counts_leak, _ = np.histogram(xT[ungroupped], bins=bin_edges)
    counts_total, _ = np.histogram(xT, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])    

    # Create the plot
    fig, axis = create_christmas_tree_plot(
        xT, tesc_escaped, groups, colors, bin_centers, counts_total,
        counts_groups, counts_leak, XORB, YORB, phase_space,
        escaped, time_of_interest, limits=limits
    )
    axis[0].set_title(title, fontsize=12)
    
    # Save the figure
    figname = f"christmas_tree_{i:03d}.png"
    fig.savefig(baseout+figname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved frame {i} to {baseout+figname}")


def get_present_frame_numbers(directory):
    """
    Get a list of integers present in the file names in the given directory.
    
    Parameters:
    directory (str): Path to the directory containing the files.
    
    Returns:
    list: A sorted list of integers extracted from the file names.
    """
    frame_numbers = []
    # Regular expression to match the pattern "christmas_tree_<number>.png"
    pattern = re.compile(r'christmas_tree_(\d+)\.png')
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the integer and add it to the list
            frame_numbers.append(int(match.group(1)))
    
    # Return the sorted list of integers
    return sorted(frame_numbers)    

def main(
    GCname="Pal5",
    MWpotential="pouliasis2017pii-GCNBody",
    # NPs=np.array([4500, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000]),
    NPs = np.arange(3317,3347+1,1),
    internal_dynamics="isotropic-plummer_mass_radius_grid",
    montecarloindex=9,
    MASS=1,
    RADIUS=3,
    fnametype='physical',
    ONLYUNPROCESSED=True
):
    # Pick the color map for the tree
    mycmap = mpl.cm.hsv

    # The limits for the histogram
    limits = [-20, 20]

    # base out for the frames
    baseout="/scratch2/sferrone/plots/stream_analysis/christmastree/" + MWpotential + "/" + GCname + "/" + "MASS_"+str(MASS).zfill(3)+"_RADIUS_"+str(RADIUS).zfill(3)+"/"
    # the title for each frame
    title = "{:s} $M_{{\odot}}$"
    os.makedirs(baseout, exist_ok=True)
    # Load in the time stamps that were saved
    i = 0  # dummy NPs index
    montecarlokey = "monte-carlo-" + str(montecarloindex).zfill(3)
    # path = "/scratch2/sferrone/simulations/StreamSnapShots/" + MWpotential + "/" + GCname + "/" + str(NPs[i]) + "/" + internal_dynamics + "/"
    NP = NPs[i]
    fname=get_streamSnapShotsFileName(GCname, NP, MWpotential, internal_dynamics, montecarlokey, MASS, RADIUS, mytype=fnametype)

    # obtain valid time stamps 
    with h5py.File(fname,"r") as f:
        valid_time_stamps=f['time_stamps'][:]    
        mass=int(np.ceil(f.attrs['MASS']))
        halfmassradius=int(np.ceil(1000*f.attrs['HALF_MASS_RADIUS']))
    title = r"{:s} {:s} {:d} $M_{{\odot}}$ {:d} pc".format(GCname, MWpotential, mass,halfmassradius)        
    # Load in the orbit 
    orbitpath = "/scratch2/sferrone/simulations/Orbits/{:s}/".format(MWpotential)
    orbitfilepath = orbitpath + "{:s}-orbits.hdf5".format(GCname)
    with h5py.File(orbitfilepath, "r") as orbitfile:
        torbit = orbitfile[montecarlokey]['t'][:]
        xorbit = orbitfile[montecarlokey]['xt'][:]
        yorbit = orbitfile[montecarlokey]['yt'][:]
        zorbit = orbitfile[montecarlokey]['zt'][:]
        vxorbit = orbitfile[montecarlokey]['vxt'][:]
        vyorbit = orbitfile[montecarlokey]['vyt'][:]
        vzorbit = orbitfile[montecarlokey]['vzt'][:]

    # Stack all the simulations of the streams that were computed in parallel
    fnames, valid_NPs = grab_valid_fnames(GCname, NPs, MWpotential,  internal_dynamics, montecarlokey, MASS, RADIUS, mytype=fnametype)
    
    # Locate the pericenter passages
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

    # Set the colors for the tree
    colors = mycmap(np.linspace(0, 1, len(minima_indices)))
    
    # Use multiprocessing to generate frames
    ntimestamps = len(valid_time_stamps)
    timestampindexes= np.arange(ntimestamps)
    fignamebase = "{:s}-StreamSnapShots-{:s}_mass_{:s}_radius_{:s}".format(GCname, montecarlokey, str(MASS).zfill(3), str(RADIUS).zfill(3))
    # make some directories for this....
    if ONLYUNPROCESSED:
        # Get the present frame numbers
        present_frame_numbers = get_present_frame_numbers(baseout)
        # Remove the present frame numbers from the list of all timestamps
        absent_frame_numbers = np.delete(timestampindexes, present_frame_numbers)

        print("Number of missing frames", len(absent_frame_numbers))
        # Remove the present frame numbers from the list of all timestamp
    else:
        absent_frame_numbers= timestampindexes
    
    args = [
        (i, valid_time_stamps, fnames, valid_NPs, torbit, xorbit, yorbit, zorbit, vxorbit, vyorbit, vzorbit, 
         pericenter_passages_indices, colors, limits, baseout, title)
        for i in absent_frame_numbers
    ]
    # for arg  in args:
        # print("arg",arg)
    
    # # Use multiprocessing to generate frames
    # cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 1))  # Default to 1 if not set
    # print("CPUS", cpus) # should be 10
    # # Use multiprocessing to generate frames
    # with Pool(processes=cpus) as pool:
    #     pool.starmap(generate_frame, args)


if __name__ == "__main__":
    # find out the mass and radius of the stream
    myinput = int(sys.argv[1])
    nmass = 5
    nradius = 5
    assert myinput < nmass*nradius, "input must be less than 25"
    MASS_INDEX = myinput // nradius
    RADIUS_INDEX = myinput % nradius
    # set the mass and radius
    GCname="Pal5"
    MWpotential="pouliasis2017pii-GCNBody"
    NPs = np.arange(3317,3347+1,1)
    internal_dynamics="isotropic-plummer_mass_radius_grid"
    montecarloindex=9
    MASS=1
    RADIUS=3

    MASS = int(np.floor(MASS_GRID[MASS_INDEX]))
    print("RADIUS_GRID[RADIUS_INDEX]",RADIUS_GRID[RADIUS_INDEX])
    RADIUS = int(np.floor(1000*RADIUS_GRID[RADIUS_INDEX]))
    print("Radius", RADIUS)

    # Call the main function with the specified parameters
    main(MASS=MASS, RADIUS=RADIUS)

