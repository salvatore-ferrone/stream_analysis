# %%
import gcs
import h5py

import numpy as np 

import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib as mpl
import sys 
import os 
sys.path.append("/obs/sferrone/stream_analysis/stream_analysis/")
import tailCoordinates as TC
sys.path.append("/obs/sferrone/stream_analysis/stream_analysis/plotters/")
import binned_density as bd


def grab_valid_fnames(GCname, montecarlokey, NPs, MASS_INDEX, RADIUS_INDEX, potential_env, internal_dynamics):
    fnames = []
    valid_NPs=[]
    FNAME = "{:s}-StreamSnapShots-{:s}_mass_{:s}_radius_{:s}.hdf5".format(GCname, montecarlokey, str(MASS_INDEX).zfill(3), str(RADIUS_INDEX).zfill(3))
    for i in range(len(NPs)):
        path=gcs.path_handler._StreamSnapShots(GCname=GCname,NP=NPs[i],potential_env=potential_env,internal_dynamics=internal_dynamics)
        fpath=path+FNAME
        if os.path.exists(fpath):
            fnames.append(fpath)
            valid_NPs.append(NPs[i])
        else:
            print("file does not exist",fpath)
    valid_NPs=np.array(valid_NPs)
    return fnames,valid_NPs


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

# get the position of the GC at the escape time 
def get_position_at_escape(tGC,xGC,yGC,zGC,vxGC,vyGC,vzGC,escape_time):
    position_at_escape=np.zeros((6,len(escape_time)))
    indexes_at_escape=np.zeros(len(escape_time))
    for i in range(len(escape_time)):
        indexes_at_escape[i]=np.argmin(np.abs(tGC-escape_time[i]))
    indexes_at_escape=indexes_at_escape.astype(int)
    position_at_escape[0]=xGC[indexes_at_escape]
    position_at_escape[1]=yGC[indexes_at_escape]
    position_at_escape[2]=zGC[indexes_at_escape]
    position_at_escape[3]=vxGC[indexes_at_escape]
    position_at_escape[4]=vyGC[indexes_at_escape]
    position_at_escape[5]=vzGC[indexes_at_escape]
    return position_at_escape




def setup_plot():
    """Set up the figure, axis, and colorbar axis."""
    fig = plt.figure(figsize=(8.25 - 2, 11.75 - 4))
    GS = fig.add_gridspec(
        7, 2, height_ratios=[1 / 2, 1 / 3, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        hspace=0, wspace=0, width_ratios=[1, 0.01]
    )
    axis = [
        fig.add_subplot(GS[0]), fig.add_subplot(GS[2]), fig.add_subplot(GS[4]),
        fig.add_subplot(GS[6]), fig.add_subplot(GS[8]), fig.add_subplot(GS[10]),
        fig.add_subplot(GS[12])
    ]
    caxis = [
        fig.add_subplot(GS[1]), fig.add_subplot(GS[5]), fig.add_subplot(GS[3:7, 1])
    ]

    # Remove ticks and labels from all axes
    for ax in axis:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])

    return fig, axis, caxis


def update_axis_0(axis, caxis, xT, tesc, position_at_escape, SCAT, filter_time_1, filter_time_2, xlims, AXIS, clabel):
    """Update axis[0] with the scatter plot and horizontal lines."""
    im = axis[0].scatter(xT, tesc, c=np.abs(position_at_escape[2]), **SCAT)
    cbar = plt.colorbar(im, ax=axis[0], cax=caxis[0], ticks=np.arange(0, 15 + 3, 3))
    axis[0].hlines(filter_time_1, xlims[0], xlims[1], color="k", linestyle="-", zorder=0)
    axis[0].hlines(filter_time_2, xlims[0], xlims[1], color="k", linestyle="-", zorder=0)
    axis[0].set(**AXIS)
    cbar.set_label(clabel)


def update_axis_1(axis, bin_centers, HH1D, HH1D_before, HH1D_after, HH1D_middle, line_all, line_before, line_after, line_middle, AXIS_1D):
    """Update axis[1] with the 1D density profile."""
    axis[1].plot(bin_centers, HH1D, **line_all)
    axis[1].plot(bin_centers, HH1D_before, **line_before)
    axis[1].plot(bin_centers, HH1D_after, **line_after)
    axis[1].plot(bin_centers, HH1D_middle, **line_middle)
    axis[1].legend(frameon=False)
    axis[1].set(**AXIS_1D)


def update_axis_2(axis, caxis, xT, yT, escape_time, SCAT1, AXIS_map_tesc):
    """Update axis[2] with the density map colored by escape time."""
    im = axis[2].scatter(xT, yT, c=escape_time, **SCAT1)
    axis[2].set(**AXIS_map_tesc)
    cbar = plt.colorbar(im, ax=axis[2], cax=caxis[1], ticks=[-4, -3, -2, -1])
    cbar.set_label(r"$t_{esc}$ [s $\frac{kpc}{km}$]")


def update_axis_3(axis, caxis, XX, YY, HH, SCAT_DEN, AXIS_map_density):
    """Update axis[3] with the density map."""
    im = axis[3].scatter(XX, YY, c=HH, **SCAT_DEN)
    axis[3].set(**AXIS_map_density)
    cbar = plt.colorbar(im, ax=axis[3], cax=caxis[2], ticks=[1e1])
    cbar.set_label(r"$\rho$ [N]")


def update_axis_4(axis, XX_before, YY_before, HH_before, label_before, SCAT_DEN, AXIS_map_density):
    """Update axis[4] with the density map before impact."""
    im = axis[4].scatter(XX_before, YY_before, label=label_before, c=HH_before, **SCAT_DEN)
    axis[4].set(**AXIS_map_density)
    axis[4].legend(frameon=False)


def update_axis_5(axis, XX_middle, YY_middle, HH_middle, label_middle, SCAT_DEN, AXIS_map_density):
    """Update axis[5] with the density map during the middle time."""
    im = axis[5].scatter(XX_middle, YY_middle, label=label_middle, c=HH_middle, **SCAT_DEN)
    axis[5].set(**AXIS_map_density)
    axis[5].legend(frameon=False)


def update_axis_6(axis, XX_after, YY_after, HH_after, label_after, SCAT_DEN, AXIS_map_density, xticks):
    """Update axis[6] with the density map after impact."""
    im = axis[6].scatter(XX_after, YY_after, label=label_after, c=HH_after, **SCAT_DEN)
    axis[6].set(**AXIS_map_density)
    axis[6].legend(frameon=False)
    AXIS_map_density["xticks"] = xticks
    AXIS_map_density["xticklabels"] = xticks

def process_data_and_make_plot(
    GCname, montecarlokey, NPs, MASS, RADIUS, potential_env, internal_dynamics,
    xlims, ylims, filter_time_1, filter_time_2, time_of_interest
):
    
    IMAGE_NAME = "{:s}_{:s}_mass_{:s}_radius_{:s}".format(GCname, montecarlokey, str(MASS).zfill(3), str(RADIUS).zfill(3))
    """Process data and generate the plot for a given MASS and RADIUS."""
    # Load the file names that exist
    fnames, valid_NPs = grab_valid_fnames(GCname, montecarlokey, NPs, MASS, RADIUS, potential_env, internal_dynamics)

    # Get the mass and radius of the GC
    myfile = h5py.File(fnames[0], "r")
    MASS_ATTR = myfile.attrs['MASS']
    HALF_MASS_RADIUS = myfile.attrs['HALF_MASS_RADIUS']
    title = r"{:s} {:s} {:s} $M_\odot$ {:s} pc".format(
        GCname, montecarlokey, str(int(MASS_ATTR)), str(int(1000 * HALF_MASS_RADIUS))
    )

    # Stack the stream from many files
    phase_space, tesc, snapshottime = stack_phase_space(fnames, valid_NPs, time_of_interest)

    # Load the orbit
    fnameorbit = gcs.path_handler.GC_orbits(potential_env, GCname=GCname)
    tGC, xGC, yGC, zGC, vxGC, vyGC, vzGC = gcs.extractors.GCOrbits.extract_whole_orbit(
        fnameorbit, montecarlokey=montecarlokey
    )
    TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB = TC.filter_orbit_by_dynamical_time(
        tGC, xGC, yGC, zGC, vxGC, vyGC, vzGC, time_of_interest, nDynTimes=2
    )

    # Transform to tail coordinates
    xT, yT, zT, vxT, vyT, vzT, indexes = TC.transform_from_galactico_centric_to_tail_coordinates(
        phase_space[0], phase_space[1], phase_space[2], phase_space[3], phase_space[4], phase_space[5],
        TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB, time_of_interest
    )

    # Get only the escaped particles
    cond_escape = tesc > tesc.min()
    escape_time = tesc[cond_escape]

    # Get the positions at the escape time
    position_at_escape = get_position_at_escape(tGC, xGC, yGC, zGC, vxGC, vyGC, vzGC, escape_time)

    # Filter for before, middle, and after impact
    cond_before = escape_time < filter_time_1
    cond_time_middle = (filter_time_1 < escape_time) & (escape_time < filter_time_2)
    cond_after = escape_time > filter_time_2

    # Compute the 2D distributions
    XX, YY, HH = bd.short_cut(int(1e5), xT, yT, xlims, ylims)
    XX, YY, HH = bd.order_by_density(XX, YY, HH)
    XX_before, YY_before, HH_before = bd.short_cut(
        int(1e5), xT[cond_escape][cond_before], yT[cond_escape][cond_before], xlims, ylims
    )
    XX_before, YY_before, HH_before = bd.order_by_density(XX_before, YY_before, HH_before)
    XX_after, YY_after, HH_after = bd.short_cut(
        int(1e5), xT[cond_escape][cond_after], yT[cond_escape][cond_after], xlims, ylims
    )
    XX_after, YY_after, HH_after = bd.order_by_density(XX_after, YY_after, HH_after)
    XX_middle, YY_middle, HH_middle = bd.short_cut(
        int(1e5), xT[cond_escape][cond_time_middle], yT[cond_escape][cond_time_middle], xlims, ylims
    )
    XX_middle, YY_middle, HH_middle = bd.order_by_density(XX_middle, YY_middle, HH_middle)

    # Compute the 1D distributions
    bins = np.linspace(xlims[0], xlims[1], int(np.sqrt(len(xT))))
    HH1D, bin_edges = np.histogram(xT, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    HH1D_before, _ = np.histogram(xT[cond_escape][cond_before], bins=bins)
    HH1D_after, _ = np.histogram(xT[cond_escape][cond_after], bins=bins)
    HH1D_middle, _ = np.histogram(xT[cond_escape][cond_time_middle], bins=bins)

    # Prepare plot parameters
    label_before = r"$t_{{esc}} <$ {:.1f}".format(filter_time_1)
    label_middle = r"{:.1f} $< t_{{esc}} <$ {:.1f}".format(filter_time_1, filter_time_2)
    label_after = r"$t_{{esc}} >$ {:.1f}".format(filter_time_2)

    SCAT = {"norm": mpl.colors.Normalize(vmin=1, vmax=15), "cmap": "jet", "s": 1}
    SCAT1 = {"norm": mpl.colors.Normalize(vmin=escape_time.min(), vmax=0), "cmap": "twilight_shifted", "s": 1, "alpha": 0.5}
    SCAT_DEN = {"norm": mpl.colors.LogNorm(vmin=1, vmax=1e2), "cmap": "rainbow", "s": 1, "alpha": 0.5}

    AXIS = {"ylabel": r"$t_{esc}$ [s $\frac{kpc}{km}$]", "xlim": xlims, "ylim": [escape_time.min(), 0], "xticks": [], "yticks": [-4, -3, -2, -1, 0], "title": title}
    AXIS_1D = {"xlabel": "x [kpc]", "ylabel": "Density", "xlim": xlims, "ylim": [0.5, 1e3], "yscale": "log", "yticks": [1, 10, 100]}
    AXIS_map_tesc = {"xlabel": "x' [kpc]", "ylabel": "y' [kpc]", "xlim": xlims, "ylim": ylims, "yticks": [-0.5, 0.5], "yticklabels": [-0.5, 0.5]}
    AXIS_map_density = {"xlabel": "x' [kpc]", "ylabel": "y' [kpc]", "xlim": xlims, "ylim": ylims, "yticks": [-0.5, 0.5], "yticklabels": [-0.5, 0.5]}

    line_all = {"color": "black", "label": "All", "linestyle": "-", "linewidth": 2, "zorder": 3}
    line_before = {"color": "green", "label": label_before, "linestyle": "--", "linewidth": 2, "zorder": 10}
    line_after = {"color": "blue", "label": label_after, "linestyle": "-.", "linewidth": 2, "zorder": 10}
    line_middle = {"color": "red", "label": label_middle, "linestyle": "-.", "linewidth": 2, "zorder": 10}
    clabel = "|z| @ escape [kpc]"

    step = 3
    xticks = np.arange(xlims[0], xlims[1] + step, step)

    # Set up the plot
    fig, axis, caxis = setup_plot()

    # Update each axis
    update_axis_0(axis, caxis, xT[cond_escape], tesc[cond_escape], position_at_escape, SCAT, filter_time_1, filter_time_2, xlims, AXIS, clabel)
    update_axis_1(axis, bin_centers, HH1D, HH1D_before, HH1D_after, HH1D_middle, line_all, line_before, line_after, line_middle, AXIS_1D)
    update_axis_2(axis, caxis, xT[cond_escape], yT[cond_escape], escape_time, SCAT1, AXIS_map_tesc)
    update_axis_3(axis, caxis, XX, YY, HH, SCAT_DEN, AXIS_map_density)
    update_axis_4(axis, XX_before, YY_before, HH_before, label_before, SCAT_DEN, AXIS_map_density)
    update_axis_5(axis, XX_middle, YY_middle, HH_middle, label_middle, SCAT_DEN, AXIS_map_density)
    update_axis_6(axis, XX_after, YY_after, HH_after, label_after, SCAT_DEN, AXIS_map_density, xticks)
    fig.savefig("frames/{:s}.png".format(IMAGE_NAME), dpi=300)
    plt.close(fig)




def main():
    """Main function to iterate over MASS and RADIUS."""
    GCname = "Pal5"
    potential_env = "pouliasis2017pii-GCNBody"
    internal_dynamics = "isotropic-plummer_mass_radius_grid"
    montecarlokey = "monte-carlo-009"
    START, END, skip = 9000, 10000, 100
    NPs = np.arange(START, END + skip, skip, dtype=int)
    NPs = np.concatenate(([int(4500)], NPs))
    xlims = [-15, 15]
    ylims = [-1, 1]
    filter_time_1 = -5
    filter_time_2 = -3
    time_of_interest = 0

    cpus = mp.cpu_count()
    with mp.Pool(cpus) as pool:
        pool.starmap(
            process_data_and_make_plot,
            [
                (GCname, montecarlokey, NPs, MASS, RADIUS, potential_env, internal_dynamics, xlims, ylims, filter_time_1, filter_time_2, time_of_interest)
                for MASS in range(5)
                for RADIUS in range(5)
            ]
        )
    # for MASS in range(5):
    #     for RADIUS in range(5):
    #         arguments = (GCname, montecarlokey, NPs, MASS, RADIUS, potential_env, internal_dynamics, xlims, ylims, filter_time_1, filter_time_2, time_of_interest)

            # process_data_and_make_plot(
                # GCname, montecarlokey, NPs, MASS, RADIUS, potential_env, internal_dynamics,
                # xlims, ylims, filter_time_1, filter_time_2, time_of_interest)



if __name__ == "__main__":
    main()