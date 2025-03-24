"""A module for preparing plots of the tail coordinates of a stream.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt


#########################################################################
############################## PROPERTIES ###############################
#########################################################################
def xy_AXIS_properties(xlims=[-10,10],ylims=[-1,1]):
    return {"xlim":xlims,"ylim":ylims,"xlabel":" x' [kpc]","ylabel":" y' [kpc]"}

def profile_AXIS_properties(xlims=[-10,10]):
    return  {"yscale":"log","xlabel":" x' [kpc]","ylabel":" N/NP","xlim":xlims}

def density_SCAT_properties(NP,s=3,upfactor=1e2,cmap=mpl.colormaps.get("rainbow")):
    cnorm=mpl.colors.LogNorm(vmin=1/NP, vmax=upfactor/NP)
    return {'s':s,'cmap':cmap,'norm':cnorm}



############################################################################
############################## FIGURE LAYOUTS ##############################
############################################################################
def gspec_double_xy_profile():
    fig=plt.figure(figsize=(15,6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1,1,4],width_ratios=[1, 1/100],wspace=0.05)
    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[1,0])
    ax2=fig.add_subplot(gs[2,0])
    cbar_ax = fig.add_subplot(gs[0:2,1])
    return fig,ax0,ax1,ax2,cbar_ax

def gspec_single_xy():
    fig=plt.figure(figsize=(15,1))
    gs=fig.add_gridspec(1,2,width_ratios=[1,0.01],wspace=0.02)
    ax=fig.add_subplot(gs[0,0])
    cbar_ax=fig.add_subplot(gs[0,1])
    return fig,ax,cbar_ax

def gspec_xy_and_profile():
    fig=plt.figure(figsize=(15,4))
    gs=fig.add_gridspec(2,2,height_ratios=[1,4],width_ratios=[1,0.01],wspace=0.05)
    axis0 = fig.add_subplot(gs[0,0])
    axis1 = fig.add_subplot(gs[1,0])
    cbar_ax=fig.add_subplot(gs[0,1])
    return fig,axis0,axis1,cbar_ax





def gspec_tau_gamma_and_profile(FIG={"figsize":(10,5)},GSPEC={
    'width_ratios':[1,0.03],
    'height_ratios':[1,1],
    'hspace':0.001,
    'wspace':0.01
}):
    fig=plt.figure(figsize=(10,5))
    gspec=mpl.gridspec.GridSpec(2,2,width_ratios=[1,0.03],height_ratios=[1,1],hspace=0.001,wspace=0.01)
    axis0 = fig.add_subplot(gspec[0,0])
    axis1 = fig.add_subplot(gspec[1,0])
    cbar_ax=fig.add_subplot(gspec[0,1])
    return fig,axis0,axis1,cbar_ax