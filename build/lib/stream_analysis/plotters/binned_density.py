import numpy as np

#####################################################################
############### Functions for making the density maps ###############
#####################################################################



def get_edges(NP,xlims,ylims):
    '''
    given the number of particles, return the number of bins
    '''
    ylen = ylims[1]-ylims[0]
    xlen = xlims[1]-xlims[0]
    print('xlen',xlen)
    print('ylen',ylen)
    
    nBins=int(np.ceil(np.sqrt(NP)))
    nYbins=int(np.ceil(nBins*np.sqrt(ylen/xlen)))
    nXbins=int(np.ceil(nBins)*np.sqrt(xlen/ylen))
    xedges=np.linspace(xlims[0],xlims[1],nXbins)
    yedges=np.linspace(ylims[0],ylims[1],nYbins)
    return xedges,yedges

def get_2D_density(X,Y,xedges,yedges):
    H, xedges, yedges = np.histogram2d(X, Y, bins=(xedges, yedges))
    xcenters,ycenters = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    XX,YY = np.meshgrid(xcenters,ycenters)
    H = H.T
    return XX,YY,H

def discard_empty_bins(XX,YY,H):
    cond = H>0
    H = H[cond]
    XX = XX[cond]
    YY = YY[cond]
    return XX,YY,H

def short_cut(NP,X,Y,xlims,ylims):
    xedges,yedges = get_edges(NP,xlims,ylims)
    XX,YY,H = get_2D_density(X,Y,xedges,yedges)
    XX,YY,H = discard_empty_bins(XX,YY,H)
    return XX,YY,H

def normalize_density_by_particle_number(H,NP):
    return H/NP

def order_by_density(XX,YY,H):
    order = np.argsort(H)
    return XX[order],YY[order],H[order]

