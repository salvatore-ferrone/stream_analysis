import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def histogramEdges(NP,xmax,ymax):
    '''
    given the number of particles, return the number of bins
    '''
    nBins=int(np.ceil(np.sqrt(NP)))
    nYbins=int(np.ceil(nBins/np.sqrt(xmax/ymax)))
    nXbins=int(np.ceil(nBins*(xmax/ymax)))
    xedges=np.linspace(-xmax,xmax,nXbins)
    yedges=np.linspace(-ymax,ymax,nYbins)
    return xedges,yedges  


def getHistogram2d(x,y,xedges,yedges):
    '''
    given the x and y coordinates, return the histogram
    '''
    H,_,_=np.histogram2d(x,y,bins=(xedges,yedges))
    H=H.T
    cond=H>0
    xcenter,ycenter=0.5*(xedges[1:]+xedges[:-1]),0.5*(yedges[1:]+yedges[:-1])
    X,Y=np.meshgrid(xcenter,ycenter)
    X,Y=X[cond],Y[cond]
    H=H[cond]
    zsort=np.argsort(H)
    X,Y,H=X[zsort],Y[zsort],H[zsort]
    return X,Y,H


def getHistogram2dDIFF(x1,y1,x2,y2,xedges,yedges):
    '''
    given the x and y coordinates, return the histogram
    '''

    xcenter,ycenter=0.5*(xedges[1:]+xedges[:-1]),0.5*(yedges[1:]+yedges[:-1])
    X,Y=np.meshgrid(xcenter,ycenter)

    H1,_,_=np.histogram2d(x1,y1,bins=(xedges,yedges))
    H1=H1.T

    H2,_,_=np.histogram2d(x2,y2,bins=(xedges,yedges))
    H2=H2.T

    DIFF = H1 - H2

    cond=np.abs(DIFF)>0

    X,Y,DIFF=X[cond],Y[cond],DIFF[cond]
        
    return X,Y,DIFF


def plot2dHist(X,Y,H,
               xlim=10,ylim=0.5,
               scat_params={'s':1,'alpha':0.9,'cmap':'rainbow','norm':LogNorm(vmin=1e-2, vmax=1)}):
    fig=plt.figure(figsize=(11,2))
    gs=gridspec.GridSpec(1,2, width_ratios=[1,0.01])
    axis=[]
    axis.append(fig.add_subplot(gs[0]))
    axis.append(fig.add_subplot(gs[1]))
    # axis.append(fig.add_subplot(gs[2]))
    im=axis[0].scatter(X,Y,c=H,**scat_params)
    cbar=fig.colorbar(im,cax=axis[1])
    axis[0].set_xlabel("x' [kpc]")
    axis[0].set_ylabel("y' [kpc]")
    axis[0].set_xlim(-xlim,xlim)
    axis[0].set_ylim(-ylim,ylim)
    return fig,axis,im,cbar