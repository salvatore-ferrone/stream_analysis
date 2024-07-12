import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
import numpy as np

mystyle = "dark_background"
plt.style.use(mystyle)


def time_tau_color(XX,YY,C,
                   cmap='rainbow',
                   norm=colors.LogNorm(vmin=1e-5, vmax=1e0)):
    fig = plt.figure(figsize=(10,4))
    # Create a 1x2 grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[100, 1], figure=fig)
    # Create the main plot on the left
    ax0 = fig.add_subplot(gs[0])
    # Use a logarithmic colormap and a logarithmic normalization
    im = ax0.pcolormesh(XX, YY, C.T, cmap=cmap, norm=norm)
    # Create the colorbar on the right
    ax1 = fig.add_subplot(gs[1])
    cbar=fig.colorbar(im, cax=ax1)
    fig.tight_layout()
    return fig,ax0,ax1,im,cbar



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



def plot2dHist(X,Y,H,
               myfig={"figsize":(11,2)},
               scat_params = {"alpha":0.9, "s":1, "cmap":'rainbow', "norm":colors.LogNorm(vmin=1e-2, vmax=1)},
               myaxis={"xlim":(-10,10),"ylim":(-0.5,0.5),"xlabel":"$x'$ [kpc]","ylabel":"$y'$ [kpc]"},
               mygridSpec={"width_ratios":[1,0.01]},
               cbarlabel=r"$\sigma$ [normalized]"):
    fig=plt.figure(**myfig)
    gs=gridspec.GridSpec(1,2,**mygridSpec)
    axis=[]
    axis.append(fig.add_subplot(gs[0]))
    axis.append(fig.add_subplot(gs[1]))    
    im=axis[0].scatter(X,Y,c=H,**scat_params)
    cbar=fig.colorbar(im,cax=axis[1])
    cbar.set_label(cbarlabel)
    axis[0].set(**myaxis)
    return fig,axis,im,cbar