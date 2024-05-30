import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors


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