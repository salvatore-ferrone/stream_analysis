from gcs import path_handler as ph
import gcs
import os 
import h5py
import stream_analysis as sa
import numpy as np 
from astropy.io import fits
from datetime import datetime
import multiprocessing as mp
import plotly.graph_objects as go

outdir="/home/sferrone/plots/stream_analysis/makeplots/animations/plotly_force_on_orbit/"

def main(data_params,hyperparams):

    

    starttimemain=datetime.now()

    GCname,MWpotential,montecarlokey,NP,internal_dynamics = data_params

    threshold,limits,camera_distance,backgroundcolor = hyperparams


    ## load paths
    path_tau        =   ph.tauDensityMaps(GCname,MWpotential,montecarlokey,NP,internal_dynamics)
    path_FOORB      =   ph.ForceOnOrbit(GCname, MWpotential, montecarlokey, )
    path_host_orb   =   ph.GC_orbits(MWpotential, GCname)    
    
    #########################
    ##### LOAD THE DATA #####
    #########################
    # open the perturber data
    starttime=datetime.now()
    GCnames                 =   get_perturber_names(GCname)
    ts,xs,ys,zs,vxs,vys,vzs =   get_perturbing_orbits(GCnames, MWpotential, montecarlokey)
    Masses,rs               =   get_masses_and_radii(GCnames,montecarlokey) 
    # get the host orbit 
    tH,xH,yH,zH,vxH,vyH,vzH     =   gcs.extractors.GCOrbits.extract_whole_orbit(path_host_orb,montecarlokey=montecarlokey)    
    # extract 1D stream density profile and get the desired length
    time_stamps,tau_centers,tau_counts  =   extraxt_dau_density(path_tau)
    left_indexes,right_indexes          =   sa.streamLength.get_envelop_indexes(tau_counts,threshold)
    tau_left,tau_right                  =   sa.streamLength.tau_envelopes(tau_centers,left_indexes,right_indexes)
    # open the FORCE ON ORBIT data
    ax,ay,az,tau,time,tau_indexing,time_indexing,magnitude = extract_foorb_data(path_FOORB)    
    endtime=datetime.now()
    print("Data loaded in",endtime-starttime)


    # the mean rate at which Pal5 goes around the galactic pole
    thetaH=np.arctan2(yH,xH)
    theta0=thetaH[0]
    t0=tH[0]
    unwraptheta=np.unwrap(thetaH)
    meanThetaDot = (unwraptheta[-1]-unwraptheta[0])/(tH[-1]-tH[0])    
    # the size of the clusters
    sizes = np.sqrt(Masses/np.max(Masses)*100)    

    i0 = int(np.abs(time.shape[0] - time_stamps.shape[0]))


    ## plot properties that don't change
    textcolor,linecolor,perturbercolor=colored_objects(backgroundcolor)
    cone_params = {
        'colorscale': 'rainbow',
        'cmin': 0,
        'cmax': 50,
        'sizemode': 'absolute',
        'sizeref': 3,
        'showscale': True,
        'colorbar': dict(
            title=dict(text=r'g '),
            thickness=5,  # Adjust thickness
            len=0.5       # Adjust length
        )
    }    
    xaxisline = go.Scatter3d(x=[-limits, limits], y=[0, 0], z=[0, 0], marker=dict(color=linecolor, size=1),showlegend=False)
    yaxisline = go.Scatter3d(x=[0, 0], y=[-limits, limits], z=[0, 0], marker=dict(color=linecolor, size=1),showlegend=False)
    zaxisline = go.Scatter3d(x=[0, 0], y=[0, 0], z=[-limits, limits], marker=dict(color=linecolor, size=1),showlegend=False)
    xaxis=dict(
        title='',
        range=[-limits, limits],  # Set the x-axis limits
        showbackground=False,  # Remove the background pane
        showgrid=False,  # Remove the grid lines
        zeroline=False,  # Remove the zero line
        showticklabels=False,
        )
    yaxis=dict(
        title='',
        range=[-limits, limits],  # Set the y-axis limits
        showbackground=False,  # Remove the background pane
        showgrid=False,  # Remove the grid lines
        zeroline=False,  # Remove the zero line
        showticklabels=False,
        )
    zaxis=dict(
            title='',
            range=[-limits, limits],  # Set the z-axis limits
            showbackground=False,  # Remove the background pane
            showgrid=False,  # Remove the grid lines
            zeroline=False,  # Remove the zero line
            showticklabels=False

            )
    annotationX = dict(x=0.99*limits,y=0,z=0,text="x".format(limits),showarrow=False,font=dict(size=10,color=textcolor))
    annotationY = dict(x=0,y=0.99*limits,z=0,text="y".format(limits),showarrow=False,font=dict(size=10,color=textcolor))
    annotationZ = dict(x=0,y=0,z=0.99*limits,text="z".format(limits),showarrow=False,font=dict(size=10,color=textcolor))



    # prepare the loop

    ncpu=mp.cpu_count()
    print("Using",ncpu,"CPUs")
    pool=mp.Pool(ncpu)
    for i in range(i0,time_stamps.shape[0]):

        ## coordinate the data sets
        time_index_foorb = np.argmin(np.abs(time-time_stamps[i]))
        simulation_time_index = np.argmin(np.abs(ts-time_stamps[i]))    

        ## load the stream segment
        my_time_indexing = time_indexing[time_index_foorb]
        x_stream,y_stream,z_stream,ax_sub,ay_sub,az_sub,magnitude_sub=prepare_proxy_stream(tau,(xH,yH,zH),(ax[time_index_foorb],ay[time_index_foorb],az[time_index_foorb],magnitude[time_index_foorb]),tau_left[i],tau_right[i],my_time_indexing,tau_indexing)    
        # get the positions of all the perturbers at this time
        x_per,y_per,z_per = xs[:,simulation_time_index], ys[:,simulation_time_index],zs[:,simulation_time_index]    
        # get the position of the host at this time
        host=xH[simulation_time_index],yH[simulation_time_index],zH[simulation_time_index]    
        # get the current camera angle
        mytheta = meanThetaDot*(time_stamps[i]-t0) + theta0
        x_camera, y_camera, z_camera  = camera_distance*np.cos(mytheta),camera_distance*np.sin(mytheta),(1/2)*camera_distance

        stream=(x_stream,y_stream,z_stream)
        forcevectors=(x_stream,y_stream,z_stream,ax_sub,ay_sub,az_sub)
        perturber=(x_per,y_per,z_per)


        unchanging_params = [cone_params,xaxisline,yaxisline,zaxisline,xaxis,yaxis,zaxis,annotationX,annotationY,annotationZ,perturbercolor]
        camera_pos = x_camera,y_camera,z_camera
        # do_plot(i,host,stream,forcevectors,perturber,sizes,camera_pos,unchanging_params)
        pool.apply_async(do_plot,args=(i,host,stream,forcevectors,perturber,sizes,camera_pos,unchanging_params))
    print("All frames sent to pool")
    pool.close()
    pool.join()

    endtimemain=datetime.now()
    print("Total time taken",endtimemain-starttimemain)


def do_plot(i,host,stream,forcevectors,perturber,sizes,camera_pos,unchanging_params):

    cone_params,xaxisline,yaxisline,zaxisline,xaxis,yaxis,zaxis,annotationX,annotationY,annotationZ,perturbercolor = unchanging_params
    x_camera,y_camera,z_camera = camera_pos
    scene = prepare_scene(x_camera,y_camera,z_camera,xaxis,yaxis,zaxis,annotationX,annotationY,annotationZ) 

    perturber_trace,host_trace,stream_trace,cone = create_traces(host,stream,forcevectors,perturber,sizes,cone_params,perturbercolor=perturbercolor)

    fig = go.Figure()

    fig.add_trace(xaxisline)
    fig.add_trace(yaxisline)
    fig.add_trace(zaxisline)
    fig.add_trace(perturber_trace)
    fig.add_trace(host_trace)
    fig.add_trace(stream_trace)
    fig.add_trace(cone) 


    fig.update_layout(scene=scene,  
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins to make the plot area larger
        scene_aspectmode='manual',  # Set aspect mode to manual
        scene_aspectratio=dict(x=1, y=1, z=1),  # Adjust aspect ratio to make the plot area larger
        plot_bgcolor=backgroundcolor,  # Plot area background color
        paper_bgcolor=backgroundcolor,  # Entire figure background color
    )

    width = 1280  # Intermediate width of the image in pixels
    height = 720  # Intermediate height of the image in pixels
    scale = 1.5  # Intermediate scale factor to increase the resolution

    fig.write_image(outdir+"frame-"+str(i).zfill(4)+".png",width=width, height=height, scale=scale)
    print("Saved to "+outdir+"frame-"+str(i).zfill(4)+".png")

    del fig

####################################################################################
############################ PLOT PREPARATION ######################################
####################################################################################


def create_traces(host,stream,forcevectors,perturber,sizes,cone_params,perturbercolor="black"):
        x_per,y_per,z_per = perturber
        x_stream,y_stream,z_stream = stream
        xHH,yHH,zHH = host
        x_cone,y_cone,z_cone,ax_sub,ay_sub,az_sub = forcevectors
        perturber_trace = go.Scatter3d(x=x_per, y=y_per, z=z_per, mode='markers', marker=dict(size=sizes,color=perturbercolor),showlegend=False)
        host_trace = go.Scatter3d(x=[xHH], y=[yHH], z=[zHH], mode='markers', marker=dict(size=4),showlegend=False)
        stream_trace = go.Scatter3d(x=x_cone, y=y_cone, z=z_cone, mode='lines', marker=dict(size=4),showlegend=False)
        # Add 3D cones for the vectors
        cone=go.Cone(x=x_stream, y=y_stream, z=z_stream,u=ax_sub, v=ay_sub, w=az_sub,**cone_params)
        return perturber_trace,host_trace,stream_trace,cone

def prepare_scene(x_camera,y_camera,z_camera,xaxis,yaxis,zaxis,annotationX,annotationY,annotationZ):
    camera = dict(eye=dict(x=x_camera, y=y_camera, z=z_camera))
    scene=dict(
            camera=camera,
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=zaxis,
            aspectmode='cube',
            annotations=[annotationX,annotationY,annotationZ]
            )
    return scene

def colored_objects(backgroundcolor="black"):
    if backgroundcolor == "black":
        textcolor = "white"
        linecolor = "gray"
        perturbercolor = "white"
    else:
        textcolor = "black"
        linecolor = "gray"
        perturbercolor = "black"
    return textcolor,linecolor,perturbercolor


####################################################################################
############################ DOING SUBSETTING ######################################
####################################################################################

def prepare_proxy_stream(tau,hostorbit,accels,my_tau_left,my_tau_right,my_time_indexing,tau_indexing):
    """
    The convience functions down samples the FOORB to only be the lenght permitted by the tau density map.
    The function assums that the host orbit and the FOORB are indexed to the same time array.
    The function also requires that the accels are already downsampled to the time in question

    """


    # unpack the host orbit
    xH,yH,zH=hostorbit
    # unpack the accelerations
    ax,ay,az,magnitude=accels
    assert(len(ax.shape)==1)
    assert(len(ay.shape)==1)
    assert(len(az.shape)==1)

    ### THE hostorbit and accels are already indexed to the time and tau arrays
    # example 
    # xH[my_time_indexing] is the x position of the host at the time of the stream
    
    blank_dexes=np.arange(0,len(tau),1)
    leftdex=np.argmin(np.abs(tau - my_tau_left))
    rightdex=np.argmin(np.abs(tau - my_tau_right))
    cond1 = blank_dexes >= leftdex
    cond2 = blank_dexes <= rightdex
    cond = np.logical_and(cond1, cond2)
    # extract the stream-proxy data
    x_stream=xH[my_time_indexing + tau_indexing[cond]]
    y_stream=yH[my_time_indexing + tau_indexing[cond]]
    z_stream=zH[my_time_indexing + tau_indexing[cond]]

    # get the force on the stream-proxy
    ax_sub=ax[cond]
    ay_sub=ay[cond]
    az_sub=az[cond]
    magnitude_sub=magnitude[cond]
    return x_stream,y_stream,z_stream,ax_sub,ay_sub,az_sub,magnitude_sub




#####################################################################################
############################ EXTRACTING RAW DATA ####################################
#####################################################################################
def get_masses_and_radii(GCnames,montecarlokey):
    Masses,rh_mes,_,_,_,_,_,_=gcs.extractors.MonteCarloObservables.extract_all_GC_observables(GCnames,montecarlokey)
    rs = np.array([gcs.misc.half_mass_to_plummer(x).value for x in rh_mes])
    Masses = np.array([x.value for x in Masses])    
    return Masses,rs


def extract_foorb_data(infilename):
    with h5py.File(infilename, "r") as foorb:
        ax=foorb['ax'][()]
        ay=foorb['ay'][()]
        az=foorb['az'][()]
        tau=foorb['tau'][()]
        time=foorb['time'][()]
        tau_indexing=foorb['tau_indexing'][()]
        time_indexing=foorb['time_indexing'][()]
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    return ax,ay,az,tau,time,tau_indexing,time_indexing,magnitude


def get_perturbing_orbits(GCnames, MWpotential, montecarlokey):
    ts,xs,ys,zs,vxs,vys,vzs=gcs.extractors.GCOrbits.extract_orbits_from_all_GCS(GCnames, MWpotential, montecarlokey) # type: ignore
    today_index = np.argmin(np.abs(ts))
    ts=ts[0:today_index]
    xs,ys,zs=xs[:,0:today_index],ys[:,0:today_index],zs[:,0:today_index]
    vxs,vys,vzs=vxs[:,0:today_index],vys[:,0:today_index],vzs[:,0:today_index]
    return ts,xs,ys,zs,vxs,vys,vzs


def get_perturber_names(GCname):
    path_to_cluster_data="/home/sferrone/tstrippy/tstrippy/data/2023-03-28-merged.fits"
    with fits.open(path_to_cluster_data) as myfits:
        myGCnames=myfits[1].data['Cluster']
    GCnames = [str(myGCnames[i]) for i in range(len(myGCnames))]
    GCnames.pop(GCnames.index(GCname))
    return GCnames


def extraxt_dau_density(path_tau):
    with h5py.File(path_tau,'r') as myfile:
        time_stamps = myfile['time_stamps'][:]
        tau_centers = myfile['tau_centers'][:]
        tau_counts  = myfile['tau_counts'][:]
    return time_stamps,tau_centers,tau_counts



if __name__ == "__main__":
    GCname = "Pal5"
    MWpotential         =   "pouliasis2017pii-GCNBody"
    montecarlokey       =   "monte-carlo-000"
    NP = int(1e5)
    internal_dynamics = "isotropic-plummer"
    data_params = [GCname,MWpotential,montecarlokey,NP,internal_dynamics]

    # hyperparameters
    threshold = 50
    limits = 20
    camera_distance=1.5
    backgroundcolor = "black"
    hyperparams = [threshold,limits,camera_distance,backgroundcolor]

    main(data_params,hyperparams)