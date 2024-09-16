"""
This module was created to help answer the question, which globular cluster is responsible for the perturbation of the stream? 
    i.e. who caused the gap? 
This module contains functions for calculating the acceleration on an orbit due to perturbing masses.
We are using the orbit as a proxy for the force on the stream.
The philosophy is to downsample so that the output data is smaller. 


IN A NUT SHELL:

1. Given a host orbit, estimate the size of the stream by some factor of the dynamical time
2. Use the points from the orbit ahead and behind the mass to approximate the stellar stream
3. Down sample the time and tau indices to reduce the data size
4. Compute the acceleration on the stream due to each of the perturbing masses.
5. Return 
    a. The components of the total acceleration from all perturbers
    b. The magnitude of the acceleration for each perturber at each time step
    note. This way we can
        i. see which perturber is responsible for the gap at a given time
        ii. have the acceleration vectors for the animations I want to make to verify the computation

6. Write the file!

"""

import numpy as np 
# import tstrippy
import h5py
import datetime

### This is the main function that calculates the acceleration on an orbit due to perturbing masses
def compute_force_from_all_gcs(G,Masses,orbit_perturber,orbit_host,indexing,outputs):
    """
    Calculate the acceleration on an orbit due to perturbing masses.

    Parameters
    ----------
    G : float
        Gravitational constant.
    Masses : array_like
        Array of masses of the perturbing objects.
    orbit_perturber : tuple of array_like
        Tuple containing the x, y, and z coordinates of the perturbing objects' orbits.
    orbit_host : tuple of array_like
        Tuple containing the x, y, and z coordinates of the host object's orbit.
    indexing : tuple of array_like
        Tuple containing the time indexing and tau indexing arrays.
    outputs : tuple of array_like
        Tuple containing the arrays for the x, y, and z components of acceleration and the magnitude of acceleration.

    Returns
    -------
    ax : array_like
        Array of x components of the net acceleration.
    ay : array_like
        Array of y components of the net acceleration.
    az : array_like
        Array of thez components of the net acceleration.
    magnitude : array_like
        Array of magnitudes of the acceleration for each perturbing mass at each time step.

    Notes
    -----
    The function assumes that the lengths of `orbit_perturber` and `orbit_host` are 3, 
    the length of `indexing` is 2, and the length of `outputs` is 4.

    The function iterates over the time steps and calculates the acceleration due to 
    each perturbing mass at each time step, summing the contributions to the total 
    acceleration components and magnitude.

    Several assertions are included to ensure the input arrays have the expected shapes 
    and lengths.
    """
    assert(len(orbit_perturber)==3)
    assert(len(orbit_host)==3)
    assert(len(indexing)==2)
    assert(len(outputs)==4)
    # unpack the inputs
    xs,ys,zs = orbit_perturber
    x,y,z = orbit_host
    time_indexing,tau_indexing = indexing
    ax,ay,az,magnitude = outputs

    # get the size of each variable 
    nGCs,ntime,ntau = magnitude.shape
    
    # check some more assertions
    assert(nGCs==len(Masses))
    assert(len(time_indexing)==ntime)
    assert(len(tau_indexing)==ntau)
    assert(len(x)==len(y)==len(z))
    
    assert(len(xs.shape)==2)
    assert(len(ys.shape)==2)
    assert(len(zs.shape)==2)
    
    assert(zs.shape[1]==xs.shape[1]==ys.shape[1])
    assert(xs.shape[0]==nGCs)
    assert(ys.shape[0]==nGCs)
    assert(zs.shape[0]==nGCs)
    
    assert len(x.shape)==1, "x should be a 1D array"
    assert len(y.shape)==1, "y should be a 1D array"
    assert len(z.shape)==1, "z should be a 1D array"
    assert len(x) > time_indexing[-1]+tau_indexing[-1], "the host orbit should be longer than the time and tau indexing"


    axtemp,aytemp,aztemp = np.zeros(ntau),np.zeros(ntau),np.zeros(ntau)
    for i in range(ntime):
        # get the sampling positions and orbits at the current time
        current_time_index = time_indexing[i]
        xsampling = x[current_time_index+tau_indexing]
        ysampling = y[current_time_index+tau_indexing]
        zsampling = z[current_time_index+tau_indexing]
        j = 0 # GC index
        for j in range(nGCs):
            axtemp,aytemp,aztemp,_=pointmassconfiguration(G,[Masses[j]],[xs[j,current_time_index]],[ys[j,current_time_index]],[zs[j,current_time_index]],xsampling,ysampling,zsampling)
            magnitude[j,i,:] = np.sqrt(axtemp**2 + aytemp**2 + aztemp**2) # magnitude of the acceleration per GC
            ax[i]+=axtemp  # vector sum of each component
            ay[i]+=aytemp  # vector sum of each component
            az[i]+=aztemp  # vector sum of each component
    return ax,ay,az,magnitude



def pointmassconfiguration(G,Masses,xs,ys,zs,x,y,z):
    """
    Calculate the acceleration due to a point mass at a given position.

    Parameters
    ----------
    G : float
        Gravitational constant.
    Masses : array_like
        Array of masses of the perturbing objects.
    xs : array_like
        Array of x coordinates of the perturbing objects.
    ys : array_like
        Array of y coordinates of the perturbing objects.
    zs : array_like
        Array of z coordinates of the perturbing objects.
    x : array_like
        Array of x coordinates of the host object.
    y : array_like
        Array of y coordinates of the host object.
    z : array_like
        Array of z coordinates of the host object.

    Returns
    -------
    ax : array_like
        Array of x components of the acceleration.
    ay : array_like
        Array of y components of the acceleration.
    az : array_like
        Array of thez components of the acceleration.
    magnitude : array_like
        Array of magnitudes of the acceleration.

    Notes
    -----
    The function assumes that the lengths of `Masses`, `xs`, `ys`, `zs`, `x`, `y`, and `z` are all the same.

    The function calculates the acceleration due to a point mass at each position in the host object's orbit.
    """
    assert(len(Masses)==len(xs)==len(ys)==len(zs))
    assert(len(x)==len(y)==len(z))
    ax,ay,az = np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))
    phi = np.zeros(len(x))
    for j in range(len(Masses)):
        dx = xs[j]-x
        dy = ys[j]-y
        dz = zs[j]-z
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        dr3 = dr**3
        ax += G*Masses[j]*dx/dr3
        ay += G*Masses[j]*dy/dr3
        az += G*Masses[j]*dz/dr3
        phi -= G*Masses[j]/dr
    return ax,ay,az,phi

def write_file(outfilename,attrs,xycoords,xyindexing,accelerations,magnitude,perturbers):
    """
    Write the output data to a file.
    """
    GCname,MWpotential,montecarlokey,hostorbitfilename,tdyn,side_length_factor,ntauskip,ntskip = attrs
    time,tau = xycoords
    time_indexing,tau_indexing=xyindexing
    ax,ay,az = accelerations
    magnitude = magnitude
    GCnames,Masses,rs= perturbers
    with h5py.File(outfilename,"w") as myfile:
        myfile.attrs["GCname"] = GCname
        myfile.attrs["MWpotential"] = MWpotential
        myfile.attrs["montecarlokey"] = montecarlokey
        myfile.attrs["hostorbitfilename"] = hostorbitfilename
        myfile.attrs["dynamic_time"] = tdyn
        myfile.attrs["side_length_factor"] = side_length_factor
        myfile.attrs["ntauskip"] = ntauskip
        myfile.attrs["ntskip"] = ntskip
        myfile.attrs["date"] = str(datetime.datetime.now())
        myfile.attrs["author"] = "Salvatore Ferrone"
        myfile.attrs["email"] = "salvatore.ferrone@uniroma1.it"
        myfile.create_dataset("time",data=time)
        myfile.create_dataset("tau",data=tau)
        myfile.create_dataset("time_indexing",data=time_indexing)
        myfile.create_dataset("tau_indexing",data=tau_indexing)
        myfile.create_dataset("ax",data=ax)
        myfile.create_dataset("ay",data=ay)
        myfile.create_dataset("az",data=az)
        myfile.create_dataset("magnitude",data=magnitude)
        myfile.create_dataset("Perturbers",data=GCnames)
        myfile.create_dataset("Masses",data=Masses)
        myfile.create_dataset("rs",data=rs)
    return None


def median_dynamical_time(x,y,z,vx,vy,vz):
    """Return the median dynamical estiamted as the radius divided by the velocity for a given orbit.
    """
    assert(x.shape==y.shape==z.shape==vx.shape==vy.shape==vz.shape)
    assert(len(x.shape)==1)
    r=np.sqrt(x**2 + y**2 + z**2)
    v=np.sqrt(vx**2 + vy**2 + vz**2)
    return np.median(r/v)


def index_width_of_dynamical_time(t,tdyn):
    """Return the index width of the dynamical time.
    """
    assert(len(t.shape)==1)
    return np.argmin(np.abs(t-(t[0]+tdyn)))


def tau_index_down_sampling(index_width,ntauskip=15,extension_factor=2):
    """Down sample the tau index given a desired width about the center of mass.
        We need to sample because otherwise we'd have too much data
    index_width: int
        length of one side away from the center of mass in indices of the simulation time
    extension_factor: int
        how many times to extend the index width
    ntauskip: int
        how many indices to skip
    """
    return np.arange(-int(np.floor(index_width*extension_factor)),int(np.ceil(index_width*extension_factor)),ntauskip)


def time_index_down_sampling(t_index_final,tau_index_initial,ntskip=10):
    """
    Down sample the simulation time. We also need to discard the initial timesteps so that we only begin considering the stream after the COM has travel far enough to use the orbit as a proxy to sample the stream
    """
    assert(t_index_final>tau_index_initial)
    assert(t_index_final>0)
    assert(tau_index_initial<0)
    return np.arange(-tau_index_initial,t_index_final,ntskip)


def initialize_outputs(nGCs,ntime,ntau):
    """
    the output arrays are initialized to zero
    nGCs: int
        number of globular clusters
    ntime: int
        number of simulation time steps
    ntau: int
        number of tau indices (orbit sampling points)

    Returns
    -------
    ax,ay,az: np.ndarray,np.ndarray,np.ndarray
        the acceleration components of the sum of all cluster acceleration
    magnitude: np.ndarray
        the magnitude of the acceleration vector for each cluster
    """
    ax,ay,az                    =   np.zeros((ntime,ntau)),np.zeros((ntime,ntau)),np.zeros((ntime,ntau))
    magnitude                   =   np.zeros((nGCs,ntime,ntau))
    return ax,ay,az,magnitude
