"""
Using analytic arguments, we can compute the length of the stream
Altough this is subsequently done numerically. 
We compute the hill sphere using an approximation that the GC and galaxy are point masses

Then, we palce two particles along orbital vector \vec{r} at \vec{r} \pm \delta{r}
We integrate these two orbits
At each timestep, I compute the cylincridal angle \theta between them
This allows us to know the stream growth versus time
This is reported in time, i.e. the time since the globular cluster was at the former position
and the time until the globular cluster will be at the latter position
"""


import numpy as np 
import astropy.units as u # type: ignore
from galpy import potential # type: ignore



####################################################
######## Computing the stream length ###############
####################################################

def length_time_from_theta(time,trailing,host,leading,unitT=u.kpc/(u.km/u.s)):
    """
    trailing, host, leading are measurements about the z-axis in cylindrical coordinates
        it's always increasing and not cyclic
    """
    leading_time = []
    trailing_time = []
    for i in range(time.shape[0]):
        leading_index=np.argmin(np.abs(leading[i]-host))
        trailing_index=np.argmin(np.abs(trailing[i]-host))
        leading_time.append(time[leading_index]-time[i])
        trailing_time.append(time[trailing_index]-time[i])
    leading_time=leading_time*unitT
    leading_time=leading_time.to(u.Myr)
    trailing_time=trailing_time*unitT
    trailing_time=trailing_time.to(u.Myr)
    return leading_time,trailing_time


####################################################
########### GENERAL COORDINATE STUFF ###############
####################################################
def unwrap_angles(theta):
    unwrapped_theta = theta.copy()
    correction = 0
    for i in range(1, len(theta)):
        delta = theta[i] - theta[i-1]
        if delta < -np.pi:
            correction += 2*np.pi
        elif delta > np.pi:
            correction -= 2*np.pi
        unwrapped_theta[i] = theta[i] + correction
    return unwrapped_theta

def cart_to_cylin(x,y,z,vx,vy,vz):
    R = np.sqrt(x**2 + y**2)
    theta_n = np.arctan2(y,x)
    theta = unwrap_angles(theta_n)
    theta_dot = (x*vy - y*vx)/R
    vR = (x*vx + y*vy)/R
    return R,theta,z,vR,theta_dot,vz

def two_radial_points(x,y,z,dr):
    # get initial positions of the far and near point
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/r)
    rfar = r+dr
    rnear = r-dr
    xfar,yfar,zfar=rfar*np.sin(theta)*np.cos(phi),rfar*np.sin(theta)*np.sin(phi),rfar*np.cos(theta)
    xnear,ynear,znear=rnear*np.sin(theta)*np.cos(phi),rnear*np.sin(theta)*np.sin(phi),rnear*np.cos(theta)
    pos_far = np.array([xfar,yfar,zfar])
    pos_near = np.array([xnear,ynear,znear])
    return pos_far,pos_near
####################################################
########### MIYAMOTO NAGAI DISK + HALO #############
####################################################
def hill_radius(r,m,M):
    return r*(m/(3*(M+m)))**(1/3)

def enclosed_mass_PII(R,Z,params):
    r0=1*u.kpc
    v0=1*u.km/u.s
    r=np.sqrt(R**2 + Z**2)
    
    halo = [params[0],params[1],params[2],params[3],params[4]]
    thindisk = [params[0],params[5],params[6],params[7]]
    thickdisk = [params[0],params[8],params[9],params[10]]

    disc1=potential.MiyamotoNagaiPotential(\
    amp=thindisk[0]*thindisk[1],\
    a=thindisk[2],b=thindisk[3],ro=r0,vo=v0)
    
    disc2=potential.MiyamotoNagaiPotential(\
        amp=thickdisk[0]*thickdisk[1],\
        a=thickdisk[2],b=thickdisk[3],ro=r0,vo=v0)
    
    halo_mass=allan_mass_enc(r,halo[1],halo[2],halo[3]) 
    disc1_mass=disc1.mass(R,Z)
    disc2_mass=disc2.mass(R,Z)
    return halo_mass+disc1_mass+disc2_mass
    
def allan_mass_enc(r,M,a,gamma):
    # getting the halo mass
    A=(r/a)**gamma
    B=1+(r/a)**(gamma-1)
    return M*A/B

####################################################
################## PLUMMER SPHERE ##################
####################################################
def plummer_beyond_radial_density_moment(r_lower,M,a):
    """
    The expectation value of the radius given a lower limit 
    """
    moment = (0-plummer_density_r_moment_integrand(r_lower,M,a))
    shell_mass = M-plummer_M_enc(r_lower,M,a)
    return moment/shell_mass
    
def plummer_density_r_moment_integrand(r,M,a):
    u = r/a
    return 3*M*a*( (1/3)*u**(-3/2) - u**(1/2) )

def half_mass_to_plummer(r_hm):
    return r_hm/1.3

def plummer_M_enc(r,M,a):
    return M*(r**3 / (r**2 + a**2)**(3/2))
