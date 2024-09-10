"""
This module contains functions that are used to predict how the length of a stream will change in time

NUMERICAL 
    - dr_RMS_distributions
        - loads particles from plummer and gets the RMS radial distance
    - get_two_particles_about_galactocentric_radius
        - places two particles a distance dr away from the original position along the radial line
    - perform_integration
        - performs the integration of the particles
    - get_time_from_theta
        - uses the cylindrical angle to find how far ahead or behind in time the leading and trailing particles are from the host


DATA BASED
    -

"""

import numpy as np
# import gcs


# import tstrippy
def dr_RMS_distributions(xp,yp,zp):
    rp                  =   np.sqrt(xp**2+yp**2+zp**2) #type: ignore
    dr                  =   np.sqrt(np.mean(rp**2))
    return dr


# def get_two_particles_about_galactocentric_radius(dr,position):
#     """
#     Given a position and a change in radius, this function returns the position and velocity of two particles that are a distance dr away from the original position along the radial line
#     """
#     xp0,yp0,zp0,vxp0,vyp0,vzp0 = position
    
#     r, theta, phi, rdot, thetadot, phidot = gcs.misc.cartesian_to_spherical(xp0,yp0,zp0,vxp0,vyp0,vzp0)
    
#     x0,y0,z0,vx0,vy0,vz0 = gcs.misc.spherical_to_cartesian(r-dr, theta, phi, rdot, thetadot, phidot) #type: ignore
#     x1,y1,z1,vx1,vy1,vz1 = gcs.misc.spherical_to_cartesian(r+dr, theta, phi, rdot, thetadot, phidot) #type: ignore

#     x,y,z = np.array([x0,x1]),np.array([y0,y1]),np.array([z0,z1])
#     vx,vy,vz = np.array([vxp0,vxp0]),np.array([vyp0,vyp0]),np.array([vzp0,vzp0])

#     return x,y,z,vx,vy,vz


# def perform_integration(staticgalaxy,integrationparameters,initialkinematics,inithostperturber):
#     Nstep = integrationparameters[2]
#     nparticles = len(initialkinematics[0])
#     integrator=tstrippy.integrator
#     integrator.setstaticgalaxy(*staticgalaxy)
#     integrator.setintegrationparameters(*integrationparameters)
#     integrator.setinitialkinematics(*initialkinematics)
#     integrator.inithostperturber(*inithostperturber)
#     xt,yt,zt,vxt,vyt,vzt = integrator.leapfrogintime(Nstep,int(nparticles))
#     integrator.deallocate()
#     del(integrator)
#     return xt,yt,zt,vxt,vyt,vzt


# def get_time_from_theta(tHost,thetaHost,thetaL,thetaT):
#     """
#     NOTE: these theta values are in radians and need to be unwrapped before being passed to this function!
#     gcs.misc.unwrap_theta(theta) will do this for you


#     Find how far ahead or behind in time the leading and trailing particles are from the host

#     The assertions are to check that tHost is longer than thetaHost and that thetaT and thetaL are the same length

#     tHost should cover into the future where thetaT & thetaL should cover the whole simulation time

#     This is important because if tHost only covers the simulation time, then the leading particle will not have a time to compare to
    
#     """
    
#     assert len(thetaT)==len(thetaL)
#     assert len(tHost)==len(thetaHost)
#     assert len(tHost)>len(thetaT)
#     leading_time = []
#     trailing_time = []



#     for i in range(thetaT.shape[0]):
#         leading_index=np.argmin(np.abs(thetaL[i]-thetaHost))
#         trailing_index=np.argmin(np.abs(thetaT[i]-thetaHost))
#         leading_time.append(tHost[leading_index]-tHost[i])
#         trailing_time.append(tHost[trailing_index]-tHost[i])
#     return np.array(leading_time),np.array(trailing_time)




####################################################
############## DATA BASED FUNCTIONS ################
####################################################



def get_envelop_indexes(density_array, density_min):
    """
    Finds the limits of a stream based on a density array and a minimum density threshold.

    The first axis of the density array is the simulation time
    The second axis the array is the 1D profile of the stream on the orbit 

    Parameters:
    - density_array (ndarray): A 2D array representing the density values.
    - density_min (float): The minimum density threshold.

    Returns:
    - index_from_left (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the left for each row of the density array.
    - index_from_right (ndarray): An array containing the indices of the first elements that surpass the density threshold when scanning from the right for each row of the density array.
    """
    
    Nstamps, _ =density_array.shape
    
    index_from_left, index_from_right = np.zeros(Nstamps), np.zeros(Nstamps)
    for i in range(Nstamps):
        array = density_array[i]
        # Find the first element that surpasses THRESHOLD when scanning from the left
        index_from_left[i] = np.argmax(array > density_min)
        # Find the first element that surpasses THRESHOLD when scanning from the right
        index_from_right[i] = len(array) - np.argmax(array[::-1] > density_min) - 1

    index_from_left = index_from_left.astype(int)
    index_from_right = index_from_right.astype(int)
    return index_from_left, index_from_right


def tau_envelopes(tau, index_from_left, index_from_right):
    """
    we want \tau(t), tau as a function of time...
    """
    tau_left = [tau[xx] for xx in index_from_left]
    tau_right = [tau[xx] for xx in index_from_right]
    return np.array(tau_left), np.array(tau_right)



def sig_clip(quantity, Nstdflag, Nstdclip, sides = 100, trial_max = 10000):

    # some signal processing in case there are problems with the end points
    std,mean    = np.std(quantity),np.mean(quantity)
    flag,clip   = mean+Nstdflag*std,mean+Nstdclip*std
    abs_diff = np.abs(quantity-mean)
    bad_dexes = np.where(abs_diff>flag)[0]
    sig_clipped=quantity.copy()
    cc = 0 
    conditions = cc < trial_max and len(bad_dexes)>0
    while conditions:
        # replace the bad dexes with the average of the two neighbors
        for bd in bad_dexes:
            uplim,lowlim=bd+sides,bd-sides
            # make sure the limits are within the array
            if uplim>len(sig_clipped):
                uplim=len(sig_clipped)
            if lowlim<0:
                lowlim=0
            sig_clipped[bd] = np.mean(sig_clipped[bd-sides:bd+sides-1])
            if sig_clipped[bd]<clip:
                bad_dexes = np.delete(bad_dexes,np.where(bad_dexes==bd))
        cc+=1
        
        if cc == trial_max:
            print('Max number of iterations reached')
        conditions = cc < trial_max and len(bad_dexes)>0
    return sig_clipped