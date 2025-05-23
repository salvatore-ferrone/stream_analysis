'''
A module to calculate the coordinates of the particles in the stream relative to the orbit of the GC

Typical usage:
    import tailCoordinates as TC
    # get the coordinates of the particles in the orbit frame

    xprimeP,yprimeP,zprimeP,vxprimeP,vyprimeP,vzprimeP,indexes = TC.transform_from_galactico_centric_to_tail_coordinates(
        xp,yp,zp,vxp,vyp,vzp,tORB,xORB,yORB,zORB,vxORB,vyORB,vzORB,t0=0)

'''
import numpy as np 


def transform_from_galactico_centric_to_tail_coordinates(xp,yp,zp,vxp,vyp,vzp,tORB,xORB,yORB,zORB,vxORB,vyORB,vzORB,t0=0):
    '''
    transform the galcentric coordinates of the particles to the orbit coordinates
    '''
    indexes=get_closests_orbital_index(xORB,yORB,zORB,xp,yp,zp)
    xprimeOrbi=get_orbital_path_length(tORB,xORB,yORB,zORB,t0=t0)
    xprimeP=np.zeros(xp.shape)
    yprimeP=np.zeros(xp.shape)
    zprimeP=np.zeros(xp.shape)
    vxprimeP=np.zeros(xp.shape)
    vyprimeP=np.zeros(xp.shape)
    vzprimeP=np.zeros(xp.shape)
    for i in range(xp.shape[0]):
        POSpart=np.array([xp[i],yp[i],zp[i]])
        POS_near_point=np.array([xORB[indexes[i]],yORB[indexes[i]],zORB[indexes[i]]])
        vel_near= np.array([vxORB[indexes[i]],vyORB[indexes[i]],vzORB[indexes[i]]])
        VelPart = np.array([vxp[i], vyp[i], vzp[i]])
        dV=VelPart-vel_near
        dX = POSpart-POS_near_point
        vunit,runit,zunit,compensation=get_local_unit_vectors_and_particle_offset_compensation(
            POSpart,POS_near_point,vel_near)
        # outputs
        xprimeP[i]=xprimeOrbi[indexes[i]]+np.dot(compensation,vunit)
        yprimeP[i]=np.dot(dX,runit)
        zprimeP[i]=np.dot(dX,zunit)
        vxprimeP[i] = np.dot(dV,vunit)
        vyprimeP[i] = np.dot(dV,runit)
        vzprimeP[i] = np.dot(dV,zunit)
    return xprimeP,yprimeP,zprimeP,vxprimeP,vyprimeP,vzprimeP,indexes

def get_orbital_path_length(tORB:np.ndarray,xORB:np.ndarray,yORB:np.ndarray,zORB:np.ndarray,t0:float=0):
    """
    Calculate the xprime orbit coordinates.
    
    xprime is the pathlength along the orbit ahead of behind the center of mass.
    ORB refers to the orbit of the globular cluster.

    Parameters:
    - tORB (np.ndarray): Array of time values.
    - xORB (np.ndarray): Array of x-coordinate values.
    - yORB (np.ndarray): Array of y-coordinate values.
    - zORB (np.ndarray): Array of z-coordinate values.
    - t0 (float): Reference time value (default: 0).

    Returns:
    - orbital_path_length (np.ndarray): Array of xprime orbit coordinates.
    """
    
    orbital_path_length=np.zeros(xORB.shape)
    center_of_mass_index=np.argmin(np.abs(tORB-t0))
    dx = np.diff(xORB-xORB[center_of_mass_index])
    dy = np.diff(yORB-yORB[center_of_mass_index])
    dz = np.diff(zORB-zORB[center_of_mass_index])
    dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    ahead = dr[center_of_mass_index::]
    behind = dr[:center_of_mass_index]
    orbital_path_length[center_of_mass_index+1::] = np.cumsum(ahead)
    orbital_path_length[:center_of_mass_index] = -np.flip(np.cumsum(np.flip(behind)))
    return orbital_path_length

def get_closests_orbital_index(xORB:np.ndarray,yORB:np.ndarray,zORB:np.ndarray,xp:np.ndarray,yp:np.ndarray,zp:np.ndarray) -> np.ndarray:
    """
    Returns an array of indexes representing the closest orbital index for each point in the given arrays.

    Parameters:
    - xORB (np.ndarray): Array of x-coordinates of orbital points.
    - yORB (np.ndarray): Array of y-coordinates of orbital points.
    - zORB (np.ndarray): Array of z-coordinates of orbital points.
    - xp (np.ndarray): Array of x-coordinates of points to find closest orbital index for.
    - yp (np.ndarray): Array of y-coordinates of points to find closest orbital index for.
    - zp (np.ndarray): Array of z-coordinates of points to find closest orbital index for.

    Returns:
    - indexes (np.ndarray): Array of nearest corresponding orbital index for each particle.


    For instance, the nearst time stamp along the orbit to a given particle is:
    xp[i], yp[i], zp[i] is the particle position
    tORB[indexes[i]], xORB[indexes[i]], yORB[indexes[i]], zORB[indexes[i]]
    the length of the indexes is the same as the particles

    """
    indexes=np.zeros(xp.shape,dtype=int)
    for i in range(xp.shape[0]):
        dx,dy,dz = xp[i]-xORB,yp[i]-yORB,zp[i]-zORB
        dist=dx**2 + dy**2 + dz**2
        indexes[i]=np.argmin(dist) 
    return indexes

def get_local_unit_vectors_and_particle_offset_compensation(particle_position: np.ndarray,position_orbital_near_point: np.ndarray,velocity_orbital_near_point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    def get_local_unit_vectors_and_particle_offset_compensation(particle_position, position_orbital_near_point, velocity_orbital_near_point):
        """
        The coordinate system of tail coordinates is defined locally for each particle. 
        This function finds those three unit vectors. They are defined as 
        runit: unit radial vector from the galactic center to the nearest point along the orbit to the particle
        vunit: unit velocity vector in the direction of the orbital motion of the host globular cluster at the nearest point along the orbit to the particle
        zunit: unit vector perpendicular to the plane defined by runit and vunit. The angular momentum in essence.

        Parameters:
        - particle_position (np.ndarray): Vector with 3 components representing the particle position.
        - position_orbital_near_point (np.ndarray): Vector with 3 components representing the position of the orbital near point.
        - velocity_orbital_near_point (np.ndarray): Vector with 3 components representing the velocity of the orbital near point.

        Returns:
        - vunit (np.ndarray): Vector representing the unit velocity vector.
        - runit (np.ndarray): Vector representing the unit radial vector.
        - zunit (np.ndarray): Vector representing the unit z vector.
        - compensation (np.ndarray): Vector representing the particle offset compensation.
        """
        # Implementation goes here
    '''
    assert particle_position.shape[0]==3,"particle_position must be vector with 3 components"
    assert position_orbital_near_point.shape[0]==3,"position_orbital_near_point must be vector with 3 components"
    assert velocity_orbital_near_point.shape[0]==3,"velocity_orbital_near_point must be vector with 3 components"
    dX = particle_position-position_orbital_near_point
    vunit =velocity_orbital_near_point/np.linalg.norm(velocity_orbital_near_point)
    # adjust the point slightly, due to the discrete grid
    compensation=np.dot(dX,vunit)*vunit
    interpolationPoint=position_orbital_near_point+compensation 
    runit = interpolationPoint/np.linalg.norm(interpolationPoint)
    zunit=np.cross(runit,vunit)
    return vunit,runit,zunit,compensation
 
def filter_orbit_by_dynamical_time(tORB: np.ndarray, xtORB: np.ndarray, ytORB: np.ndarray, ztORB: np.ndarray,
                      vxtORB: np.ndarray, vytORB: np.ndarray, vztORB: np.ndarray,
                      time_of_interest: float, nDynTimes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the orbit coordinates based on the dynamical time about a time of interest.
    This is useful since I am doing a naive computation of finding the nearest point of each particle to the orbit.

    Args:
        tORB (np.ndarray): Array of time values.
        xtORB (np.ndarray): Array of x-coordinate values.
        ytORB (np.ndarray): Array of y-coordinate values.
        ztORB (np.ndarray): Array of z-coordinate values.
        vxtORB (np.ndarray): Array of x-velocity values.
        vytORB (np.ndarray): Array of y-velocity values.
        vztORB (np.ndarray): Array of z-velocity values.
        currenttime (float): Current time value.
        nDynTimes (int): Number of dynamical times.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
        the filtered orbit coordinates: TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB.
    """
    
    assert ( nDynTimes > 0), "nDynTimes must be greater than 0"
    assert ( nDynTimes < 5), "nDynTimes must be less than 5"
    assert( time_of_interest > np.min(tORB)), "time_of_interest must be greater than the minimum time in tORB"
    assert( time_of_interest < np.max(tORB)), "time_of_interest must be less than the maximum time in tORB"
    
    rORB = np.sqrt(xtORB ** 2 + ytORB ** 2 + ztORB ** 2)
    vORB = np.sqrt(vxtORB ** 2 + vytORB ** 2 + vztORB ** 2)

    todayindex = np.argmin(np.abs(tORB - time_of_interest))
    ttoday = tORB[todayindex]

    Tdyn = np.median(rORB / vORB)

    cond1 = (tORB - ttoday) < (nDynTimes * Tdyn)
    cond2 = (tORB - ttoday) > (-nDynTimes * Tdyn)
    cond = cond1 * cond2

    XORB = xtORB[cond]
    YORB = ytORB[cond]
    ZORB = ztORB[cond]
    VXORB = vxtORB[cond]
    VYORB = vytORB[cond]
    VZORB = vztORB[cond]
    TORB = tORB[cond]

    return TORB, XORB, YORB, ZORB, VXORB, VYORB, VZORB

