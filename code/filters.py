import numpy as np





######################## ######################## #####
######################## ORBIT ########################
#################### ######################## #########
def orbit_by_fixed_time(t,orbit,current_index,filter_time):
    xt,yt,zt,vxt,vyt,vzt=orbit
    down_time = t[current_index] - filter_time
    down_dex = np.argmin(np.abs(t-down_time))
    up_time = t[current_index] + filter_time
    up_dex = np.argmin(np.abs(t-up_time))
    tOrb=t[down_dex:up_dex]
    xOrb=xt[down_dex:up_dex]
    yOrb=yt[down_dex:up_dex]
    zOrb=zt[down_dex:up_dex]
    vxOrb=vxt[down_dex:up_dex]
    vyOrb=vyt[down_dex:up_dex]
    vzOrb=vzt[down_dex:up_dex]

    return tOrb,xOrb,yOrb,zOrb,vxOrb,vyOrb,vzOrb

######################## ######################## ######
######################## STREAM ########################
#################### ######################## ##########
def stream_impacted_side(
    x_stream_tail_coordinates: np.ndarray, \
    x_perturber_tail_coordinates: float) -> np.ndarray:
    """
    Filter the impacted stream side based on the coordinates of the stream tail and perturber tail.

    Parameters:
        x_stream_tail_coordinates (np.ndarray): The coordinates of the stream tail.
        x_perturber_tail_coordinates (float): The coordinates of the perturber tail.

    Returns:
        np.ndarray: An array indicating the impacted side of the stream.

    """
    if x_perturber_tail_coordinates > 0:
        impacted_side = x_stream_tail_coordinates > 0
    else:
        impacted_side = x_stream_tail_coordinates < 0
    return impacted_side


def stream_impose_limits_in_tail_coordinates(
    stream_tail_coordinates: np.ndarray,
    xlim: float,
    ylim: float,
    zlim: float) -> np.ndarray:
    """
    Filters the stream tail coordinates based on the given limits.

    Parameters:
        stream_tail_coordinates (np.ndarray): The stream tail coordinates to filter.
        xlim (float): The limit for the x-coordinate.
        ylim (float): The limit for the y-coordinate.
        zlim (float): The limit for the z-coordinate.

    Returns:
        np.ndarray: A boolean array indicating which coordinates pass the filter.

    """
    cond0 = np.abs(stream_tail_coordinates[0, :]) < xlim
    cond1 = np.abs(stream_tail_coordinates[1, :]) < ylim
    cond2 = np.abs(stream_tail_coordinates[2, :]) < zlim
    cond = cond0 & cond1 & cond2
    return cond