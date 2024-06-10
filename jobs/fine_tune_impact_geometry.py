"""
This is intended to be a job that will fine tune all of the impact geometries!
"""
from scipy import optimize
import numpy as np 
import datetime
import h5py
import os 
import sys 
codepath="/obs/sferrone/stream_analysis/code/"
sys.path.append(codepath)
import path_handler as PH               # type: ignore
import fine_tune_erkal_geometry as FTEG # type: ignore
import parametric_stream_fitting as PSF # type: ignore
import data_extractors as DE            # type: ignore
import compute_stream_1D_density as CSD    # type: ignore


def main(mcarlo_int, perturberName,
        GCname="Pal5",\
        potential_stream="pouliasis2017pii-Pal5-suspects",\
        potential_GCs="pouliasis2017pii-GCNBody",\
        NP=int(1e5)):
    """
    DOES THE ANALYSIS FOR A SINGLE IMPACT AND WRITES THE RESULTS TO A FILE

    Parameters:
    - mcarlo (int/str): The number of Monte Carlo simulations to perform.
    - perturberName (str): The name of the perturber.
    - output_path (str, optional): The output path to write the results. Default is "../impact-geometry-results/".

    Returns:
    None
    """
    # set parameters 
    montecarlokey="monte-carlo-"+str(mcarlo_int).zfill(3)

    print("Starting:", montecarlokey, perturberName)
    geometry_erkal, parameters_stream_and_perturber =\
        full_impact_geometry_analysis_one_impact(\
            GCname=GCname,\
            montecarlokey=montecarlokey,\
            perturberName=perturberName,\
            potential_GCs=potential_GCs,\
            potential_stream=potential_stream,\
            NP=NP)

    # prepare the outpuit 
    attributes = make_outfile_attributes(\
                            montecarlokey       =   montecarlokey, 
                            perturberName       =   perturberName,
                            gc_orbit_gfield     =   potential_GCs,
                            stream_orbit_gfield =   potential_stream,
                            NP                  =   NP,
                            GCname              =   GCname)

    tempfilepath=PH.temporary_impact_geometry(GCname=GCname,montecarlokey=montecarlokey,potential=potential_stream,perturber=perturberName)
    
    write_results_to_temp_file(tempfilepath,attributes,geometry_erkal,parameters_stream_and_perturber)
    
    # fname = PH.impact_geometry_results(\
    #     GCname=GCname,montecarlokey=montecarlokey,potential=potential_stream)
    # write_results_to_file(fname, attributes, perturberName, geometry_erkal, parameters_stream_and_perturber)


def full_impact_geometry_analysis_one_impact(GCname,montecarlokey,perturberName,potential_GCs,potential_stream,NP):
    """
    Perform a full impact geometry analysis for a single impact.

    Parameters:
    ----------
    mcarlo : str
        The name of the Monte Carlo simulation.
    perturberName : str
        The name of the perturber.

    Returns:
    -------
    See Erkal et al 2015 doi: 10.1093/mnras/stv655
    geometry_erkal : dict
        A dictionary containing the impact geometry parameters:
        - impact_parameter : float
            The impact parameter.
        - w_par : float
            The parallel velocity component.
        - w_per : float
            The perpendicular velocity component.
        - alpha : float
            The angle perturber's trajectory and the x-axis in impact coordinate frame.

    parameters_stream_and_perturber : dict
        A dictionary containing the parameters of the stream and perturber:
        - s : float
            The stream parameter.
        - t : float
            The time parameter.
        - stream_impact_position_galactocentric : array-like
            The galactocentric position of the stream impact.
        - stream_impact_velocity_galactocentric : array-like
            The galactocentric velocity of the stream impact.
        - perturber_position_galactocentric : array-like
            The galactocentric position of the perturber.
        - b_vec_galactic : array-like
            The galactic basis vector.
        - b_vec_tail_basis : array-like
            The tail basis vector.
        - coefficient_time_fit_params : array-like
            The coefficients of the time fit parameters.
        - trajectory_coeffs : array-like
            The trajectory coefficients.
        - s_range : array-like
            The range of the stream coordinate.
        - t_time_stamps : array-like
            The time stamps of the simulation sampling.

    """
    pathStreamOrbit, pathPerturOrbit, pathPal5Orbit, pathForceFile, pathTauFile =   retrieve_paths(\
                    montecarlokey       =   montecarlokey, \
                    perturberName       =   perturberName,\
                    gc_orbit_gfield     =   potential_GCs,
                    stream_orbit_gfield =   potential_stream,\
                    NP                  =   NP,\
                    GCname              =   GCname)
    
    streamOrbit, perturberOrbit, pal5Orbit, forceFile, tauFile=   open_input_data_files(\
        pathStreamOrbit,pathPerturOrbit,pathPal5Orbit,pathForceFile,pathTauFile)
    
    
    # 1. Get approximate impact time from the force file
    approx_impact_time,approx_impact_tau=get_approx_impact_time_with_stream_density_filter(tauFile,forceFile,perturberName)
    # 2. fit the stream to a 3D parabola for a few time stamps around the approx impact time
    simulation_time_stamps, coefficient_time_fit_params,stream_time_coordinate_range=\
        obtain_parametric_stream_coefficients(montecarlokey,
                                          streamOrbit,
                                          pal5Orbit,
                                          approx_impact_time,
                                          approx_impact_tau)
    
    time_range = [simulation_time_stamps.min(),simulation_time_stamps.max()]

    trajectory_coeffs = parameterize_oribtal_trajectory(perturberOrbit, montecarlokey, time_range)
    
    # 3. minimize the distance between the stream and the perturber
    results=minimize_distance_between_stream_and_perturber(\
        stream_time_coordinate_range,\
        simulation_time_stamps,\
        coefficient_time_fit_params,\
        trajectory_coeffs,)
    s,t=results.x
    
    # 4. get the full impact geometry
    impact_parameter,w_par,w_per,alpha,rs,vs,rp,b_vec_galactic,b_vec_tail_basis=\
        FTEG.get_full_impact_geometry_from_parametrization(\
            s,t,coefficient_time_fit_params,trajectory_coeffs)
    mass=perturberOrbit[montecarlokey]["initialConditions"]['Mass'][0]
    rh_m=perturberOrbit[montecarlokey]["initialConditions"]['rh_m'][0]
    plummer_radius=half_mass_to_plummer_radius(rh_m)
    
    # 5. Store the results
    geometry_erkal={
        "impact_parameter":impact_parameter,
        "w_par":w_par,
        "w_per":w_per,
        "alpha":alpha,
        "mass":mass,
        "plummer_radius":plummer_radius,
    }
    parameters_stream_and_perturber={
        "s":s,
        "t":t,
        "stream_impact_position_galactocentric":rs,
        "stream_impact_velocity_galactocentric":vs,
        "perturber_position_galactocentric":rp,
        "b_vec_galactic":b_vec_galactic,
        "b_vec_tail_basis":b_vec_tail_basis,
        "coefficient_time_fit_params":coefficient_time_fit_params,
        "trajectory_coeffs":trajectory_coeffs,
        "s_range":stream_time_coordinate_range,
        "t_time_stamps":simulation_time_stamps,
    }
    
    # 6. close th efiles
    streamOrbit.close()
    perturberOrbit.close()
    pal5Orbit.close()
    forceFile.close()
    tauFile.close()
    return geometry_erkal,parameters_stream_and_perturber


#########################################
################## I/O ##################
#########################################
def combine_temp_files(perturberName:str,GCname:str,potential:str,):
    bigfilename=PH.impact_geometry_results(GCname=GCname,perturber=perturberName,potential=potential,)
    bigfile = h5py.File(bigfilename, 'w')
    montecarlokeys = ["monte-carlo-"+str(i).zfill(3) for i in range(50)]
    
    for montecarlokey in montecarlokeys:
        tempfilepath=PH.temporary_impact_geometry(
            GCname=GCname,
            montecarlokey=montecarlokey,
            potential=potential,
            perturber=perturberName)
        
        if not os.path.exists(tempfilepath):
            print("File DOES NOT EXIST:",tempfilepath)
            continue
        tempfile=h5py.File(tempfilepath, 'r')
        
        # Check if the group already exists
        if montecarlokey in bigfile:
            group = bigfile[montecarlokey]
            print("WARNING: Group already exists. Overwriting.")
        else:
            group = bigfile.create_group(montecarlokey)

        # Copy the impact geometry
        for item in tempfile:
            tempfile.copy(item, group, expand_soft=True, expand_external=True, expand_refs=True)
            
        # Copy the attributes
        for attr_name, attr_value in tempfile.attrs.items():
            group.attrs[attr_name] = attr_value    
        
        tempfile.close()

        
    bigfile.close()
    
    
def write_results_to_temp_file(tempfilepath,attributes,erkal_2015_params,parametric_equation_params):
    """
    Write the results to a file in HDF5 format.

    Parameters:
    - output_path (str): The path where the output file will be saved.
    - mcarlo (str): The name of the Monte Carlo simulation.
    - perturberName (str): The name of the perturber.
    - geometry_erkal (dict): A dictionary containing the geometry parameters for Erkal 2015.
    - parameters_stream_and_perturber (dict): A dictionary containing the parameters for the parametric equation.

    Returns:
    None
    """
    with h5py.File(tempfilepath,'w') as f:
        f.create_group('erkal_2015_params')
        f.create_group('parametric_equation_params')
        for key in erkal_2015_params.keys():
            f["erkal_2015_params/"].create_dataset(key,data=erkal_2015_params[key])
        for key in parametric_equation_params.keys():
            f["parametric_equation_params/"].create_dataset(key,data=parametric_equation_params[key])
        for key in attributes.keys():
            f.attrs[key]=attributes[key]


def make_outfile_attributes(montecarlokey:str, 
                            perturberName:str,
                            gc_orbit_gfield:str="pouliasis2017pii-GCNBody",
                            stream_orbit_gfield:str="pouliasis2017pii-Pal5-suspects",
                            NP:int=100000,
                            GCname:str="Pal5"):
    """
    Create a dictionary of attributes for the output file.

    Parameters:
    - mcarlo (str): The name of the mcarlo parameter.
    - perturberName (str): The name of the perturber.

    Returns:
    - attributes (dict): A dictionary containing the following attributes:
        - author (str): The author of the file.
        - creation-date (str): The date and time of file creation.
        - contact (str): The contact email address.
        - description (str): A description of the file.
        - pathStreamOrbit (str): The path to the stream orbit file.
        - pathPerturOrbit (str): The path to the perturber orbit file.
        - pathPal5Orbit (str): The path to the Pal5 orbit file.
        - pathForceFile (str): The path to the force file.
    """
    pathStreamOrbit, pathPerturOrbit, pathPal5Orbit, pathForceFile, pathTauFile = \
            retrieve_paths(montecarlokey=montecarlokey, 
                            perturberName=perturberName,
                            gc_orbit_gfield=gc_orbit_gfield,
                            stream_orbit_gfield=stream_orbit_gfield,
                            NP=NP,
                            GCname=GCname)
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attributes = {
        "author": "Salvatore Ferrone",
        "creation-date": datestring,
        "contact": "salvatore.ferrone@uniroma1.it",
        "description": "impact geometry of simulations following Erkal et al 2015",
        "pathStreamOrbit": pathStreamOrbit,
        "pathPerturOrbit": pathPerturOrbit,
        "pathPal5Orbit": pathPal5Orbit,
        "pathForceFile": pathForceFile,
        "pathTauFile": pathTauFile,
    }
    return attributes


def retrieve_paths(montecarlokey:str, perturberName:str,\
                    gc_orbit_gfield:str,
                    stream_orbit_gfield:str,\
                    NP:int=100000,\
                    GCname:str="Pal5"
                   ):
    """
    Retrieve the paths to various files used in the simulation.

    Parameters:
    -----------
    mcarlo : str
        The name of the mcarlo file.
    perturberName : str
        The name of the perturber orbit file.

    Returns:
    --------
    tuple
        A tuple containing the paths to the stream orbit, perturber orbit, Pal5 orbit, and force file.
        
    Examples:
    ---------
    pathStreamOrbit,pathPerturOrbit,pathPal5Orbit,pathForceFile=retrieve_paths(mcarlo,perturberName)
    """
    

    # get streamOrbit
    pathStreamOrbit = PH.stream_orbit(GCname=GCname,montecarlokey=montecarlokey,potential=stream_orbit_gfield,NP=NP)

    # get path to perturber orbit
    pathPerturOrbit = PH.orbit(GCname=perturberName,potential=gc_orbit_gfield)

    # path to Pal5 orbit 
    pathPal5Orbit = PH.orbit(GCname=GCname,potential=gc_orbit_gfield)

    # path to force file
    pathForceFile = PH.force_on_orbit(GCname=GCname,montecarlokey=montecarlokey,potential=gc_orbit_gfield)

    # path to stream projections onto orbit
    pathTauFile = PH.tau_coordinates(GCname,montecarlokey=montecarlokey,potential_stream_orbit=stream_orbit_gfield)

    return pathStreamOrbit, pathPerturOrbit, pathPal5Orbit, pathForceFile, pathTauFile


def open_input_data_files(pathStreamOrbit, pathPerturOrbit, pathPal5Orbit, pathForceFile, pathTauFile):
    """
    Open input data files.

    Parameters
    ----------
    pathStreamOrbit : str
        Path to the stream orbit file.
    pathPerturOrbit : str
        Path to the perturber orbit file.
    pathPal5Orbit : str
        Path to the Pal5 orbit file.
    pathForceFile : str
        Path to the force file.

    Returns
    -------
    streamOrbit : h5py.File
        The opened stream orbit file.
    perturberOrbit : h5py.File
        The opened perturber orbit file.
    pal5Orbit : h5py.File
        The opened Pal5 orbit file.
    forceFile : h5py.File
        The opened force file.
    
    Examples
    --------
    streamOrbit, perturberOrbit, pal5Orbit, forceFile, tauFile =open_input_data_files(pathStreamOrbit,pathPerturOrbit,pathPal5Orbit,pathForceFile)
    """
    streamOrbit = h5py.File(pathStreamOrbit, 'r')
    perturberOrbit = h5py.File(pathPerturOrbit, 'r')
    pal5Orbit = h5py.File(pathPal5Orbit, 'r')
    forceFile = h5py.File(pathForceFile, 'r')
    tauFile = h5py.File(pathTauFile, 'r')
    
    return streamOrbit, perturberOrbit, pal5Orbit, forceFile, tauFile


######################################################
######## COMPUTATIONS WITH OPENING DATA FILES ########
######################################################
def obtain_parametric_stream_coefficients(montecarlokey,
                                          streamOrbit,
                                          pal5Orbit,
                                          approx_impact_time,
                                          approx_impact_tau):
    """
    Obtain the parametric stream coefficients.

    Parameters:
    -----------
    montecarlokey : str
        index of which montecarlo iteration to use
    perturberName : str
        Name of the perturber.
    streamOrbit : Orbit
        Orbit of the stream.
    perturberOrbit : Orbit
        Orbit of the perturber.
    pal5Orbit : Orbit
        Orbit of the Pal 5 stream.
    approx_impact_time : float
        the approximate impact time


    Returns:
    --------
    coefficient_time_fit_params : array-like
        Coefficient time fit parameters.
    stream_coordinate_range : array-like
        Stream coordinate range.
    simulation_sampling_time_stamps : array-like
        Simulation sampling time stamps.
    """
    
    ##### STEP 1, parameterize the stream's tau coordinate

    simulation_time_stamps,stream_time_coordinate,stream_galactic_coordinates=parameterize_stream_snapshots(\
            montecarlokey=montecarlokey,
            streamOrbit=streamOrbit,
            pal5Orbit=pal5Orbit,
            approx_impact_time=approx_impact_time,
            approx_impact_tau=approx_impact_tau,)
    
    stream_time_coordinate_range = [np.min(stream_time_coordinate),np.max(stream_time_coordinate)]
    
    ##### STEP 2, fit the stream to a 3D parabola for a few time stamps around the approx impact time
    temporal_coefficients_array=get_3D_parabola_stream_snapshot_coefficients(stream_time_coordinate,stream_galactic_coordinates)
    
    ##### Step 3, make the coefficients of the parbola move in time
    coefficient_time_fit_params = PSF.linearize_temporal_stream_coefficients_array(
        temporal_coefficients_array, simulation_time_stamps)
    
    return simulation_time_stamps, coefficient_time_fit_params,stream_time_coordinate_range


def parameterize_stream_snapshots(\
    montecarlokey,
    streamOrbit,
    pal5Orbit,\
    approx_impact_time,\
    approx_impact_tau,
    n_adjacent=2):
    ############################################
    ########### SET BASE PARAMETERS ###########
    n_dynamic_time      =   2
    xmin,xlim,ylim,zlim =   0.5,20,0.5,0.5


    #########################################
    ################ INPUTS #################
    ##########################################

    stream_orbit_time_stamps = DE.extract_time_steps_from_stream_orbit(streamOrbit).value
    stream_orbit_index = np.argmin(np.abs(stream_orbit_time_stamps-approx_impact_time))

    stream_indexes_of_interest = np.arange(stream_orbit_index-n_adjacent,stream_orbit_index+n_adjacent+1)

    ##########################################
    ###### GET THE TIME INDEX TO PARAMETERIZE THE STREAM 
    ##########################################
    # get the whole stream because we want to filter the stream... 
    stream_galactic_coordinates = DE.get_galactic_coordinates_of_stream(\
            stream_orbit_index,streamOrbit)
    

    ##### GRAB PIECES OF THE ORBIT 
    pal5_tuple = (  \
                        pal5Orbit[montecarlokey]['xt'][:], pal5Orbit[montecarlokey]['yt'][:], pal5Orbit[montecarlokey]['zt'][:], \
                        pal5Orbit[montecarlokey]['vxt'][:], pal5Orbit[montecarlokey]['vyt'][:], pal5Orbit[montecarlokey]['vzt'][:])

    host_orbit_galactic_coordinates = DE.filter_orbit_by_dynamical_time(\
        pal5Orbit[montecarlokey]['t'][:],\
        pal5_tuple,\
        approx_impact_time,\
        n_dynamic_time=n_dynamic_time)
    

    # convert the stream and perturber to tail coordinates # NEEDS TO BE AN ARRAY 
    _,stream_tail_coordinates=DE.convert_instant_to_tail_coordinates(\
                stream_galactic_coordinates,host_orbit_galactic_coordinates,approx_impact_time)

    # the time range of the stream coordinate    
    myfilter = filter_stream_about_suspected_impact_time(\
        stream_tail_coordinates,approx_impact_tau,xmin,xlim,ylim,zlim)
    

    
    mydata = streamOrbit['timestamps'][str(stream_orbit_index)][:].copy().T
    mydata = mydata[myfilter,:].T
    stream_stamps   = []
    tau_coordinate  = np.zeros((len(stream_indexes_of_interest),mydata.shape[1]))

    cc = 0 
    for ii in stream_indexes_of_interest:
        current_time = stream_orbit_time_stamps[ii]
        mydata = streamOrbit['timestamps'][str(ii)][:].copy().T
        mydata  =   mydata[myfilter,:].T
        stream_stamps.append(mydata)
        temp,_=DE.convert_instant_to_tail_coordinates(\
            mydata,host_orbit_galactic_coordinates,current_time)
        tau_coordinate[cc,:]=temp
        cc+=1
    
    simulation_time_stamps = stream_orbit_time_stamps[stream_indexes_of_interest]
    
    return simulation_time_stamps, tau_coordinate, stream_stamps


def parameterize_oribtal_trajectory(\
    orbit_file: h5py.File, \
    mcarlo: str, \
    simulation_time_range:np.ndarray) -> np.ndarray:
    """
    Parameterizes the orbital trajectory using polynomial fitting.

    Parameters:
    orbit_file (h5py.File): The HDF5 file containing the orbital data.
    mcarlo (str): The name of the Monte Carlo simulation.
    simulation_time_range (np.ndarray): The time range for the simulation.

    Returns:
    trajectory_coeffs (np.ndarray): The coefficients of the polynomial fit for each dimension of the trajectory.
    
    Example: 
    >>> trajectory_coeffs=parameterize_oribtal_trajectory(orbit_file, mcarlo="monte-carlo-009", simulation_time_range=np.array([-0.14, -0.10])
    >>> # extract the coefficients
    >>> alpha0, alpha1, alpha2 = trajectory_coeffs[0,:]
    >>> beta0, beta1, beta2 = trajectory_coeffs[1,:]
    >>> gamma0, gamma1, gamma2 = trajectory_coeffs[2,:]
    >>> # get the x,y,z coordinates at time t
    >>> x_t = alpha0 + alpha1*t + alpha2*t^2
    >>> y_t = beta0 + beta1*t + beta2*t^2
    >>> z_t = gamma0 + gamma1*t + gamma2*t^2
    """

    polynomial_degree = 2
    
    simulation_initial_time_step    =   np.argmin(np.abs(orbit_file[mcarlo]['t'][:] - simulation_time_range[0]))
    simulation_end_time_step        =   np.argmin(np.abs(orbit_file[mcarlo]['t'][:] - simulation_time_range[-1]))


    t_simulation    =   orbit_file[mcarlo]['t'][simulation_initial_time_step:simulation_end_time_step]
    xt_pert         =   orbit_file[mcarlo]['xt'][simulation_initial_time_step:simulation_end_time_step]
    yt_pert         =   orbit_file[mcarlo]['yt'][simulation_initial_time_step:simulation_end_time_step]
    zt_pert         =   orbit_file[mcarlo]['zt'][simulation_initial_time_step:simulation_end_time_step]
    
    trajectory_coeffs = np.zeros((3,polynomial_degree+1))

    trajectory_coeffs[0,:]=np.polyfit(t_simulation,xt_pert,polynomial_degree)
    trajectory_coeffs[1,:]=np.polyfit(t_simulation,yt_pert,polynomial_degree)
    trajectory_coeffs[2,:]=np.polyfit(t_simulation,zt_pert,polynomial_degree)
    
    return trajectory_coeffs


def get_approx_impact_time_with_stream_density_filter(\
    taufile,forceFile,perturberName,density_min=4e-4):
    """
    Get the approximate impact time with a stream density filter.

    Parameters:
    -----------
    None

    Returns:
    --------
    float
        The approximate impact time.
    """
    #### IMPORT THE FORCE FILE
    X,Y,C=DE.extract_acceleration_arrays_from_force_file(forceFile, perturberName)
    X_tstamps,Y_tau,density_array=CSD.construct_2D_density_map(\
        taufile['tau'][:],taufile['time_stamps'][:],taufile.attrs['period'])
    _,_,C_masked=CSD.get_envelope_and_mask(X_tstamps[0],Y_tau[:,0],density_array,X,Y,C,density_min)
    time_dex,tau_dex=np.unravel_index(np.nanargmax(C_masked),C_masked.shape)
    approx_impact_time = X[0][time_dex]
    approx_impact_tau = Y[tau_dex][0]
    return approx_impact_time,approx_impact_tau

###################################################
################ PURE COMPUTATIONS ################
###################################################    
def get_3D_parabola_stream_snapshot_coefficients(stream_time_coordinate,stream_galactic_coordinates):
    """
    This function fits a 3D parabola to the stream at a given time.
    
    Parameters:
    stream_time_coordinate (np.ndarray): The time coordinate of the stream.
    stream_galactic_coordinates (list): each element is an np.ndarray
    
    """ 
    ########### SET BASE PARAMETERS ###########
    minimization_method =   'Nelder-Mead'
    
    n_stream_samplings=stream_time_coordinate.shape[0]
    assert n_stream_samplings==len(stream_galactic_coordinates)

    ############### PERFORM THE FIT AT TIME OF INTEREST ###############
    for i in range(n_stream_samplings):
        initial_guess=PSF.constrain_3D_parabola_initial_guess(
                                stream_time_coordinate[i],
                                stream_galactic_coordinates[i][0:3,:])
        
        results=optimize.minimize(  PSF.objective_parametric_3D_parabola,
                                    initial_guess,
                                    args=(stream_time_coordinate[i],
                                        stream_galactic_coordinates[i][0:3,:]),
                                    method=minimization_method,)
        
        if i==0:
            temporal_coefficients_array=np.zeros((n_stream_samplings,len(results.x)))
        
        temporal_coefficients_array[i,:]=results.x

    return temporal_coefficients_array



def filter_stream_about_suspected_impact_time(stream_tail_coordinates:np.ndarray,
                                tau_suspected:float,
                                xmin:float=0.5,
                                xlim:float=15,
                                ylim:float=0.5,
                                zlim:float=0.5) -> np.ndarray:
    """
    Filters the stream to only include the one of the two tails. Pick the side the the GC is on. 
        Also, filter to only include a given range of the stream based on the xlim,ylim,zlim
        additionally, xmin above zero clips the stars that are still bound to the globular cluster

    Parameters:
    - stream_tail_coordinates (np.ndarray): The coordinates of the stream tail.
    - perturber_tail_coordinates (np.ndarray): The coordinates of the perturber tail.
    - xmin (float, optional): The minimum x-coordinate value. Defaults to 0.5. Intended to remove the globular cluster, only taking the stream.
    - xlim (float, optional): The maximum x-coordinate value. Defaults to 15.
    - ylim (float, optional): The maximum y-coordinate value. Defaults to 0.5.
    - zlim (float, optional): The maximum z-coordinate value. Defaults to 0.5.

    Returns:
    - my_filter (np.ndarray): A boolean array indicating which elements of the stream should be included.

    """
    # filter1 = DE.filter_impacted_stream_side(stream_tail_coordinates[0], perturber_tail_coordinates[0,0])
    if tau_suspected>0:
        filter1 = stream_tail_coordinates[0]>0
    else:
        filter1 = stream_tail_coordinates[0]<0
    filter2 = DE.filter_stream_in_tail_coordinates(stream_tail_coordinates, xlim, ylim, zlim)
    filter3 = np.abs(stream_tail_coordinates[0]) > xmin
    my_filter = filter1 & filter2 & filter3
    return my_filter


def minimize_distance_between_stream_and_perturber(stream_coordinate_range,
                                                   simulation_sampling_time_stamps,
                                                   coefficient_time_fit_params,
                                                   trajectory_coeffs,
                                                   minimization_method='Nelder-Mead'):
    """
    Minimizes the distance between a stream and a perturber.

    Parameters:
    -----------
    stream_coordinate_range : array-like
        Range of stream coordinates.
    simulation_sampling_time_stamps : array-like
        Time stamps of the simulation sampling.
    coefficient_time_fit_params : array-like
        Coefficient time fit parameters.
    trajectory_coeffs : array-like
        Trajectory coefficients.
    minimization_method : str, optional
        Method used for minimization. Default is 'Nelder-Mead'.

    Returns:
    --------
    results : OptimizeResult
        The optimization result.

    """
    # make initial guess for the time 
    t0 = np.mean(simulation_sampling_time_stamps)
    s0 = np.mean(stream_coordinate_range)
    p0 = (s0, t0)
    results = optimize.minimize(
        PSF.objective_distance_perturber_and_moving_stream,
        p0,
        args=(coefficient_time_fit_params, trajectory_coeffs),
        method=minimization_method)
    return results

 
def half_mass_to_plummer_radius(half_mass_radius):
    """
    Convert half-mass radius to Plummer radius.

    Parameters:
    -----------
    half_mass_radius : float
        The half-mass radius.

    Returns:
    --------
    float
        The Plummer radius.
    """
    factor = (1/(1/2)**(2/3)-1)**(1/2)
    return half_mass_radius*factor


if __name__=="__main__":
    # mcarlo_index=sys.argv[1]
    # perturber_name=sys.argv[2]
    
    mcarlo_index=19
    perturber_name = "NGC2808"
    main(mcarlo_index,perturber_name)
    # # for output in geometry_erkal.keys():
    #     # print(output,geometry_erkal[output])
    # # for output in parameters_stream_and_perturber.keys():
    #     # print(output,parameters_stream_and_perturber[output])
