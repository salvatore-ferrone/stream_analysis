def generate_plot_filename(gc_name, internal_dynamics, stream_potential, monte_carlo_index):
    """
    Generate a unique filename for the plot based on the input parameters.
    
    Parameters:
    - gc_name (str): The name of the globular cluster.
    - internal_dynamics (str): The type of internal dynamics.
    - stream_potential (str): The potential environment of the stream.
    - monte_carlo_index (int): The index for the Monte Carlo simulation.
    
    Returns:
    - str: The generated filename.
    """
    return f"{gc_name}_{internal_dynamics}_{stream_potential}_mc{monte_carlo_index}.png"

def generate_output_directory(base_path, stream_potential, gc_name, internal_dynamics, monte_carlo_index):
    """
    Generate the output directory path for saving plots based on input parameters.
    
    Parameters:
    - base_path (str): The base path for saving plots.
    - stream_potential (str): The potential environment of the stream.
    - gc_name (str): The name of the globular cluster.
    - internal_dynamics (str): The type of internal dynamics.
    - monte_carlo_index (int): The index for the Monte Carlo simulation.
    
    Returns:
    - str: The generated output directory path.
    """
    return os.path.join(base_path, stream_potential, gc_name, internal_dynamics, f"monte-carlo-{str(monte_carlo_index).zfill(3)}")