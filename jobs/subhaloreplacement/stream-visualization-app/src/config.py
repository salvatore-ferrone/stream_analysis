import os

# Configuration settings for the Streamlit application
class Config:
    DEFAULT_PLOT_DIR = os.path.join("data", "plots")
    DEFAULT_SAMPLE_DATA_FILE = os.path.join("data", "sample_data.yaml")
    DEFAULT_MASS_RANGE = (1e4, 1e5)  # Mass range in Msun
    DEFAULT_RADIUS_RANGE = (0.002, 0.03)  # Radius range in kpc
    DEFAULT_NUM_EXPERIMENTS = 150

    @staticmethod
    def get_plot_file_name(mass, radius, experiment_index):
        return f"plot_mass_{mass}_radius_{radius}_experiment_{experiment_index}.png"