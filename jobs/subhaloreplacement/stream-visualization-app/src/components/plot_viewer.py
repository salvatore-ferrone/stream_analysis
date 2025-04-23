import streamlit as st
import os
from utils.file_handler import load_plot
from utils.path_generator import generate_plot_path

def plot_viewer(selected_params):
    plot_file = generate_plot_path(selected_params)
    
    if os.path.exists(plot_file):
        st.image(plot_file, caption='Simulation Plot', use_column_width=True)
    else:
        st.warning("Plot file does not exist. Please check your parameters and try again.")