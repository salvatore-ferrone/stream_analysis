import streamlit as st
from components.input_forms import create_input_form
from components.plot_viewer import display_plot
from utils.path_generator import generate_plot_path

def main():
    st.title("Stream Visualization App")
    
    # Create input form for user parameters
    user_params = create_input_form()
    
    if user_params:
        # Generate plot file path based on user input
        plot_file_path = generate_plot_path(user_params)
        
        # Display the plot if the file exists
        display_plot(plot_file_path)

if __name__ == "__main__":
    main()