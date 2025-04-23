# Stream Visualization App

This project is a Streamlit application designed to allow users to select input parameters for simulations, generate corresponding file names for plots, and visualize these plots in a user-friendly interface.

## Project Structure

```
stream-visualization-app
├── src
│   ├── app.py                # Main entry point of the Streamlit application
│   ├── components
│   │   ├── __init__.py       # Marks the components directory as a Python package
│   │   ├── input_forms.py     # Contains input forms for user parameter selection
│   │   └── plot_viewer.py     # Responsible for displaying plots based on user input
│   ├── utils
│   │   ├── __init__.py       # Marks the utils directory as a Python package
│   │   ├── file_handler.py    # Functions for handling file operations
│   │   └── path_generator.py   # Functions for generating file paths for plots
│   └── config.py             # Configuration settings for the application
├── data
│   └── sample_data.yaml      # Sample data in YAML format for testing
├── requirements.txt           # Lists dependencies required for the project
├── .streamlit
│   └── config.toml           # Configuration settings for the Streamlit application
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stream-visualization-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Features

- User-friendly interface for selecting simulation parameters.
- Dynamic generation of plot file names based on user input.
- Visualization of plots corresponding to the selected parameters.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.