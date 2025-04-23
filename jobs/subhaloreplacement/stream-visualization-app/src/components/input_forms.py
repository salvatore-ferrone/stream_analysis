from streamlit import st

def input_parameters():
    st.header("Simulation Input Parameters")

    # User input for mass
    mass = st.number_input("Select Perturber Mass (M☉)", min_value=1, max_value=1e6, value=1000)

    # User input for radius
    radius = st.number_input("Select Perturber Radius (kpc)", min_value=0.1, max_value=100.0, value=10.0)

    # User input for internal dynamics
    internal_dynamics = st.selectbox("Select Internal Dynamics", ["Option 1", "Option 2", "Option 3"])

    # User input for GC name
    gc_name = st.text_input("Enter GC Name", value="Pal5")

    # Generate file name based on inputs
    if st.button("Generate File Name"):
        file_name = f"{gc_name}_mass{mass}_radius{radius}_internal{internal_dynamics}.png"
        st.success(f"Generated File Name: {file_name}")

    return mass, radius, internal_dynamics, gc_name