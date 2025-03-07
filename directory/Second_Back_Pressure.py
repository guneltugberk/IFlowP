import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import os
from directory.GasProperties import GasProperties 

# -------------------------
# 1. Page Configuration & Theming
# -------------------------
st.set_page_config(
    page_title="Second Back Pressure IPR Prediction",
    page_icon="⛽",
    layout="wide"
)

# Custom CSS for styling 
st.markdown(
    """
    <style>
        /* Sidebar custom label */
        [data-testid="stSidebarNav"]::before {
            content: "IFP School";
            font-family: 'Comic Sans MS', sans-serif;
            margin-left: 100px;
            margin-top: 30px;
            font-size: 21px;
            position: relative;
            top: 5px;
            text-align: center;
            font-weight: bold;
        }
        /* Logo and title styling */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .logo-img {
            width: 120px;
            height: auto;
            margin-right: 10px;
        }
        .stTitle {
            font-size: 40px;
            font-weight: bold;
            color: #0256FE;
            margin-bottom: 20px;
            text-align: center;
            font-family: 'Comic Sans MS', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# 2. Logo & Title Display
# -------------------------
st.markdown(
    """
    <div class="logo-container">
        <img class="logo-img" src="https://upload.wikimedia.org/wikipedia/commons/6/69/IFP_Logo.png">
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='stTitle'>Second Back Pressure IPR Prediction</div>", unsafe_allow_html=True)

# -------------------------
# 3. Caching: Load the PCA Model and PINN
# -------------------------
@st.cache_resource
def load_pca_model():
    # Call the pca scaler and pca
    time.sleep(1)

    base_dir = os.path.dirname(__file__)
    scaler_pca_path = os.path.join(base_dir, "scaler_pca.pkl")
    pca_path = os.path.join(base_dir, "pca.pkl")

    # Load the pca scaling function
    with open(scaler_pca_path, "rb") as f:
        scaler_pca = pickle.load(f)

    # Load the pca model
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    # Calling the features in the same order as given to the scaling model
    features_pca = df[["Pr, psi", "Temperature, F", "k, md", "Porosity, fraction", "h, ft", "SG", "Bg, bbl/scf",
                            "Gas Viscosity, cp", "Gas Density, lb/ft3"]].values
    
    # Scaling the features before the PCA
    features_pca_scaled = scaler_pca.fit_transform(features_pca)

    # Transforming the dataset using fitted PCA
    pca_data = pca.transform(features_pca_scaled)

    return pca_data

@st.cache_resource
def load_pinn_model(Pr, Pwf, SG, T):
    # Simulate a model load delay and return a dummy prediction function.
    time.sleep(1)

    # Compute the real gas pseudo-pressure
    real_gas = GasProperties(gamma=SG, Pressure=Pr, Temperature=T)
    real_gas_pseudo_pressure = [real_gas.real_gas_pseudo_pressure(Pwf=p) for p in Pwf]
    
    def dummy_predict(input_df):
        # Prepare the flow rates
        pwf = input_df['Pwf, psi'].values

        # Prepare the model inputs

        # This please will be replaced by the actual model.
        return (input_df["Pr, psi"] - input_df["Pwf, psi"]) * 10
    return dummy_predict

predict_flow = load_pinn_model()

# -------------------------
# 4. Sidebar Information
# -------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This tool uses a PINN model optimized for the **first back pressure equation** to predict gas flow rate based on different Pwf values and reservoir properties.
        Adjust the inputs on the main page and click **Predict**.
        """
    )

# -------------------------
# 5. Toggle for PVT Data
# -------------------------
pvt_available = st.toggle("PVT Data Available", value=False)

# -------------------------
# 6. Input Form
# -------------------------
with st.form("prediction_form"):
    st.subheader("Reservoir & Fluid Properties")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pr = st.number_input('Reservoir Pressure, psi', min_value=1000.0, max_value=10000.0, value=3000.0)
        Tr = st.number_input('Temperature, °F', min_value=77.0, max_value=350.0, value=150.0)
    with col2:
        SG = st.number_input('Gas Specific Gravity', min_value=0.58, max_value=0.9, value=0.7)
        PHI = st.number_input('Porosity (fraction)', min_value=0.05, max_value=0.4, value=0.2)
    with col3:
        PERM = st.number_input('Permeability, md', min_value=1.0, max_value=1000.0, value=100.0)
        h = st.number_input('Net Thickness, ft', min_value=5.0, max_value=300.0, value=50.0)
    
    # Use the toggle's state to decide which PVT input section to show
    if pvt_available:
        st.subheader("PVT Data")
        Bg = st.number_input('Bg, bbl/scf', min_value=0.0001, max_value=0.01, value=0.0015, format="%.6f")
        Viscosity = st.number_input('Viscosity, cp', min_value=0.001, max_value=0.09, value=0.02, format="%.4f")
        rho = st.number_input('Gas Density, lb/ft³', min_value=5.0, max_value=20.0, value=10.0)

    else:
        st.info("The gas properties will be estimated by correlations!")
        gas = GasProperties(gamma=SG, Pressure=Pr, Temperature=Tr)
        Bg = gas.Bg()
        Viscosity = gas.viscosity_gas()
        rho = gas.gas_density()
    
    submitted = st.form_submit_button("Predict")

# -------------------------
# 7. Prediction & Output Display
# -------------------------
if submitted:
    # Loading the scalers, PCA and PINN models

    with st.spinner("Predicting gas flow rate..."):
        # Generate an array of Pwf values from Pr down to AOF (in steps of -50 psi)
        Pwf_vals = np.arange(Pr, 14.7 - 1, -50)
        
        # Build a DataFrame with repeated property values and varying Pwf values.
        df = pd.DataFrame({
            "Pr, psi": np.full(len(Pwf_vals), Pr),
            "Temperature, F": np.full(len(Pwf_vals), Tr),
            "k, md": np.full(len(Pwf_vals), PERM),
            "Porosity, fraction": np.full(len(Pwf_vals), PHI),
            "h, ft": np.full(len(Pwf_vals), h),
            "SG": np.full(len(Pwf_vals), SG),
            "Bg, bbl/scf": np.full(len(Pwf_vals), Bg),
            "Gas Viscosity, cp": np.full(len(Pwf_vals), Viscosity),
            "Gas Density, lb/ft3": np.full(len(Pwf_vals), rho),
            "Pwf, psi": Pwf_vals
        })
        
        # Run prediction using the cached PINN model
        df["Qg, Mscf/d"] = predict_flow(df)
        time.sleep(1)  

    st.success("Prediction complete!")
    
    # Use st.metric to highlight the key result (e.g., the AOF predicted Qg)
    st.metric(
        label="Predicted AOF Gas Flow (Mscf/d)",
        value=f"{df['Qg, Mscf/d'].iloc[-1]:.2f}",
        delta=f"{df['Qg, Mscf/d'].max() - df['Qg, Mscf/d'].iloc[-1]:.2f}"
    )
    
    # Display prediction results in an interactive table
    st.dataframe(df)
    
    # Display an interactive chart: Qg vs. Pwf
    st.line_chart(data=df.set_index("Pwf, psi")[["Qg, Mscf/d"]])
    
    # Provide a download button for the results
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    # Optional toast notification to enhance interactivity
    st.toast("Your predictions are ready!", icon="✅")
