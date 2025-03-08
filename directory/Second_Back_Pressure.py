import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import os
from directory.GasProperties import GasProperties 
import plotly.graph_objects as go

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

# Define PINN
class GasFlowModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_logC = nn.Linear(64, 1)  # Predicts log10(C)
        self.output_n    = nn.Linear(64, 1)  # Predicts n

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        logC = self.output_logC(x)
        n = F.sigmoid(self.output_n(x)) * 0.5 + 0.5  

        return logC, n

# -------------------------
# 3. Caching: Load the PCA Model and PINN
# -------------------------
@st.cache_resource
def load_prediction_objects():
    base_dir = os.path.dirname(__file__)
    
    # Load PCA-related objects
    with open(os.path.join(base_dir, "scaler_pca.pkl"), "rb") as f:
        scaler_pca = pickle.load(f)
    with open(os.path.join(base_dir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    
    # Load the scaler for PCA outputs
    with open(os.path.join(base_dir, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    
    # Load the trained PINN model
    model_path = os.path.join(base_dir, "second_pressure.pth")

    # Determine input dimension from scaler_X (number of features after PCA)
    input_dim = 6
    model = GasFlowModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return scaler_pca, pca, scaler_X, model

def predict_flow(df):
    # Load preprocessing objects and model
    scaler_pca, pca, scaler_X, model = load_prediction_objects()
    
    # Extract features for PCA transformation
    feature_cols = ["Pr, psi", "Temperature, F", "k, md", "Porosity, fraction",
                    "h, ft", "SG", "Bg, bbl/scf", "Gas Viscosity, cp", "Gas Density, lb/ft3"]
    
    X_raw = df[feature_cols].values

    # IMPORTANT: Use transform, not fit_transform, so that the same parameters are used.
    X_scaled_for_pca = scaler_pca.transform(X_raw)
    X_pca = pca.transform(X_scaled_for_pca)

    # Scale PCA output as done during training
    X_model = scaler_X.transform(X_pca)
    
    # Convert features to torch tensor
    X_tensor = torch.tensor(X_model, dtype=torch.float32)
    
    model.eval()

    # Get model predictions (logC and n)
    with torch.inference_mode():
        logC, n = model(X_tensor)
    
    # Compute real gas pseudo pressure.
    dp_list = []
    
    for i, row in df.iterrows():
        # Reservoir and fluid properties
        Pr = row["Pr, psi"]
        T  = row["Temperature, F"]
        SG = row["SG"]
        Pwf = row["Pwf, psi"]

        gas = GasProperties(gamma=SG, Pressure=Pr, Temperature=T)
        dp_val = gas.real_gas_pseudo_pressure(Pwf=Pwf)

        dp_list.append(dp_val)

    dp_array = np.array(dp_list)
    
    # Compute predicted log_q using the physics-informed relationship:
    # log_q_pred = logC + n * dp
    logC_np = logC.squeeze().cpu().numpy()
    n_np = n.squeeze().cpu().numpy()
    
    C_pred = 10 ** logC_np[-1]
    n_pred = n_np[-1]

    Q_pred = C_pred * (dp_array ** n_pred)
    
    return Q_pred, C_pred, n_pred, dp_array

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
    with st.status("In progress..", expanded=True) as status:
        st.write("Validation...")
        time.sleep(1)
        st.write("Scaling the inputs...")
        time.sleep(0.5)
        st.write("Predicting...")
        time.sleep(0.5)
        status.update(label="Predicted!", state="complete", expanded=False)
        
        # Create a DataFrame with one row per Pwf value (or however you wish to generate predictions)
        # For example, generate a range of Pwf values from Pr downwards:
        Pwf_vals = np.arange(Pr, 14.7 - 1, -100)
        # Build the input DataFrame; note that all columns must be provided.
        input_df = pd.DataFrame({
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
        
        # Use the prediction function
        Q_pred, C_pred, n_pred, dp = predict_flow(input_df)
        st.write("C: ", C_pred)
        st.write("n: ", n_pred)
        
        # Add predictions to your DataFrame for display
        input_df["Qg, Mscf/d"] = Q_pred / 1000
        input_df["DP"] = dp
        
        time.sleep(1)  # Simulate delay if desired

    st.success("Prediction complete!")
    
    # Display a key metric (for example, the prediction corresponding to the lowest Pwf)
    st.metric(
        label="Predicted AOF Gas Flow (Mscf/d)",
        value=f"{round(input_df['Qg, Mscf/d'].iloc[-1], 0)}"
    )
    
    # Show the results in a table and chart
    st.dataframe(input_df)
    # Set the MAE value (given as 1086)
    mae = 1086

    # Extract Pwf and Qg values from the predictions DataFrame
    # (Make sure the DataFrame 'df' has these columns from your prediction step.)
    Pwf_vals = input_df["Pwf, psi"].tolist()
    Qg_vals = input_df["Qg, Mscf/d"].tolist()

    # Compute the lower and upper bounds for the confidence interval on Qg
    Qg_lower = [max(0, q - mae) for q in Qg_vals]  # avoid negative values
    Qg_upper = [q + mae for q in Qg_vals]

    # Create the Plotly figure
    fig = go.Figure()

    # 1) Plot the predicted Qg line (x-axis: Qg, y-axis: Pwf)
    fig.add_trace(go.Scatter(
        x=Qg_vals,
        y=Pwf_vals,
        mode='lines+markers',
        name='Predicted Qg',
        line=dict(color='blue', width=2)
    ))

    # 2) Add an invisible trace for the upper bound to set up the fill
    fig.add_trace(go.Scatter(
        x=Qg_upper,
        y=Pwf_vals,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 1)', width=1),  # visible red border
        showlegend=False
    ))

    # 3) Add the lower bound trace and fill between it and the upper bound
    fig.add_trace(go.Scatter(
        x=Qg_lower,
        y=Pwf_vals,
        mode='lines',
        fill='tonextx',  # fill horizontally between this trace and the previous trace
        fillcolor='rgba(255, 0, 0, 0.4)',  # brighter red with some transparency
        line=dict(color='rgba(255, 0, 0, 1)', width=1),  # visible red border
        name='MAE Confidence Interval'
    ))

    # Update layout labels
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Qg, MSCF/D', font=dict(size=20)),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(text='Pwf, psi', font=dict(size=20)),
            tickfont=dict(size=14)
        )
    )
    fig.update_xaxes(range=[0, max(Qg_vals)+1000])
    fig.update_yaxes(range=[0, max(Pwf_vals)])

    # Display the figure in the Streamlit app
    st.plotly_chart(fig)
    
    # Provide a download button for the CSV
    csv_data = input_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.toast("Your predictions are ready!", icon="✅")
