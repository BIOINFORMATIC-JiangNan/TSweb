# Import required libraries
# Streamlit for creating web applications
import streamlit as st
# Pandas for data manipulation and analysis
import pandas as pd
# NumPy for numerical operations
import numpy as np
# Pickle for model serialization
import pickle
# SHAP for model interpretability
import shap
# Matplotlib for plotting
import matplotlib.pyplot as plt
# GradientBoostingClassifier for the ML model
from sklearn.ensemble import GradientBoostingClassifier
# OS for file operations
import os

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Tourette Syndrome Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar by default for faster loading
)

# Optimize model loading with caching
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model_and_preprocessor():
    """Load model components with optimized caching"""
    files = {
        'ts_model.pkl': None,
        'preprocessor.pkl': None,
        'feature_names.pkl': None
    }
    
    try:
        # Batch load files
        for filename in files.keys():
            if not os.path.exists(filename):
                st.error(f"Missing file: {filename}")
                return None, None, None
            
            with open(filename, 'rb') as f:
                files[filename] = pickle.load(f)
        
        model = files['ts_model.pkl']
        preprocessor = files['preprocessor.pkl']
        features = files['feature_names.pkl']
        
        # Validate model type
        if not isinstance(model, GradientBoostingClassifier):
            st.error("Invalid model type")
            return None, None, None
            
        return model, preprocessor, features
        
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        return None, None, None

# Optimize SHAP calculations with caching
@st.cache_data(ttl=3600)
def calculate_shap_values(model, input_processed):
    """Cache SHAP calculations"""
    explainer = shap.TreeExplainer(model)
    return explainer(input_processed)

def create_input_fields(features, half):
    """Create input fields for features"""
    inputs = {}
    
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    # First column of inputs
    with col1:
        st.subheader("Indicators Input (1/2)")
        for feature in features[:half]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                key=f"input_{feature}"  # Add unique key for each input
            )
    
    # Second column of inputs
    with col2:
        st.subheader("Indicators Input (2/2)")
        for feature in features[half:]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                key=f"input_{feature}"  # Add unique key for each input
            )
    
    return inputs

def display_results(prediction_proba):
    """Display prediction results"""
    # Create three columns for results
    result_col1, result_col2, result_col3 = st.columns(3)
    
    # Display probability
    with result_col1:
        st.metric(
            label="TS Risk Probability",
            value=f"{prediction_proba:.1%}"
        )
    
    # Display risk level
    with result_col2:
        risk_level = "High Risk" if prediction_proba > 0.5 else "Low Risk"
        st.metric(
            label="Risk Level",
            value=risk_level,
            delta="Needs Attention" if prediction_proba > 0.5 else "Good Status"
        )
    
    # Display progress bar
    with result_col3:
        st.progress(prediction_proba)

def main():
    # Set up main title
    st.title("🏥 Tourette Syndrome Risk Assessment System")
    
    # Display instructions
    st.markdown("""
    ### Instructions
    1. Enter values below
    2. Click "Predict"
    3. View results
    """)

    # Load model components
    model, preprocessor, features = load_model_and_preprocessor()
    
    if not all([model, preprocessor, features]):
        return

    # Create input fields
    inputs = create_input_fields(features, len(features) // 2)

    # Prediction button
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner('Analyzing...'):
            try:
                # Process input data and make prediction
                input_df = pd.DataFrame([inputs])
                input_processed = preprocessor.transform(input_df)
                prediction_proba = float(model.predict_proba(input_processed)[0][1])
                
                # Display results
                st.markdown("---")
                display_results(prediction_proba)

                # SHAP analysis and visualization
                st.subheader("Feature Impact Analysis")
                shap_values = calculate_shap_values(model, input_processed)
                
                # Create and display SHAP plot
                fig = plt.figure(figsize=(12, 4))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig)  # Explicitly close figure to free memory
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Entry point of the application
if __name__ == "__main__":
    main()
