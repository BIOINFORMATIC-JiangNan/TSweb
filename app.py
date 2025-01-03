import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# Page configuration
st.set_page_config(
    page_title="Tourette Syndrome Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the saved model, preprocessor, and feature names from pickle files
    Returns: model, preprocessor, features
    """
    try:
        # Safe loading method for model components
        with open('ts_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
            
        # Validate model type
        if not isinstance(model, GradientBoostingClassifier):
            st.error("Model type mismatch. Please ensure using GradientBoostingClassifier")
            return None, None, None
            
        return model, preprocessor, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def main():
    st.title("🏥 Tourette Syndrome Risk Assessment System")
    
    # Add instructions
    st.markdown("""
    ### Instructions
    1. Enter the test indicator values below
    2. Click "Predict" button to get assessment results
    3. System will display Tourette Syndrome risk probability and factor analysis
    """)

    # Load model components
    model, preprocessor, features = load_model_and_preprocessor()
    
    if model is None or preprocessor is None or features is None:
        st.error("Model loading failed. Please check model files")
        return

    # Create two-column layout
    col1, col2 = st.columns(2)

    # Split features into two halves
    half = len(features) // 2
    inputs = {}
    
    # First column inputs
    with col1:
        st.subheader("Indicators Input (1/2)")
        for feature in features[:half]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                help=f"Enter value for {feature}"
            )

    # Second column inputs
    with col2:
        st.subheader("Indicators Input (2/2)")
        for feature in features[half:]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                help=f"Enter value for {feature}"
            )

    # Prediction button
    if st.button("Predict", type="primary"):
        with st.spinner('Analyzing...'):
            try:
                # Convert inputs to DataFrame
                input_df = pd.DataFrame([inputs])
                
                # Preprocess input data
                input_processed = preprocessor.transform(input_df)
                
                # Make prediction
                prediction_proba = float(model.predict_proba(input_processed)[0][1])
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Three-column layout for results
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric(
                        label="TS Risk Probability",
                        value=f"{prediction_proba:.1%}"
                    )
                
                with result_col2:
                    risk_level = "High Risk" if prediction_proba > 0.5 else "Low Risk"
                    st.metric(
                        label="Risk Level",
                        value=risk_level,
                        delta="Needs Attention" if prediction_proba > 0.5 else "Good Status"
                    )
                
                with result_col3:
                    st.progress(prediction_proba)

                # SHAP value explanation
                st.subheader("Feature Impact Analysis")
                
                try:
                    # Calculate SHAP values using safer method
                    explainer = shap.TreeExplainer(model, feature_names=list(features))
                    shap_values = explainer(input_processed)
                    
                    # Create SHAP visualization
                    fig = plt.figure(figsize=(12, 4))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
                    plt.clf()
                    
                    st.markdown("""
                    **Plot Interpretation**:
                    - Red indicates features increasing TS risk
                    - Blue indicates features decreasing TS risk
                    - Bar width represents feature impact magnitude
                    """)
                    
                except Exception as e:
                    st.error(f"SHAP analysis error: {str(e)}")
                    
            except Exception as e:
                st.error(f"Prediction process error: {str(e)}")

if __name__ == "__main__":
    main()
