import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

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
    with open('ts_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, preprocessor, features

def main():
    st.title("🏥 Tourette Syndrome Risk Assessment System")
    
    # Add instructions
    st.markdown("""
    ### Instructions
    1. Enter the test indicator values below
    2. Click "Predict" button to get assessment results
    3. System will display Tourette Syndrome risk probability and factor analysis
    """)

    try:
        # Load model components
        model, preprocessor, features = load_model_and_preprocessor()

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
                # Convert inputs to DataFrame
                input_df = pd.DataFrame([inputs])
                
                try:
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
                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(model)
                        
                        # Calculate SHAP values
                        shap_values = explainer.shap_values(input_processed)
                        
                        # Select positive class SHAP values for binary classification
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        
                        # Create force plot
                        plt.figure(figsize=(12, 4))
                        shap.force_plot(
                            explainer.expected_value if not isinstance(explainer.expected_value, list) 
                            else explainer.expected_value[1],
                            shap_values[0],
                            input_processed[0],
                            feature_names=list(features),
                            matplotlib=True,
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        # Add explanation text
                        st.markdown("""
                        **Plot Interpretation**:
                        - Red indicates features increasing TS risk
                        - Blue indicates features decreasing TS risk
                        - Bar width represents feature impact magnitude
                        - Base value shows model's average prediction
                        """)
                        
                    except Exception as e:
                        st.error(f"Error in SHAP analysis: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    
    except Exception as e:
        st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
