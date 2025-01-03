import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="Tourette Syndrome Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize model
@st.cache_resource
def init_model():
    # Load data
    data = pd.read_csv('python.csv')
    
    # Prepare features and target
    X = data.drop('Group', axis=1)
    y = data['Group']
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
    
    pipeline.fit(X, y)
    return pipeline, list(X.columns)

# Optimize SHAP calculations
@st.cache_data
def calculate_shap_values(_model, _input_processed, feature_names):
    """Cache SHAP calculations"""
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(_input_processed)
    # 设置特征名称
    shap_values.feature_names = feature_names
    return shap_values

def create_input_fields(features):
    """Create input fields"""
    inputs = {}
    
    col1, col2 = st.columns(2)
    half = len(features) // 2
    
    with col1:
        st.subheader("Indicators Input (1/2)")
        for feature in features[:half]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                key=f"input_{feature}"
            )
    
    with col2:
        st.subheader("Indicators Input (2/2)")
        for feature in features[half:]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f",
                key=f"input_{feature}"
            )
    
    return inputs

def display_results(prediction_proba):
    """Display prediction results"""
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

def main():
    st.title("🏥 Tourette Syndrome Risk Assessment System")
    
    st.markdown("""
    ### Instructions
    1. Enter values below
    2. Click "Predict"
    3. View results
    """)

    # Initialize model
    pipeline, features = init_model()

    # Create input fields
    inputs = create_input_fields(features)

    # Predict button
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner('Analyzing...'):
            try:
                # Process data and predict
                input_df = pd.DataFrame([inputs])
                prediction_proba = float(pipeline.predict_proba(input_df)[0][1])
                
                # Display results
                st.markdown("---")
                display_results(prediction_proba)

                # SHAP analysis
                st.subheader("Feature Impact Analysis")
                # Get processed data and model
                input_processed = pipeline.named_steps['scaler'].transform(input_df)
                model = pipeline.named_steps['classifier']
                
                # 传入特征名称
                shap_values = calculate_shap_values(model, input_processed, features)
                
                fig = plt.figure(figsize=(12, 4))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
