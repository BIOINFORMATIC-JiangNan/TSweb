import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load
import os

# Page configuration
st.set_page_config(
    page_title="Tourette Syndrome Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 优化模型加载
@st.cache_resource(ttl=3600)
def load_model_components():
    """Load model components with optimized caching"""
    try:
        # 加载模型管道
        pipeline = load('ts_pipeline.joblib')
        
        # 加载特征名称
        features = load('feature_names.joblib')
        
        return pipeline, features
        
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        return None, None

# 优化SHAP计算
@st.cache_data(ttl=3600)
def calculate_shap_values(model, input_processed):
    """Cache SHAP calculations"""
    explainer = shap.TreeExplainer(model)
    return explainer(input_processed)

def create_input_fields(features):
    """创建输入字段"""
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
    """显示预测结果"""
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

    # 加载模型组件
    pipeline, features = load_model_components()
    
    if not all([pipeline, features]):
        return

    # 创建输入字段
    inputs = create_input_fields(features)

    # 预测按钮
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner('Analyzing...'):
            try:
                # 数据处理和预测
                input_df = pd.DataFrame([inputs])
                prediction_proba = float(pipeline.predict_proba(input_df)[0][1])
                
                # 显示结果
                st.markdown("---")
                display_results(prediction_proba)

                # SHAP分析
                st.subheader("Feature Impact Analysis")
                # 获取预处理后的数据和模型
                input_processed = pipeline.named_steps['preprocessor'].transform(input_df)
                model = pipeline.named_steps['classifier']
                
                shap_values = calculate_shap_values(model, input_processed)
                
                fig = plt.figure(figsize=(12, 4))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
