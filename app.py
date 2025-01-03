import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

# Page configuration
st.set_page_config(
    page_title="Tourette Syndrome Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 初始化 session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def train_model():
    """训练模型并保存到 session state"""
    try:
        # 读取数据
        data = pd.read_csv('python.csv')
        st.success(f"数据加载成功，形状: {data.shape}")

        # 准备特征和目标变量
        X = data.drop('Group', axis=1)
        y = data['Group']
        
        # 保存特征名称
        st.session_state.features = list(X.columns)

        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 创建管道
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        # 训练模型
        pipeline.fit(X_train, y_train)
        
        # 评估模型
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        st.success(f"模型训练完成！\n训练集准确率: {train_score:.4f}\n测试集准确率: {test_score:.4f}")
        
        # 保存到 session state
        st.session_state.pipeline = pipeline
        st.session_state.model_trained = True
        
        return True
        
    except Exception as e:
        st.error(f"训练过程出错: {str(e)}")
        return False

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
    
    # 添加训练模型的部分
    st.sidebar.title("Model Training")
    if st.sidebar.button("Train Model"):
        train_model()
    
    if not st.session_state.model_trained:
        st.warning("请先在侧边栏点击 'Train Model' 按钮训练模型")
        return

    st.markdown("""
    ### Instructions
    1. Enter values below
    2. Click "Predict"
    3. View results
    """)

    # 创建输入字段
    inputs = create_input_fields(st.session_state.features)

    # 预测按钮
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner('Analyzing...'):
            try:
                # 数据处理和预测
                input_df = pd.DataFrame([inputs])
                prediction_proba = float(st.session_state.pipeline.predict_proba(input_df)[0][1])
                
                # 显示结果
                st.markdown("---")
                display_results(prediction_proba)

                # SHAP分析
                st.subheader("Feature Impact Analysis")
                # 获取预处理后的数据和模型
                input_processed = st.session_state.pipeline.named_steps['scaler'].transform(input_df)
                model = st.session_state.pipeline.named_steps['classifier']
                
                shap_values = calculate_shap_values(model, input_processed)
                
                fig = plt.figure(figsize=(12, 4))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()

