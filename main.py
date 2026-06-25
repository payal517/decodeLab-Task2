

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_breast_cancer
import joblib
import os

# PAGE CONFIGURATION 
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA 
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_data()

#  LOAD MODEL 
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

#  SIDEBAR 
st.sidebar.markdown("""
<div style='text-align: center;'>
    <h1 style='font-size: 3em;'>🔬</h1>
    <h2 style='color: #e0e0e0;'>Breast Cancer</h2>
    <p style='color: #a0a0a0;'>Classification Tool</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Model status
if model is not None:
    st.sidebar.success("✅ Model Ready")
    st.sidebar.info("💡 **Best Model:** Random Forest")
    st.sidebar.info("📊 **Accuracy:** ~97%")
else:
    st.sidebar.error("❌ Model not found!")
    st.sidebar.info("Run: `python train_model.py` first")

st.sidebar.markdown("---")

# About section
with st.sidebar.expander("ℹ️ About Breast Cancer"):
    st.markdown("""
    - **Benign** = Non-cancerous tumor
    - **Malignant** = Cancerous tumor
    
    **Risk Factors:**
    - Age
    - Family history
    - Genetic mutations
    - Hormonal factors
    
    **Early detection saves lives!**
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🔬 DecodeLabs Project 2 | Batch 2026")

# MAIN PAGE 
st.title("🔬 Breast Cancer Diagnosis")
st.markdown("### *AI-Powered Classification Tool*")

st.markdown("""
<div style='background-color: #1e3a5f; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='color: #e0f2f1; margin: 0;'>
    🏥 This tool uses machine learning to classify breast tumors as 
    <strong style='color: #4caf50;'>Benign</strong> or 
    <strong style='color: #f44336;'>Malignant</strong> 
    based on 30 diagnostic features.
    </p>
</div>
""", unsafe_allow_html=True)

# TABS 
tab1, tab2, tab3 = st.tabs([
    "🧪 Diagnosis Tool",
    "📊 Data Explorer",
    "📈 Model Performance"
])

#  TAB 1: DIAGNOSIS TOOL 
with tab1:
    st.subheader("🧪 Enter Patient Measurements")
    st.markdown("Adjust the sliders below to input patient data")
    
    if model is None:
        st.error("❌ Model not found! Please run `python train_model.py` first.")
        st.code("python train_model.py", language="bash")
    else:
        # Create input sliders in 3 columns
        col1, col2, col3 = st.columns(3)
        
        input_features = []
        for i, feature in enumerate(feature_names):
            if i < 10:
                col = col1
            elif i < 20:
                col = col2
            else:
                col = col3
            
            with col:
                default_val = float(np.mean(X[:, i]))
                min_val = float(np.min(X[:, i]))
                max_val = float(np.max(X[:, i]))
                
                value = st.slider(
                    f"{feature[:22]}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.2f",
                    key=f"f_{i}"
                )
                input_features.append(value)
        
        # Predict button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🧪 Diagnose", type="primary", use_container_width=True)
        
        if predict_button:
            # Scale and predict
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Diagnosis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 0:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #1b5e20, #2e7d32); 
                                padding: 40px; border-radius: 15px; text-align: center;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                        <h1 style='color: #4caf50; font-size: 2.5em; margin: 0;'>✅ BENIGN</h1>
                        <p style='color: #a5d6a7; font-size: 18px; margin-top: 10px;'>Non-cancerous tumor</p>
                        <p style='color: #a5d6a7; font-size: 14px; margin-top: 5px;'>
                            Confidence: <strong>{:.1%}</strong>
                        </p>
                    </div>
                    """.format(probabilities[0]), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #4a1a1a, #6d1a1a); 
                                padding: 40px; border-radius: 15px; text-align: center;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                        <h1 style='color: #f44336; font-size: 2.5em; margin: 0;'>⚠️ MALIGNANT</h1>
                        <p style='color: #ef9a9a; font-size: 18px; margin-top: 10px;'>Cancerous tumor detected</p>
                        <p style='color: #ef9a9a; font-size: 14px; margin-top: 5px;'>
                            Confidence: <strong>{:.1%}</strong>
                        </p>
                    </div>
                    """.format(probabilities[1]), unsafe_allow_html=True)
            
            with col2:
                # Probability bar chart
                prob_df = pd.DataFrame({
                    'Diagnosis': ['Benign', 'Malignant'],
                    'Probability': probabilities
                })
                
                fig = px.bar(
                    prob_df,
                    x='Diagnosis',
                    y='Probability',
                    color='Diagnosis',
                    color_discrete_map={'Benign': '#4caf50', 'Malignant': '#f44336'},
                    title='Prediction Confidence',
                    template='plotly_dark',
                    text=prob_df['Probability'].apply(lambda x: f'{x:.1%}')
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    yaxis=dict(range=[0, 1], tickformat='.0%'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Medical disclaimer
            st.markdown("""
            <div style='background-color: #1a1a2e; padding: 10px; border-radius: 5px; margin-top: 20px;'>
                <p style='color: #888; font-size: 12px; text-align: center;'>
                ⚠️ This tool is for educational purposes only. Always consult with a healthcare professional.
                </p>
            </div>
            """, unsafe_allow_html=True)

#  TAB 2: DATA EXPLORER 
with tab2:
    st.subheader("📊 Explore Breast Cancer Dataset")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", X.shape[0])
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Benign", sum(y == 0))
    with col4:
        st.metric("Malignant", sum(y == 1))
    
    # Feature selection for visualization
    st.subheader("Interactive Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("X-axis Feature", feature_names, index=0)
    with col2:
        feature_y = st.selectbox("Y-axis Feature", feature_names, index=1)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Diagnosis'] = ['Benign' if i == 0 else 'Malignant' for i in y]
    
    # Scatter plot
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color='Diagnosis',
        color_discrete_map={'Benign': '#4caf50', 'Malignant': '#f44336'},
        title=f'{feature_x} vs {feature_y}',
        template='plotly_dark',
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data sample
    with st.expander("📋 View Data Sample"):
        st.dataframe(df.head(20))
    
    # Statistics
    with st.expander("📊 Summary Statistics"):
        st.dataframe(df.describe())

# TAB 3: MODEL PERFORMANCE 
with tab3:
    st.subheader("📈 Model Performance")
    
    # Check if assets exist
    if os.path.exists('assets/confusion_matrix.png'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image('assets/confusion_matrix.png', caption='Confusion Matrix', use_container_width=True)
        
        with col2:
            st.image('assets/feature_importance.png', caption='Feature Importance', use_container_width=True)
        
        # Display results
        st.subheader("Model Comparison Results")
        
        # Load results if saved
        if os.path.exists('assets/model_comparison.csv'):
            results_df = pd.read_csv('assets/model_comparison.csv')
            st.dataframe(results_df)
        else:
            st.info("📊 Run `python train_model.py` to generate model comparison results")
    else:
        st.info("📊 Run `python train_model.py` first to generate performance plots")
        st.code("python train_model.py", language="bash")

# FOOTER 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 10px;'>
    <p>🔬 DecodeLabs Project 2 | Batch 2026 | Built with ❤️</p>
    <p style='font-size: 12px;'>For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
