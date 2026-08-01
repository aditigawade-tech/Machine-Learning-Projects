import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnPulse AI | Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# APP-LIKE CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    /* Header Card */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .app-header h1 { color: #ffffff; margin: 0; font-size: 26px; font-weight: 700; }
    .app-header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 14px; }

    /* Prediction Card */
    .pred-card-churn {
        background-color: #fef2f2;
        border: 2px solid #f87171;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .pred-card-retain {
        background-color: #f0fdf4;
        border: 2px solid #4ade80;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling - Churn_Modelling.csv")
    df = df.dropna(subset=['CustomerId'])
    df['Exited_Label'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})
    df['IsActive_Label'] = df['IsActiveMember'].map({0: 'Inactive', 1: 'Active Member'})
    return df

@st.cache_resource
def train_model(df):
    # Preprocessing
    feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[feature_cols].copy()
    y = df['Exited']

    # Encode categorical variables
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    
    X['Geography'] = le_geo.fit_transform(X['Geography'])
    X['Gender'] = le_gender.fit_transform(X['Gender'])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Calculate Churn Risk Scores for full dataset
    X_full = df[feature_cols].copy()
    X_full['Geography'] = le_geo.transform(X_full['Geography'])
    X_full['Gender'] = le_gender.transform(X_full['Gender'])
    
    churn_probabilities = model.predict_proba(X_full)[:, 1]
    
    return model, le_geo, le_gender, acc, churn_probabilities

try:
    df = load_data()
    model, le_geo, le_gender, model_acc, churn_probs = train_model(df)
    df['Churn_Risk_Score'] = churn_probs
except Exception as e:
    st.error("Please place 'Churn_Modelling - Churn_Modelling.csv' in the root directory.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ ChurnPulse AI")
    st.caption(f"Model Accuracy: **{model_acc * 100:.1f}%**")
    st.markdown("---")
    
    st.subheader("Filter Dashboard")
    selected_geo = st.selectbox("Geography", ['All'] + list(df['Geography'].unique()))
    selected_gender = st.selectbox("Gender", ['All'] + list(df['Gender'].unique()))
    
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    selected_age = st.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    st.markdown("---")
    st.caption("Status: **AI Model Active**")

# Filtered DataFrame
filtered_df = df.copy()
if selected_geo != 'All':
    filtered_df = filtered_df[filtered_df['Geography'] == selected_geo]
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
filtered_df = filtered_df[
    (filtered_df['Age'] >= selected_age[0]) & (filtered_df['Age'] <= selected_age[1])
]

# -----------------------------------------------------------------------------
# HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>Banking Customer Churn & Risk Intelligence</h1>
    <p>Predict individual customer attrition risk and explore analytical insights driven by Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP METRICS BANNER
# -----------------------------------------------------------------------------
total_cust = len(filtered_df)
churned = len(filtered_df[filtered_df['Exited'] == 1])
churn_rate = (churned / total_cust * 100) if total_cust > 0 else 0
high_risk_count = len(filtered_df[filtered_df['Churn_Risk_Score'] >= 0.5])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered Customers", f"{total_cust:,}")
m2.metric("Historical Churned", f"{churned:,}")
m3.metric("Churn Rate", f"{churn_rate:.1f}%")
m4.metric("Predicted At-Risk Accounts", f"{high_risk_count:,}")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Single Customer", 
    "📊 Executive Overview", 
    "🚨 High Risk Accounts", 
    "📁 Raw Dataset"
])

# -----------------------------------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTOR (NEW FEATURE)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Details Input")
    st.write("Enter custom details below to run a real-time predictive risk assessment.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        geography = st.selectbox("Geography", df['Geography'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
        age = st.number_input("Age", min_value=18, max_value=100, value=38)

    with col_b:
        tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 1)

    with col_c:
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict Customer Outcome", use_container_width=True, type="primary"):
        # Map categorical variables
        has_card_val = 1 if has_card == "Yes" else 0
        is_active_val = 1 if is_active == "Yes" else 0
        
        geo_encoded = le_geo.transform([geography])[0]
        gender_encoded = le_gender.transform([gender])[0]

        # Single row input frame
        input_data = pd.DataFrame([[
            credit_score, geo_encoded, gender_encoded, age, 
            tenure, balance, num_products, has_card_val, 
            is_active_val, salary
        ]], columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

        # Prediction
        prediction = model.predict(input_data)[0]
        risk_probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="pred-card-churn">
                        <h2 style="color: #dc2626; margin: 0;">⚠️ High Risk</h2>
                        <h3 style="color: #991b1b; margin: 5px 0;">Customer Likely to EXIT</h3>
                        <p style="font-size: 20px; font-weight: bold;">Risk Score: {risk_probability * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="pred-card-retain">
                        <h2 style="color: #16a34a; margin: 0;">✅ Low Risk</h2>
                        <h3 style="color: #166534; margin: 5px 0;">Customer Likely to STAY</h3>
                        <p style="font-size: 20px; font-weight: bold;">Retention Probability: {(1 - risk_probability) * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.write("### Prediction Breakdown")
            progress_color = "#ef4444" if risk_probability >= 0.5 else "#22c55e"
            st.progress(float(risk_probability), text=f"Calculated Churn Risk: {risk_probability * 100:.1f}%")
            
            if risk_probability >= 0.5:
                st.warning("💡 **Actionable Advice:** Consider targeting this customer with engagement incentives, personalized retention offers, or a customer support outreach call.")
            else:
                st.success("💡 **Actionable Advice:** Customer exhibits strong retention signals. Good opportunity for cross-selling premium products.")

# -----------------------------------------------------------------------------
# TAB 2: OVERVIEW ANALYTICS
# -----------------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retention Status Proportions")
        fig_pie = px.pie(
            filtered_df, 
            names='Exited_Label', 
            hole=0.5,
            color='Exited_Label',
            color_discrete_map={'Retained': '#3b82f6', 'Churned': '#ef4444'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Model Feature Importance")
        importances = model.feature_importances_
        feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_fi, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: HIGH RISK ACCOUNTS (PREDICTIVE TARGETING)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🚨 Priority Intervention List (Predicted Risk > 60%)")
    high_risk_df = filtered_df[filtered_df['Churn_Risk_Score'] >= 0.6].sort_values(by='Churn_Risk_Score', ascending=False)
    
    st.write(f"Showing **{len(high_risk_df)}** accounts that the ML model flags as high probability churn risks.")
    
    st.dataframe(
        high_risk_df[['CustomerId', 'Surname', 'Churn_Risk_Score', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'Exited_Label']].style.format({'Churn_Risk_Score': '{:.2%}'}),
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# TAB 4: RAW DATA EXPLORER
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Complete Customer Dataset")
    st.dataframe(filtered_df, use_container_width=True)