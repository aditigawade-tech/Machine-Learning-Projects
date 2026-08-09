import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Analytics & Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom Styling (Background & UI Colors)
# -----------------------------
st.markdown("""
    <style>
    /* Main Background Color */
    .stApp {
        background-color: #f4f6f9;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Card/Box Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #2563eb;
    }
    /* Headers Styling */
    h1, h2, h3 {
        color: #0f172a;
    }
    /* Hide Default Menus */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Pickle Files
# -----------------------------
@st.cache_resource
def load_assets():
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        dv = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, dv, encoder, features

try:
    model, dv, encoder, features = load_assets()
except Exception as e:
    st.error("Failed to load model dependencies. Please check pickle files.")

# -----------------------------
# Sidebar Navigation & Filters (Slicers)
# -----------------------------
st.sidebar.title("🏠 Navigation & Slicers")

page = st.sidebar.radio("Go to Page", ["Prediction Tool", "Analytics Dashboard"])

st.sidebar.divider()
st.sidebar.subheader("🎯 Global Slicers")
selected_state_filter = st.sidebar.multiselect(
    "Filter State",
    ["Maharashtra", "Karnataka", "Gujarat", "Delhi", "Tamil Nadu"],
    default=["Maharashtra", "Karnataka"]
)

property_filter = st.sidebar.multiselect(
    "Property Type Slicer",
    ["Apartment", "Independent House", "Villa"],
    default=["Apartment", "Independent House", "Villa"]
)

# -----------------------------
# Page 1: Prediction Tool
# -----------------------------
if page == "Prediction Tool":
    st.title("🏡 House Price Prediction System")
    st.markdown("Configure property parameters below to forecast estimated valuations.")
    st.divider()

    # Form Columns Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location & Basic Info")
        state = st.selectbox("State", ["maharashtra", "karnataka", "gujarat", "delhi", "tamil nadu"])
        city = st.text_input("City", "Mumbai")
        property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        facing = st.selectbox("Facing", ["South", "East", "West", "North"])

    with col2:
        st.subheader("📐 Physical Dimensions")
        size_sqft = st.number_input("Size (Sq.Ft)", min_value=300, max_value=10000, value=1200)
        price_sqft = st.number_input("Price Per Sq.Ft (₹)", min_value=1000, value=5000)
        year = st.number_input("Year Built", min_value=1980, max_value=2025, value=2018)
        floor = st.number_input("Floor Number", min_value=0, value=2)
        total_floor = st.number_input("Total Floors", min_value=1, value=10)
        age = st.number_input("Age of Property (Years)", min_value=0, value=5)

    with col3:
        st.subheader("🛡️ Amenities & Specs")
        furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        school = st.number_input("Nearby Schools", min_value=0, value=5)
        hospital = st.number_input("Nearby Hospitals", min_value=0, value=3)
        transport = st.selectbox("Public Transport", ["Low", "Medium", "High"])
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["No", "Yes"])
        amenities = st.text_input("Amenities", "Gym, Pool")
        owner = st.text_input("Owner Type", "First Owner")
        availability = st.text_input("Availability Status", "Ready to Move")

    st.divider()

    # Action Row
    c_btn, c_res = st.columns([1, 2])

    with c_btn:
        st.write("")
        predict_trigger = st.button("🚀 Estimate Price", use_container_width=True)

    if predict_trigger:
        # Ordinal Encoding Process
        ordinal_df = pd.DataFrame({
            "Property_Type": [property_type],
            "Furnished_Status": [furnished],
            "Public_Transport_Accessibility": [transport],
            "Facing": [facing],
            "Security": [security]
        })

        ordinal_encoded = encoder.transform(ordinal_df)

        encoded_prop = ordinal_encoded[0][0]
        encoded_furn = ordinal_encoded[0][1]
        encoded_trans = ordinal_encoded[0][2]
        encoded_face = ordinal_encoded[0][3]
        encoded_sec = ordinal_encoded[0][4]

        input_data = {
            "State": state.lower(),
            "City": city.lower(),
            "Property_Type": encoded_prop,
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_per_SqFt": price_sqft,
            "Year_Built": year,
            "Furnished_Status": encoded_furn,
            "Floor_No": floor,
            "Total_Floors": total_floor,
            "Age_of_Property": age,
            "Nearby_Schools": school,
            "Nearby_Hospitals": hospital,
            "Public_Transport_Accessibility": encoded_trans,
            "Parking_Space": parking.lower(),
            "Security": encoded_sec,
            "Amenities": amenities.lower(),
            "Facing": encoded_face,
            "Owner_Type": owner.lower(),
            "Availability_Status": availability.lower()
        }

        try:
            X = dv.transform([input_data])
            prediction = model.predict(X)[0]

            with c_res:
                st.success("Target Price Evaluated")
                st.metric(
                    label="Estimated Market Value",
                    value=f"₹ {prediction:.2f} Lakhs"
                )
                st.balloons()

        except Exception as e:
            st.error(f"Prediction Failed: {e}")

    with st.expander("📋 Raw Technical Payload Details"):
        st.json(input_data if 'input_data' in locals() else {"status": "No inputs generated yet."})

# -----------------------------
# Page 2: Analytics Dashboard
# -----------------------------
elif page == "Analytics Dashboard":
    st.title("📊 Real Estate Market Analytics")
    st.markdown("Interactive analytical view of target metrics, property dimensions, and trends.")
    st.divider()

    # KPI Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Market Price", "₹ 78.5 Lakhs", "+4.2%")
    m2.metric("Average Rate / Sq.Ft", "₹ 6,200", "+1.1%")
    m3.metric("Selected States", len(selected_state_filter))
    m4.metric("Property Types Filtered", len(property_filter))

    st.divider()

    # Row 1 Graphs
    g1, g2 = st.columns(2)

    # Simulated Market Dataset for Graphical Visualization
    mock_df = pd.DataFrame({
        "Property_Type": ["Apartment", "Villa", "Independent House", "Apartment", "Villa", "Apartment"],
        "State": ["Maharashtra", "Karnataka", "Maharashtra", "Gujarat", "Delhi", "Tamil Nadu"],
        "Price_Lakhs": [65, 180, 120, 45, 210, 85],
        "Size_SqFt": [900, 2800, 1800, 750, 3200, 1100],
        "BHK": [2, 4, 3, 2, 5, 3]
    })

    # Apply Slicer Selection to Dashboard Visuals
    filtered_df = mock_df[
        (mock_df["State"].isin(selected_state_filter)) & 
        (mock_df["Property_Type"].isin(property_filter))
    ]

    with g1:
        st.subheader("Price Distribution by Property Type")
        if not filtered_df.empty:
            fig1 = px.bar(
                filtered_df, 
                x="Property_Type", 
                y="Price_Lakhs", 
                color="State", 
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("No data matches current slicer selections.")

    with g2:
        st.subheader("Size (Sq.Ft) vs Price Analysis")
        if not filtered_df.empty:
            fig2 = px.scatter(
                filtered_df, 
                x="Size_SqFt", 
                y="Price_Lakhs", 
                size="BHK", 
                color="Property_Type",
                hover_name="State",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No data matches current slicer selections.")

    # Row 2 Graphs
    g3, g4 = st.columns(2)

    with g3:
        st.subheader("Property Share by Category")
        if not filtered_df.empty:
            fig3 = px.pie(
                filtered_df, 
                names="Property_Type", 
                values="Price_Lakhs",
                hole=0.4
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No data matches current slicer selections.")

    with g4:
        st.subheader("Average Price Profile per State")
        if not filtered_df.empty:
            avg_state = filtered_df.groupby("State")["Price_Lakhs"].mean().reset_index()
            fig4 = px.line(
                avg_state, 
                x="State", 
                y="Price_Lakhs", 
                markers=True
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("No data matches current slicer selections.")

# -----------------------------
# Footer Section
# -----------------------------
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 14px;'>
        <strong>House Price Analytics Platform</strong> | Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)