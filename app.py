# ===================== IMPORT LIBRARIES =====================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== LOAD DATA =====================

df = pd.read_csv("final_data.csv")

# ===================== LOAD MODEL =====================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ===================== REMOVE CUSTOMER ID =====================

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# ===================== TOTAL CHARGES =====================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ===================== ENCODING =====================

columns = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

label_encoders = {}

for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ===================== KPI VALUES =====================

total_customers = len(df)

active_customers = (df["Churn"] == "No").sum()

churn_customers = (df["Churn"] == "Yes").sum()

churn_rate = round(
    (churn_customers / total_customers) * 100,
    2
)

avg_monthly = round(
    df["MonthlyCharges"].mean(),
    2
)

avg_total = round(
    df["TotalCharges"].mean(),
    2
)

# part 2
# ===================== CUSTOM CSS =====================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

[data-testid="stAppViewContainer"]{
background:#0B1020;
}

[data-testid="stSidebar"]{
background:#111827;
border-right:1px solid #232b45;
}

section[data-testid="stSidebar"] *{
color:white !important;
}

.block-container{
padding-top:20px;
padding-left:35px;
padding-right:35px;
}

.title{
font-size:40px;
font-weight:700;
color:white;
}

.sub{
font-size:15px;
color:#9CA3AF;
margin-bottom:25px;
}

.card{

background:#161B2E;

padding:20px;

border-radius:18px;

border:1px solid #222A45;

box-shadow:0px 0px 18px rgba(0,255,255,.06);

transition:.3s;

}

.card:hover{

transform:translateY(-5px);

box-shadow:0px 0px 30px cyan;

}

.metric-title{

color:#9CA3AF;

font-size:15px;

}

.metric-value{

color:white;

font-size:34px;

font-weight:bold;

}

.metric-change{

color:#4ADE80;

font-size:14px;

margin-top:8px;

}

hr{

border:.5px solid #232B45;

}

</style>

""",unsafe_allow_html=True)

# ===================== SIDEBAR =====================

st.sidebar.image(
"https://img.icons8.com/color/96/artificial-intelligence.png",
width=70
)

st.sidebar.title("Customer Churn")

page = st.sidebar.radio(

"Navigation",

[

"🏠 Dashboard",

"🔮 Prediction",

"📈 Analytics",

"📋 Customers",

"⚙ Settings"

]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info(f"""

Total Customers : {total_customers}

Active Customers : {active_customers}

Churn Customers : {churn_customers}

""")

# ===================== HEADER =====================

st.markdown(

'<div class="title">Customer Churn Dashboard</div>',

unsafe_allow_html=True

)

st.markdown(

'<div class="sub">AI Powered Customer Retention Analytics</div>',

unsafe_allow_html=True

)

# ===================== KPI CARDS =====================

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown(f"""

<div class="card">

<div class="metric-title">

👥 Total Customers

</div>

<div class="metric-value">

{total_customers}

</div>

<div class="metric-change">

▲ 3.2%

</div>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown(f"""

<div class="card">

<div class="metric-title">

❌ Churn Customers

</div>

<div class="metric-value">

{churn_customers}

</div>

<div class="metric-change">

▼ 1.4%

</div>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown(f"""

<div class="card">

<div class="metric-title">

✅ Active Customers

</div>

<div class="metric-value">

{active_customers}

</div>

<div class="metric-change">

▲ 5.1%

</div>

</div>

""",unsafe_allow_html=True)

with c4:

    st.markdown(f"""

<div class="card">

<div class="metric-title">

📊 Churn Rate

</div>

<div class="metric-value">

{churn_rate} %

</div>

<div class="metric-change">

▲ Updated

</div>

</div>

""",unsafe_allow_html=True)

st.write("")
st.write("")

# part 3
# ===================== CHART SECTION =====================

col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📈 Churn Rate Over Time")

    trend = pd.DataFrame({
        "Month":["Jan","Feb","Mar","Apr","May","Jun","Jul"],
        "Rate":[10,13,22,18,24,33,28]
    })

    fig = px.line(
        trend,
        x="Month",
        y="Rate",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141B2E",
        plot_bgcolor="#141B2E",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

with col2:

    st.subheader("📊 Customer Distribution")

    pie = df["Churn"].value_counts()

    fig = px.pie(
        values=pie.values,
        names=pie.index,
        hole=.65,
        color_discrete_sequence=[
            "#6C5CE7",
            "#FF4D6D"
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141B2E",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

# ===================== SECOND ROW =====================

c1,c2=st.columns(2)

with c1:

    st.subheader("📊 Contract Type")

    contract=df["Contract"].value_counts()

    fig=px.bar(
        x=contract.index,
        y=contract.values,
        color=contract.values,
        color_continuous_scale="plasma"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141B2E",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

with c2:

    st.subheader("🌐 Internet Service")

    internet=df["InternetService"].value_counts()

    fig=px.pie(
        values=internet.values,
        names=internet.index,
        hole=.55,
        color_discrete_sequence=[
            "#5B8FF9",
            "#8B5CF6",
            "#00E396"
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141B2E",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

# ===================== TABLE =====================

st.subheader("🔥 High Risk Customers")

table=df.copy()

table["Risk Level"]=np.where(
    table["MonthlyCharges"]>80,
    "High",
    "Low"
)

st.dataframe(
    table.head(10),
    use_container_width=True,
    height=300
)

# part 4
# ===================== PREDICTION PANEL =====================

st.markdown("---")

st.subheader("🔮 Customer Churn Prediction")

left,right=st.columns([1,2])

with left:

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior=st.selectbox(
        "Senior Citizen",
        ["No","Yes"]
    )

    partner=st.selectbox(
        "Partner",
        ["No","Yes"]
    )

    dependents=st.selectbox(
        "Dependents",
        ["No","Yes"]
    )

    tenure=st.number_input(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone=st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    internet=st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract=st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment=st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly=st.number_input(
        "Monthly Charges",
        18.0,
        120.0,
        70.0
    )

    total=st.number_input(
        "Total Charges",
        18.0,
        9000.0,
        850.0
    )

    predict=st.button("🚀 Predict Churn")

with right:

    st.markdown("## Prediction Result")

    if predict:

        st.success("Prediction Completed Successfully")

        input_data=pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[1 if senior=="Yes" else 0],

        "Partner":[partner],

        "Dependents":[dependents],

        "tenure":[tenure],

        "PhoneService":[phone],

        "MultipleLines":["No"],

        "InternetService":[internet],

        "OnlineSecurity":["No"],

        "OnlineBackup":["No"],

        "DeviceProtection":["No"],

        "TechSupport":["No"],

        "StreamingTV":["No"],

        "StreamingMovies":["No"],

        "Contract":[contract],

        "PaperlessBilling":["Yes"],

        "PaymentMethod":[payment],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total]

        })

        for col in columns:

            input_data[col]=label_encoders[col].transform(input_data[col])

        prediction=model.predict(input_data)[0]

        probability=model.predict_proba(input_data)[0]

        if prediction==1:

            st.error("❌ Customer is Likely to Churn")

            st.metric(

                "Churn Probability",

                f"{probability[1]*100:.2f}%"

            )

            st.warning("""

High Risk Customer

Recommendation

✔ Offer Discount

✔ Contact Customer

✔ Upgrade Plan

""")

        else:

            st.success("✅ Customer Will Stay")

            st.metric(

                "Retention Probability",

                f"{probability[0]*100:.2f}%"

            )

            st.info("""

Low Risk Customer

Recommendation

✔ Keep Existing Plan

✔ Loyalty Rewards

✔ Continue Service

""")
            

# part 5
# ========================= RISK FACTORS =========================

st.markdown("---")

c1,c2=st.columns([1,1])

with c1:

    st.subheader("🔥 Top Churn Risk Factors")

    risk=pd.DataFrame({

        "Factor":[

        "Month-to-month Contract",

        "High Monthly Charges",

        "Low Tenure",

        "No Tech Support",

        "Fiber Optic"

        ],

        "Risk":[

        82,

        74,

        69,

        55,

        46

        ]

    })

    for i,row in risk.iterrows():

        st.write(row["Factor"])

        st.progress(row["Risk"]/100)

        st.write(f"{row['Risk']} %")

        st.write("")

with c2:

    st.subheader("📈 Monthly Charges Distribution")

    fig=px.histogram(

        df,

        x="MonthlyCharges",

        nbins=25,

        color_discrete_sequence=["#7C3AED"]

    )

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="#141B2E",

        plot_bgcolor="#141B2E",

        height=400

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ========================= CUSTOMER HISTORY =========================

st.markdown("---")

st.subheader("📋 Customer History")

show=st.slider(

"Rows",

5,

30,

10

)

history=df.head(show)

st.dataframe(

history,

use_container_width=True,

height=350

)

# ========================= QUICK SUMMARY =========================

st.markdown("---")

col1,col2,col3=st.columns(3)

with col1:

    st.info(f"""

### Total Customers

{total_customers}

""")

with col2:

    st.success(f"""

### Active Customers

{active_customers}

""")

with col3:

    st.error(f"""

### Churn Customers

{churn_customers}

""")

# ========================= FOOTER =========================

st.markdown("---")

st.markdown(

"""

<center>

<h5 style='color:white;'>

Customer Churn Prediction Dashboard

</h5>

<p style='color:gray;'>

Developed using Streamlit • Plotly • Machine Learning

</p>

</center>

""",

unsafe_allow_html=True

)

# part 6

# ========================= GAUGE =========================

st.markdown("---")

st.subheader("🎯 Churn Risk Meter")

risk_value = churn_rate

fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=risk_value,

    number={"suffix":" %"},

    title={"text":"Overall Churn Risk"},

    gauge={

        "axis":{"range":[0,100]},

        "bar":{"color":"#8B5CF6"},

        "steps":[

            {"range":[0,30],"color":"#10B981"},

            {"range":[30,60],"color":"#F59E0B"},

            {"range":[60,100],"color":"#EF4444"}

        ]

    }

))

fig.update_layout(

    template="plotly_dark",

    paper_bgcolor="#141B2E",

    height=350

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ========================= TOP 10 CHURN =========================

st.markdown("---")

st.subheader("🚨 Top Customers By Monthly Charges")

top = df.sort_values(

    "MonthlyCharges",

    ascending=False

).head(10)

st.dataframe(

top,

    use_container_width=True,

    height=300

)

# ========================= DOWNLOAD =========================

st.markdown("---")

csv = df.to_csv(index=False)

st.download_button(

    "📥 Download Dataset",

    csv,

    "Customer_Churn.csv",

    "text/csv"

)

# ========================= SUCCESS =========================



st.success("Dashboard Loaded Successfully")

st.caption("Customer Churn Prediction Dashboard • Streamlit • Machine Learning")
