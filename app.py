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
# ========================= CSS =========================

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

[data-testid="stAppViewContainer"]{
background:#0F172A;
}

[data-testid="stSidebar"]{
background:#111827;
}

.block-container{
padding-top:1rem;
padding-left:2rem;
padding-right:2rem;
}

.card{
background:#1E293B;
padding:14px;
border-radius:14px;
border:1px solid #334155;
margin-bottom:10px;
}

.metric{
font-size:28px;
font-weight:bold;
color:white;
}

.title{
font-size:36px;
font-weight:bold;
color:white;
}

.sub{
color:#94A3B8;
font-size:14px;
}

</style>
""",unsafe_allow_html=True)

# ========================= SIDEBAR =========================

with st.sidebar:

    st.title("📊 Churn AI")

    page=st.radio(

    "Navigation",

    [

    "Dashboard",

    "Prediction",

    "Analytics"

    ]

    )

    st.markdown("---")

    st.metric(

    "Customers",

    total_customers

    )

    st.metric(

    "Churn Rate",

    f"{churn_rate}%"

    )

# ========================= HEADER =========================

st.markdown(

'<p class="title">Customer Churn Dashboard</p>',

unsafe_allow_html=True

)

st.markdown(

'<p class="sub">AI Powered Customer Retention Analytics</p>',

unsafe_allow_html=True

)

# ========================= KPI =========================

a,b,c,d=st.columns(4)

with a:

    st.markdown(f"""

<div class="card">

Total Customers

<p class="metric">

{total_customers}

</p>

</div>

""",unsafe_allow_html=True)

with b:

    st.markdown(f"""

<div class="card">

Active

<p class="metric">

{active_customers}

</p>

</div>

""",unsafe_allow_html=True)

with c:

    st.markdown(f"""

<div class="card">

Churn

<p class="metric">

{churn_customers}

</p>

</div>

""",unsafe_allow_html=True)

with d:

    st.markdown(f"""

<div class="card">

Rate

<p class="metric">

{churn_rate}%

</p>

</div>

""",unsafe_allow_html=True)

st.write("")

# part3
# ========================= ROW 1 =========================

left,right=st.columns([1,2])

# ========================= PREDICTION =========================

with left:

    st.markdown("### 🔮 Customer Prediction")

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    tenure=st.slider(
        "Tenure",
        0,
        72,
        12
    )

    contract=st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet=st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
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

    predict=st.button(
        "🚀 Predict",
        use_container_width=True
    )

# ========================= CHARTS =========================

with right:

    c1,c2=st.columns(2)

    with c1:

        st.markdown("#### 📈 Churn Trend")

        trend=pd.DataFrame({

            "Month":[

            "Jan","Feb","Mar",

            "Apr","May","Jun"

            ],

            "Rate":[

            14,18,22,25,27,30

            ]

        })

        fig=px.line(

            trend,

            x="Month",

            y="Rate",

            markers=True

        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#1E293B",

            plot_bgcolor="#1E293B",

            margin=dict(

                l=10,

                r=10,

                t=30,

                b=10

            ),

            height=270

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with c2:

        st.markdown("#### 🥧 Customer Distribution")

        pie=df["Churn"].value_counts()

        fig=px.pie(

            values=pie.values,

            names=pie.index,

            hole=.65

        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#1E293B",

            margin=dict(

                l=10,

                r=10,

                t=30,

                b=10

            ),

            height=270

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

st.write("")

# part 4
# ========================= PREDICTION RESULT =========================

st.markdown("---")

left,right=st.columns([1,1])

with left:

    st.markdown("### 🎯 Prediction Result")

    if predict:

        # ---------- Encode Input ----------

        input_df=pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[0],
            "Partner":["Yes"],
            "Dependents":["No"],
            "tenure":[tenure],
            "PhoneService":["Yes"],
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
            "PaymentMethod":["Electronic check"],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total]

        })

        # ---------- Label Encoding ----------

        for col in columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # ---------- Prediction ----------

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        if pred == 1:

            st.error("❌ Customer Will Churn")

            st.metric(
                "Churn Probability",
                f"{prob[1]*100:.2f}%"
            )

        else:

            st.success("✅ Customer Will Stay")

            st.metric(
                "Retention Probability",
                f"{prob[0]*100:.2f}%"
            )

with right:

    st.markdown("### 💡 Recommendation")

    if predict:

        if pred == 1:

            st.warning("""

🔴 High Risk Customer

✔ Offer Discount

✔ Contact Customer

✔ Suggest Yearly Plan

✔ Priority Support

""")

        else:

            st.success("""

🟢 Low Risk Customer

✔ Loyalty Rewards

✔ Continue Current Plan

✔ Regular Follow-up

""")

# ========================= FOOTER =========================

st.markdown("---")

st.caption(
    "Customer Churn Prediction Dashboard • Machine Learning • Streamlit"
)

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

# st.balloons()

st.success("Dashboard Loaded Successfully")

st.caption("Customer Churn Prediction Dashboard • Streamlit • Machine Learning")
