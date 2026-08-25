import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# Bank Customer Subscription Prediction
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Subscription Prediction",
    page_icon="🏦",
    layout="wide"
)

MODEL_FILE = "bank_subscription_model.keras"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "bank-full.csv"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

# Load model, scaler and original dataset
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_FILE, sep=";")

model, scaler = load_model_and_scaler()
bank_data = load_training_data()

# The training code for this project uses LabelEncoder (one number per category,
# 16 columns total) — NOT pd.get_dummies. We recreate the same encoders here,
# fit on the same raw data, so the codes match exactly what the model/scaler expect.
X_training = bank_data.drop("y", axis=1)

encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    X_training[col] = le.fit_transform(X_training[col])
    encoders[col] = le

training_columns = X_training.columns.tolist()  # 16 columns, matching the scaler

st.title("🏦 Bank Customer Subscription Prediction")
st.write(
    "Predict whether a bank customer is likely to subscribe "
    "to a term deposit using a trained Deep Learning model."
)

st.divider()

st.subheader("Customer Information")

# Use the categories directly from the original bank dataset
job_options = sorted(bank_data["job"].unique().tolist())
marital_options = sorted(bank_data["marital"].unique().tolist())
education_options = sorted(bank_data["education"].unique().tolist())
default_options = sorted(bank_data["default"].unique().tolist())
housing_options = sorted(bank_data["housing"].unique().tolist())
loan_options = sorted(bank_data["loan"].unique().tolist())
contact_options = sorted(bank_data["contact"].dropna().unique().tolist())
month_options = [
    m for m in
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
    if m in bank_data["month"].unique()
]
poutcome_options = sorted(bank_data["poutcome"].unique().tolist())

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=int(bank_data["age"].min()),
        max_value=int(bank_data["age"].max()),
        value=35
    )

    job = st.selectbox("Job", job_options)

    marital = st.selectbox("Marital Status", marital_options)

    education = st.selectbox("Education", education_options)

    default = st.selectbox("Credit Default", default_options)

    balance = st.number_input(
        "Account Balance",
        min_value=int(bank_data["balance"].min()),
        max_value=int(bank_data["balance"].max()),
        value=1000
    )

    housing = st.selectbox("Housing Loan", housing_options)

    loan = st.selectbox("Personal Loan", loan_options)

with col2:
    contact = st.selectbox("Contact", contact_options)

    day = st.number_input(
        "Last Contact Day",
        min_value=int(bank_data["day"].min()),
        max_value=int(bank_data["day"].max()),
        value=15
    )

    month = st.selectbox("Last Contact Month", month_options)

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=int(bank_data["duration"].min()),
        max_value=int(bank_data["duration"].max()),
        value=200
    )

    campaign = st.number_input(
        "Number of Contacts During Campaign",
        min_value=int(bank_data["campaign"].min()),
        max_value=int(bank_data["campaign"].max()),
        value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=int(bank_data["pdays"].min()),
        max_value=int(bank_data["pdays"].max()),
        value=-1
    )

    previous = st.number_input(
        "Previous Contacts",
        min_value=int(bank_data["previous"].min()),
        max_value=int(bank_data["previous"].max()),
        value=0
    )

    poutcome = st.selectbox("Previous Campaign Outcome", poutcome_options)

st.divider()

if st.button("🔮 Predict Subscription", type="primary", use_container_width=True):

    # Create one-row dataframe with exactly the same raw columns as training data
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])[training_columns]

    # Apply the SAME LabelEncoders used during training (not one-hot encoding)
    customer_encoded = customer.copy()
    for col, le in encoders.items():
        customer_encoded[col] = le.transform(customer_encoded[col])

    # Scale using the scaler fitted during training
    customer_scaled = scaler.transform(customer_encoded)

    # Deep Learning prediction
    probability = float(model.predict(customer_scaled, verbose=0)[0][0])

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.success("✅ Customer is likely to subscribe to a term deposit.")
    else:
        st.warning("❌ Customer is unlikely to subscribe to a term deposit.")

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(min(max(probability, 0.0), 1.0))

st.divider()

st.caption(
    "Model: TensorFlow / Keras Neural Network • "
    "Application: Streamlit • Task: Binary Classification"
)