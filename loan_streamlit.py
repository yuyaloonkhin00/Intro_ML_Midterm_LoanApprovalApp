import streamlit as st
import pickle
import pandas as pd
import os


st.title("Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval.")

def load_model():
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

logo_path = "images/parami.jpg"   
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("**Student Name:** Yu Ya Loon Khin")
st.sidebar.markdown("**Student ID:** PU20230084")
st.sidebar.markdown("**Project Name:** Mid-term Project")
st.sidebar.markdown("**Course:** Introduction to Machine Learning")
st.sidebar.markdown("**Professor:** Daw Nwe Nwe Htay Win")



model = load_model()


st.header("Applicant Information")

education = st.selectbox("Education Level", ['Not Graduate', 'Graduate'])
self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
income_annum = st.number_input("Annual Income (in USD)", min_value=65000, value=200000)
loan_amount = st.number_input("Loan Amount Requested", min_value=200000, value=300000)
loan_term = st.number_input("Loan Term (in years)", min_value=2, value=5)
cibil_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=200000, value=5600000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=1300000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=300000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=2300000)


input_data = {
    'education': education,
    'self_employed': self_employed,
    'no_of_dependents': no_of_dependents,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}


if st.button("Predict Loan Status"):
    input_df=pd.DataFrame([input_data])
    prediction=model.predict(input_df)[0]
    label_names=['Approved','Rejected']  # same order as LabelEncoder
    result = label_names[prediction]

    if result == 'Approved':
        st.success("The loan is **Approved**!")
        st.image("images/approved.jpg", caption="Approved", width=300)
    else:
        st.error("The loan is **Rejected**.")
        st.image("images/rejected.jpg", caption="Rejected", width=300)
