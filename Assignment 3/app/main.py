import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def transform_input(scaler,Gender, Married, SelfEmployed, PropertyArea, Dependents, Education, CreditHistory, ApplicantIncome, CoapplicantIncome, LoanAmount, LoanAmountTerm):    
    column_list = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Gender_Male', 'Gender_nan', 'Credit_History_1.0', 
                   'Credit_History_nan', 'Married_Yes' ,'Dependents_1','Dependents_2','Dependents_3+',
                   'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']

    df = pd.DataFrame([[0]*len(column_list)], columns=column_list)

    if Gender == 'Male':
        df.loc[0,'Gender_Male'] = 1
    elif Gender == 'Other':
        df.loc[0,'Gender_nan'] = 1

    if Married == 'Yes':
        df.loc[0,'Married_Yes'] = 1

    if SelfEmployed == 'Yes':
        df.loc[0,'Self_Employed_Yes'] = 1

    if PropertyArea == 'Semiurban':
        df.loc[0,'Property_Area_Semiurban'] = 1
    elif PropertyArea == 'Urban':
        df.loc[0,'Property_Area_Urban'] = 1

    if Dependents == '1':
        df.loc[0,'Dependents_1'] = 1
    elif Dependents == '2':
        df.loc[0,'Dependents_2'] = 1
    elif Dependents == '3+':
        df.loc[0,'Dependents_3+'] = 1

    if Education == 'Not Graduate':
        df.loc[0,'Education_Not Graduate'] = 1

    if CreditHistory == 'Yes':
        df.loc[0,'Credit_History_1.0'] = 1
    elif CreditHistory == 'Does not know':
        df.loc[0,'Credit_History_nan'] = 1

    df.loc[0,'ApplicantIncome'] = ApplicantIncome

    df.loc[0,'CoapplicantIncome'] = CoapplicantIncome

    df.loc[0,'LoanAmount'] = LoanAmount

    df.loc[0,'Loan_Amount_Term'] = LoanAmountTerm

    standarization_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    temp = df[standarization_columns].copy() 
    temp.loc[:,standarization_columns] = scaler.transform(temp)
    df[standarization_columns] = temp

    return df


def load_scaler():
    scaler = joblib.load('model/scaler.pkl')
    return scaler

def load_model():
    model = joblib.load('model/loan_model_rfc_gs.dat')
    
    return model


def main():
    st.title('Loan Approval')
    st.markdown("---")
    st.text("This is an automatic loan approver according to the clients data.")

    Gender = st.selectbox(
    'What is your gender?',
    ('Male', 'Female', 'Other')
    )

    Married = st.selectbox(
    'Are you married?',
    ('Yes', 'No')
    )

    Dependents = st.selectbox(
    'How many dependents do you have?',
    ('0', '1', '2','3+')
    )
    Education = st.selectbox(
    'What is your level of education?',
    ('Graduate', 'Not Graduate')
    )

    SelfEmployed = st.selectbox(
    'Are you self employed?',
    ('Yes', 'No')
    )

    CreditHistory = st.selectbox(
    'Do you have all your debts paid?',
    ('Yes', 'No', 'Does not know')
    )

    PropertyArea = st.selectbox(
    'How is the region where you live?',
    ('Semiurban', 'Urban', 'Rural')
    )

    ApplicantIncome = st.number_input('What is you income?')
  

    CoapplicantIncome = st.number_input('What is the income of your coapplicant?')

    LoanAmount = st.number_input('What is the amount of the intended loan?')

    LoanAmountTerm = st.number_input('What is the term of that loan?')


    if st.button('Approve Loan'):
        df = transform_input(load_scaler(),Gender, Married, SelfEmployed, PropertyArea, Dependents, Education, CreditHistory, ApplicantIncome, CoapplicantIncome, LoanAmount, LoanAmountTerm)
        predict = load_model().predict(df)
        if predict == 1:
            st.write('Loan Approved')
        else:
            st.write('Loan Not Approved')

    st.sidebar.title("Useful Links")
    st.sidebar.markdown("---")
    st.sidebar.markdown("[Github]"
                        "(https://github.com/Rpinto02/)")
    st.sidebar.markdown("[Linkedin]"
                        "(https://www.linkedin.com/in/rpinto02/)")
    st.sidebar.markdown("[DPhi]"
                        "(https://DPhi.tech)")


if __name__ == '__main__':

    main()