# Hero FinCorp Data Analysis Script

# This script covers multiple analyses including default risk, profitability, behavior, etc.

# Author: ChatGPT for user Raghavan


# --- Load Data ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fpdf import FPDF
import os

# Load datasets
customers = pd.read_csv("customers.csv")
loans = pd.read_csv("loans.csv")
applications = pd.read_csv("applications.csv")
transactions = pd.read_csv("transactions.csv")
defaults = pd.read_csv("defaults.csv")
branches = pd.read_csv("branches.csv")


# --- Credit Score Distribution ---

sns.histplot(customers['Credit_Score'].dropna(), kde=True, bins=30)
plt.title("Distribution of Credit Scores")

# --- Loan Amount Distribution ---

sns.histplot(loans['Loan_Amount'].dropna(), kde=True, bins=30)
plt.title("Distribution of Loan Amounts")

# --- Monthly Loan Application Trends ---

applications['Application_Date'] = pd.to_datetime(applications['Application_Date'], errors='coerce')
monthly_apps = applications['Application_Date'].dt.to_period("M").value_counts().sort_index()
monthly_apps.plot(kind='line')

# --- Default Risk Heatmap ---

default_merged = defaults.merge(loans, on=["Loan_ID", "Customer_ID"], how="left")
default_merged = default_merged.merge(customers, on="Customer_ID", how="left")
correlation_data = default_merged[["Credit_Score", "Loan_Amount", "Interest_Rate", "Default_Amount", "Annual_Income"]].dropna()
sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm")

# --- Branch Performance: Top by Disbursement & Default Rate ---

branches['Default_Rate'] = branches['Delinquent_Loans'] / branches['Total_Active_Loans']
top_branches_by_disbursement = branches.sort_values('Loan_Disbursement_Amount', ascending=False).head(10)

# --- Customer Segmentation ---

customers['Credit_Score_Segment'] = pd.cut(customers['Credit_Score'], bins=[300,600,700,850], labels=['Low','Medium','High'])
customers['Income_Segment'] = pd.qcut(customers['Annual_Income'], q=3, labels=['Low','Medium','High'])

# --- Profitability: Interest Income ---

loans['Interest_Income'] = (loans['Loan_Amount'] * loans['Interest_Rate'] * loans['Loan_Term']) / (12 * 100)

# --- Recovery Rate by Legal Action ---

defaults['Recovery_Rate'] = defaults['Recovery_Amount'] / defaults['Default_Amount']
defaults.groupby('Legal_Action')['Recovery_Rate'].mean().plot(kind='bar')

# --- EMI vs Default Risk ---

loan_defaults = loans.merge(defaults[['Loan_ID']], on='Loan_ID', how='left', indicator=True)
loan_defaults['Default_Flag'] = np.where(loan_defaults['_merge'] == 'both', 1, 0)
loan_defaults['EMI_Bin'] = pd.qcut(loan_defaults['EMI_Amount'], q=5)

# --- Loan Application Insights ---

fee_comparison = applications.groupby("Approval_Status")["Processing_Fee"].mean().reset_index()

# --- Time to Default ---

time_to_default = defaults.merge(loans[['Loan_ID', 'Disbursal_Date']], on='Loan_ID', how='left')
time_to_default['Time_to_Default'] = (pd.to_datetime(time_to_default['Default_Date']) - pd.to_datetime(time_to_default['Disbursal_Date'])).dt.days

# --- Risk Matrix ---

risk_data['Risk_Score'] = 0.3*risk_data['norm_Loan_Amount'] + 0.2*risk_data['norm_Interest_Rate'] + 0.2*risk_data['norm_Loan_Term'] + 0.3*risk_data['norm_Credit_Score']

# --- Customer Behavior Classification ---

customer_behavior['Repayment_Behavior'] = customer_behavior.apply(lambda row: 'Always On Time' if row['Total_Defaults']==0 else ('Frequent Defaulter' if row['Total_Defaults']==row['Total_Loans'] else 'Occasional Defaulter'), axis=1)

# --- Loan Disbursement Efficiency ---

applications['Processing_Time'] = (applications['Approval_Date'] - applications['Application_Date']).dt.days

# --- Transaction Pattern Analysis ---

tx_pattern['Penalty_Ratio'] = tx_pattern['Total_Penalty'] / (tx_pattern['Total_EMI'] + tx_pattern['Total_Penalty'])

# --- Branch Efficiency (Rejection by Region fallback idea) ---

# Could not use Loan_ID for rejection by region since IDs didn't match, fallback idea: rejection by Source_Channel