
# Hero FinCorp Case Study Analysis Script

# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load datasets
applications = pd.read_csv("applications.csv")
loans = pd.read_csv("loans.csv")
customers = pd.read_csv("customers.csv")
branches = pd.read_csv("branches.csv")
transactions = pd.read_csv("transactions.csv")
defaults = pd.read_csv("defaults.csv")

# Step 3: Merge applications and loans
merged = pd.merge(applications, loans, on="Loan_ID", how="left")
merged = merged.rename(columns={"Customer_ID_x": "Customer_ID"}).drop(columns=["Customer_ID_y"])

# Step 4: Merge with customer details
merged = pd.merge(merged, customers, on="Customer_ID", how="left")

# Step 5: Feature engineering
merged["Is_Approved"] = merged["Approval_Status"] == "Approved"
merged["Is_Default"] = merged["Loan_Status"] == "Overdue"
merged["Processing_Fee"] = pd.to_numeric(merged["Processing_Fee"], errors="coerce")
merged["Credit_Bucket"] = pd.cut(
    merged["Credit_Score"],
    bins=[0, 500, 650, 750, 900],
    labels=["High Risk", "Moderate", "Good", "Excellent"]
)

# Step 6: Add transaction summaries
txn_summary = transactions.groupby("Customer_ID")["Amount"].sum().reset_index()
txn_summary.columns = ["Customer_ID", "Total_Repaid"]
merged = pd.merge(merged, txn_summary, on="Customer_ID", how="left")

# Step 7: Add default status
merged = pd.merge(merged, defaults, on="Loan_ID", how="left")
merged["Legal_Action"] = merged["Legal_Action"].fillna("No")
merged["Has_Default"] = ~merged["Default_Amount"].isna()

# Step 8: Generate summaries
approval_by_region = merged.groupby("Region")["Is_Approved"].mean().reset_index(name="Approval_Rate")
default_by_region = merged.groupby("Region")["Has_Default"].mean().reset_index(name="Default_Rate")
credit_bucket_counts = merged["Credit_Bucket"].value_counts().reset_index()
credit_bucket_counts.columns = ["Credit_Segment", "Count"]

# Step 9: Export merged dataset
merged.to_csv("HeroFinCorp_Merged_Output.csv", index=False)
