import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import pickle

def rename_cols(data):
    data = data.rename({"0": "Status_checking_account", "1": "Duration_month"}, axis=1)

    data = data.rename({"2": "Credit_history", "3": "Credit_amount", "4": "Savings_account/bonds", \
        "5": "Present_employment", "9": "Age", "24": "Target"}, axis=1)

    dummy_rename_dict = {str(val): "Dummy_" + str(val) for val in range(15, 24)}
    data = data.rename(dummy_rename_dict, axis=1)

    return data

data_test = pd.read_csv("german_credit_data/data/data_test.csv")
data_train = pd.read_csv("german_credit_data/data/data_train.csv")
data_val = pd.read_csv("german_Credit_data/data/data_val.csv")

cols_drop1 = [str(col) for col in range(6,9)]
cols_drop2 = [str(col) for col in range(10,15)]

data_train = data_train.drop(cols_drop1 + cols_drop2, axis=1)
data_val = data_val.drop(cols_drop1 + cols_drop2, axis=1)
data_test = data_test.drop(cols_drop1 + cols_drop2, axis=1)

data_val = rename_cols(data_val)
data_test = rename_cols(data_test)

data_train = pd.concat([data_train, data_val], axis=0)

# Change the units of measurement for the target variable
data_train["Target"] = data_train["Target"].replace({1: 0, 2: 1})
data_test["Target"] = data_test["Target"].replace({1: 0, 2: 1})

print("The shape of the train data \n", data_train.shape)
print("The shape of the test data \n", data_test.shape)

print("The columns in the train dataset \n", list(data_train.columns))
print("The columns in the test dataset \n", list(data_test.columns))

# Apply mean encoding to the categorical variables on the train set
default_by_checking_account = \
    data_train[["Status_checking_account", "Target"]].groupby("Status_checking_account")["Target"].mean()
default_by_credit_history = \
    data_train[["Credit_history", "Target"]].groupby("Credit_history")["Target"].mean()
default_by_savings_bonds = \
    data_train[["Savings_account/bonds", "Target"]].groupby("Savings_account/bonds")["Target"].mean()
default_by_present_employment = \
    data_train[["Present_employment", "Target"]].groupby("Present_employment")["Target"].mean()

# Status of the checking account
data_train = pd.merge(data_train, default_by_checking_account, how="left", \
    left_on="Status_checking_account", right_index=True, suffixes=["_final", "_grouped"])

data_train = data_train.rename({"Target_final": "Target", "Target_grouped": \
    "Status_checking_account_mean"}, axis=1)

# Credit history
data_train = pd.merge(data_train, default_by_credit_history, how="left", \
    left_on="Credit_history", right_index=True, suffixes=["_final", "_grouped"])

data_train = data_train.rename({"Target_final": "Target", "Target_grouped": \
    "Credit_history_mean"}, axis=1)

# Savings account/bonds
data_train = pd.merge(data_train, default_by_savings_bonds, how="left", \
    left_on="Savings_account/bonds", right_index=True, suffixes=["_final", "_grouped"])

data_train = data_train.rename({"Target_final": "Target", "Target_grouped": \
    "Savings_account/bonds_mean"}, axis=1)

# Present employment
data_train = pd.merge(data_train, default_by_present_employment, how="left", \
    left_on="Present_employment", right_index=True, suffixes=["_final", "_grouped"])

data_train = data_train.rename({"Target_final": "Target", "Target_grouped": \
    "Present_employment_mean"}, axis=1)

with open("german_credit_data/models/cols_baseline.pickle", "rb") as file:
    cols_baseline = pickle.load(file)

# Predict default using logistic regression
y, X = data_train["Target"], sm.add_constant(data_train[cols_baseline])
res_reg_logit = sm.Logit(y, X).fit()
print(res_reg_logit.summary())

pred_y_reg_logit_train = (res_reg_logit.predict(X) >= 0.5) * 1

print("Accuracy on the train set for logistic regression \n", \
    accuracy_score(y, pred_y_reg_logit_train))

# Apply mean encoding to the categorical variables on the test set
# Status of the checking account
data_test = pd.merge(data_test, default_by_checking_account, how="left", \
    left_on="Status_checking_account", right_index=True, suffixes=["_final", "_grouped"])

data_test = data_test.rename({"Target_final": "Target", "Target_grouped": \
    "Status_checking_account_mean"}, axis=1)

# Credit history
data_test = pd.merge(data_test, default_by_credit_history, how="left", \
    left_on="Credit_history", right_index=True, suffixes=["_final", "_grouped"])

data_test = data_test.rename({"Target_final": "Target", "Target_grouped": \
    "Credit_history_mean"}, axis=1)

# Savings account/bonds
data_test = pd.merge(data_test, default_by_savings_bonds, how="left", \
    left_on="Savings_account/bonds", right_index=True, suffixes=["_final", "_grouped"])

data_test = data_test.rename({"Target_final": "Target", "Target_grouped": \
    "Savings_account/bonds_mean"}, axis=1)

# Present employment
data_test = pd.merge(data_test, default_by_present_employment, how="left", \
    left_on="Present_employment", right_index=True, suffixes=["_final", "_grouped"])

data_test = data_test.rename({"Target_final": "Target", "Target_grouped": \
    "Present_employment_mean"}, axis=1)

# Evaluate model performance on the test data
y_test, X_test = data_test["Target"], sm.add_constant(data_test[cols_baseline])

pred_y_reg_logit_test = (res_reg_logit.predict(X_test) >= 0.5) * 1

print("Accuracy on the test set for logistic regression \n", \
    accuracy_score(y_test, pred_y_reg_logit_test))

