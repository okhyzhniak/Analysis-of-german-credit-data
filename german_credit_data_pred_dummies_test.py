import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.iolib.smpickle as smpickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import pickle

def rename_cols(data):
    data = data.rename({"0": "Status_checking_account", "1": "Duration_month"}, axis=1)

    data = data.rename({"2": "Credit_history", "3": "Credit_amount", "4": "Savings_account/bonds", \
        "5": "Present_employment", "6": "Job", "8": "Property", "9": "Age", \
            "10": "Other_debtors/guarantors", "24": "Target"}, axis=1)

    dummy_rename_dict = {str(val): "Dummy_" + str(val) for val in range(15, 24)}
    data = data.rename(dummy_rename_dict, axis=1)

    return data

def add_dummy_vars(data, cols_categorical_data):
    for col in cols_categorical_data:
        data_dummies = pd.get_dummies(data[col], prefix="Dummy_" + str.lower(col)) * 1
        data = pd.concat([data, data_dummies], axis=1)
    return data

data_test = pd.read_csv("german_credit_data/data/data_test.csv")
data_train = pd.read_csv("german_credit_data/data/data_train.csv")
data_val = pd.read_csv("german_Credit_data/data/data_val.csv")

print(data_train.shape)
print(data_val.shape)

cols_drop1 = ["7"]
cols_drop2 = [str(col) for col in range(11,15)]

data_train = data_train.drop(cols_drop1 + cols_drop2, axis=1)
data_val = data_val.drop(cols_drop1 + cols_drop2, axis=1)
data_test = data_test.drop(cols_drop1 + cols_drop2, axis=1)

data_val = rename_cols(data_val)
data_test = rename_cols(data_test)

data_train = pd.concat([data_train, data_val], axis=0)

# Change the units of measurement for the target variable
data_train["Target"] = data_train["Target"].replace({1: 0, 2: 1})
data_test["Target"] = data_test["Target"].replace({1: 0, 2: 1})

# Add dummy variables
cols_dummies = {}
cols_categorical = {}

for col in data_train.columns:
    if col.startswith("Dummy"):
        cols_dummies[col] = list(np.sort(data_train[col].unique()))
    elif (len(np.unique(data_train[col])) >= 2) & (len(np.unique(data_train[col])) <= 10):
        cols_categorical[col] = list(np.sort(data_train[col].unique()))

data_train = add_dummy_vars(data_train, cols_categorical)
data_test = add_dummy_vars(data_test, cols_categorical)

# Drop base dummy variables 
cols_drop_dummy_base = ["Dummy_status_checking_account_2", "Dummy_credit_history_0", \
    "Dummy_savings_account/bonds_1", "Dummy_job_2", "Dummy_present_employment_2", \
        "Dummy_property_1", "Dummy_other_debtors/guarantors_1"]

data_train = data_train.drop(cols_drop_dummy_base, axis=1)
data_test = data_test.drop(cols_drop_dummy_base, axis=1)

print("The shape of the train data \n", data_train.shape)
print("The shape of the test data \n", data_test.shape)

with open("german_credit_data/models/cols_baseline.pickle", "rb") as file:
    cols_baseline = pickle.load(file)

# Predict default using logistic regression
y, X = data_train["Target"], sm.add_constant(data_train[cols_baseline])
res_reg_logit = sm.Logit(y, X).fit()
print(res_reg_logit.summary())

pred_y_reg_logit_train = (res_reg_logit.predict(X) >= 0.5) * 1

print("Accuracy on the train set for logistic regression \n", \
    accuracy_score(y, pred_y_reg_logit_train))

smpickle.save_pickle(res_reg_logit, "german_credit_data/models/baseline_reg_logit.pickle")

# Evaluate model performance on the test data
y_test, X_test = data_test["Target"], sm.add_constant(data_test[cols_baseline])

pred_y_reg_logit_test = (res_reg_logit.predict(X_test) >= 0.5) * 1

print("Accuracy on the test set for logistic regression \n", \
    accuracy_score(y_test, pred_y_reg_logit_test))

