import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.iolib.smpickle as smpickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import pickle

def add_dummy_vars(data, cols_categorical_data):
    for col in cols_categorical_data:
        data_dummies = pd.get_dummies(data[col], prefix="Dummy_" + str.lower(col)) * 1
        data = pd.concat([data, data_dummies], axis=1)
    return data

data_train = pd.read_csv("german_credit_data/data/data_train.csv")
data_val = pd.read_csv("german_credit_data/data/data_val.csv")

print("The shape of the train data \n", data_train.shape)
print("The shape of the validation data \n", data_val.shape)

cols_drop1 = ["7"]
cols_drop2 = [str(col) for col in range(11,15)]
data_train = data_train.drop(cols_drop1 + cols_drop2, axis=1)
data_val = data_val.drop(cols_drop1 + cols_drop2, axis=1)

print("The shape of the train data \n", data_train.shape)
print("The shape of the validation data \n", data_val.shape)

data_val = data_val.rename({"0": "Status_checking_account", "1": "Duration_month"}, axis=1)

data_val = data_val.rename({"2": "Credit_history", "3": "Credit_amount", "4": "Savings_account/bonds", \
    "5": "Present_employment", "6": "Job", "8": "Property", "9": "Age", "10": "Other_debtors/guarantors", \
        "24": "Target"}, axis=1)

dummy_rename_dict = {str(val): "Dummy_" + str(val) for val in range(15, 24)}
data_val = data_val.rename(dummy_rename_dict, axis=1)

print("Columns in the validation dataset \n", list(data_val.columns))
print("Columns in the train dataset are equal to the columns in the validation dataset \n", \
    sum(data_train.columns == data_val.columns))

cols_dummies = {}
cols_categorical = {}
cols_numerical = {}
cols_target = {}

for col in data_train.columns:
    if col == "Target":
        cols_target[col] = list(np.sort(data_train[col].unique()))
    elif col.startswith("Dummy"):
        cols_dummies[col] = list(np.sort(data_train[col].unique()))
    elif (len(np.unique(data_train[col])) >= 2) & (len(np.unique(data_train[col])) <= 10):
        cols_categorical[col] = list(np.sort(data_train[col].unique()))
    else:
        cols_numerical[col] = len(np.unique(data_train[col]))

del cols_numerical["Duration_month"]
del cols_numerical["Credit_amount"]

print("Target variable \n", cols_target)
print("Numerical columns \n", cols_numerical)
print("Categorical columns \n", cols_categorical)
print("Dummy variables \n", cols_dummies)

# Change the units of measurement for the target variable
data_train["Target"] = data_train["Target"].replace({1: 0, 2: 1})
data_val["Target"] = data_val["Target"].replace({1: 0, 2: 1})

# Add dummy variables
data_train = add_dummy_vars(data_train, cols_categorical)
data_val = add_dummy_vars(data_val, cols_categorical)

# Drop base dummy variables 
cols_drop_dummy_base = ["Dummy_status_checking_account_2", "Dummy_credit_history_0", \
    "Dummy_savings_account/bonds_1", "Dummy_present_employment_2", "Dummy_job_2", \
        "Dummy_property_1", "Dummy_other_debtors/guarantors_1"]

data_train = data_train.drop(cols_drop_dummy_base, axis=1)
data_val = data_val.drop(cols_drop_dummy_base, axis=1)

cols_dummies = {col: list(np.sort(data_train[col].unique())) for col in data_train.columns \
    if col.startswith("Dummy")}

cols_baseline = list(cols_dummies.keys()) + list(cols_numerical.keys())

# Look at multicollinearity
for (i, col) in enumerate(cols_baseline):
    print("VIF for column {0} \n".format(col), variance_inflation_factor(data_train[cols_baseline], i))

# Save seleted columns for future analysis
#with open("german_credit_data/models/cols_baseline.pickle", "wb") as file:
#    pickle.dump(cols_baseline, file)

# Run a logistic regression model
y, X = data_train[list(cols_target.keys())[0]], sm.add_constant(data_train[cols_baseline])
res_reg_logit = sm.Logit(y, X).fit()
pred_prob_reg_logit = res_reg_logit.predict(X)
print(res_reg_logit.summary())

pred_y_reg_logit = (pred_prob_reg_logit >= 0.5) * 1

# smpickle.save_pickle(res_reg_logit, "german_credit_data/models/baseline_reg_logit.pickle")

print("The accuracy score for logistic regression on train dataset \n", \
    accuracy_score(y, pred_y_reg_logit))

# Perform model validation
y_val, X_val = data_val[list(cols_target.keys())[0]], sm.add_constant(data_val[cols_baseline])
pred_prob_reg_logit_val = res_reg_logit.predict(X_val)
pred_y_reg_logit_val = (pred_prob_reg_logit_val >= 0.5) * 1

print("Accuracy on the validation dataset for logistic regression \n", \
    accuracy_score(y_val, pred_y_reg_logit_val))

