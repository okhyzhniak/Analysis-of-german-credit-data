import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.iolib.smpickle as smpickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

def accuracy(y_actual, y_predicted):
    return sum(y_actual == y_predicted)/y_actual.shape[0]

def find_opt_hurdle(y_actual, pred_prob, grid):
    accuracy_calc = 0
    opt_hurdle = 0

    for hurdle in grid:
        y_pred = (pred_prob >= hurdle) * 1
        accuracy(y_actual, y_pred)

        if accuracy(y_actual, y_pred) > accuracy_calc:
            accuracy_calc = accuracy(y_actual, y_pred)
            opt_hurdle = hurdle
        
    return accuracy_calc, opt_hurdle

data_train = pd.read_csv("german_credit_data/data/data_train.csv")
data_val = pd.read_csv("german_credit_data/data/data_val.csv")

print("The shape of the train data \n", data_train.shape)
print("The shape of the validation data \n", data_val.shape)

cols_drop1 = [str(col) for col in range(6,9)]
cols_drop2 = [str(col) for col in range(10,15)]
data_train = data_train.drop(cols_drop1 + cols_drop2, axis=1)
data_val = data_val.drop(cols_drop1 + cols_drop2, axis=1)

print("The shape of the train data \n", data_train.shape)
print("The shape of the validation data \n", data_val.shape)

data_val = data_val.rename({"0": "Status_checking_account", "1": "Duration_month"}, axis=1)

data_val = data_val.rename({"2": "Credit_history", "3": "Credit_amount", "4": "Savings_account/bonds", \
    "5": "Present_employment", "9": "Age", "24": "Target"}, axis=1)

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

print("Target variable \n", cols_target)
print("Numerical columns \n", cols_numerical)
print("Categorical columns \n", cols_categorical)
print("Dummy variables \n", cols_dummies)

# Change the units of measurement for the target variable
data_train["Target"] = data_train["Target"].replace({1: 0, 2: 1})
data_val["Target"] = data_val["Target"].replace({1: 0, 2: 1})

print("The distribution of the target variable \n", data_train["Target"].value_counts())

# Apply mean encoding to the categorical variables
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

cols_baseline = list(cols_dummies.keys()) + list(cols_numerical.keys()) + \
    ["Status_checking_account_mean", "Credit_history_mean", \
        "Savings_account/bonds_mean", "Present_employment_mean"]

# Run a logistic regression model
y, X = data_train[list(cols_target.keys())[0]], sm.add_constant(data_train[cols_baseline])
res_reg_logit = sm.Logit(y, X).fit()
pred_prob_reg_logit = res_reg_logit.predict(X)
print(res_reg_logit.summary())

pred_y_reg_logit = (pred_prob_reg_logit >= 0.5) * 1

# Run a decision tree and a random forest model
dtree = DecisionTreeClassifier(min_samples_leaf=20)
dtree_model = dtree.fit(X, y)

pred_y_dtree = dtree_model.predict(X)

rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, random_state=0)
rf_model = rf.fit(X, y)

pred_y_rf = rf_model.predict(X)

#with open("german_credit_data/models/cols_baseline.pickle", "wb") as file:
#    pickle.dump(cols_baseline, file)

#with open("german_credit_data/models/baseline_mean_encod_rf.pickle", "wb") as file:
#    pickle.dump(rf_model, file)

#with open("german_credit_data/models/baseline_mean_encod_dtree.pickle", "wb") as file:
#    pickle.dump(dtree_model, file)

print("The accuracy score for logistic regression on train dataset \n", \
    accuracy_score(y, pred_y_reg_logit))
print("The accuracy score for decision tree on train dataset \n", \
    accuracy_score(y, pred_y_dtree))
print("The accuracy score for random forest on train dataset \n", \
    accuracy_score(y, pred_y_rf))

#smpickle.save_pickle(res_reg_logit, "german_credit_data/models/baseline_mean_encod_reg_logit.pickle")

# Add mean encoded variables to the validation dataset
# Status of the checking account
data_val = pd.merge(data_val, default_by_checking_account, how="left", \
    left_on="Status_checking_account", right_index=True, suffixes=["_final", "_grouped"])

data_val = data_val.rename({"Target_final": "Target", "Target_grouped": \
    "Status_checking_account_mean"}, axis=1)

# Credit history
data_val = pd.merge(data_val, default_by_credit_history, how="left", \
    left_on="Credit_history", right_index=True, suffixes=["_final", "_grouped"])

data_val = data_val.rename({"Target_final": "Target", "Target_grouped": \
    "Credit_history_mean"}, axis=1)

# Savings account/bonds
data_val = pd.merge(data_val, default_by_savings_bonds, how="left", \
    left_on="Savings_account/bonds", right_index=True, suffixes=["_final", "_grouped"])

data_val = data_val.rename({"Target_final": "Target", "Target_grouped": \
    "Savings_account/bonds_mean"}, axis=1)

# Present employment
data_val = pd.merge(data_val, default_by_present_employment, how="left", \
    left_on="Present_employment", right_index=True, suffixes=["_final", "_grouped"])

data_val = data_val.rename({"Target_final": "Target", "Target_grouped": \
    "Present_employment_mean"}, axis=1)

# Perform model validation
y_val, X_val = data_val[list(cols_target.keys())[0]], sm.add_constant(data_val[cols_baseline])
pred_prob_reg_logit_val = res_reg_logit.predict(X_val)
pred_y_reg_logit_val = (pred_prob_reg_logit_val >= 0.5) * 1

pred_y_dtree_val = dtree_model.predict(X_val)

pred_y_rf_val = rf_model.predict(X_val)

print("Accuracy on the validation dataset for logistic regression \n", \
    accuracy_score(y_val, pred_y_reg_logit_val))
print("Accuracy on the validation dataset for decision tree \n", \
    accuracy_score(y_val, pred_y_dtree_val))
print("Accuracy on the validation dataset for random forest \n", \
    accuracy_score(y_val, pred_y_rf_val))

