import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_table("german_credit_data/data/german_credit_data.data-numeric", sep="\s+", header=None)

data_train = data.sample(frac=0.8, random_state=0)
data_test = data.iloc[list(set(data.index) - set(data_train.index))]
#data_test.to_csv("german_credit_data/data/data_test.csv", index=False)

print("First rows of the credit dataset \n", data_train.head())
print("Total number of missing variables in the columns \n", data_train.isna().sum(axis=0).sum(axis=0))
print("The shape of the training data \n", data_train.shape)
print("The shape of the test data \n", data_test.shape)

data_train = data_train.rename({0: "Status_checking_account", 1: "Duration_month"}, axis=1)

data_train = data_train.rename({2: "Credit_history", 3: "Credit_amount", 4: "Savings_account/bonds", \
    5: "Present_employment", 9: "Age", 24: "Target"}, axis=1)

dummy_rename_dict = {val: "Dummy_" + str(val) for val in range(15, 24)}
data_train = data_train.rename(dummy_rename_dict, axis=1)

data_train_final = data_train.sample(n=650, random_state=0)
data_val = data.iloc[list(set(data_train.index) - set(data_train_final.index))]

print("The shape of the training dataset is \n", data_train_final.shape)
print("The shape of the validation dataset is \n", data_val.shape)

#data_train_final.to_csv("german_credit_data/data/data_train.csv", index=False)
#data_val.to_csv("german_credit_data/data/data_val.csv", index=False)

cols_categorical = {}
cols_dummy = {}
cols_numerical = {}
cols_target = {}

for col in data_train_final.columns:
    if col == "Target":
        cols_target[col] = list(np.sort(data_train_final[col].unique()))
    elif str(col).startswith("Dummy"):
        cols_dummy[col] = list(np.sort(data_train_final[col].unique()))
    elif (len(np.unique(data_train_final[col])) >= 2) & (len(np.unique(data_train_final[col])) <= 10):
        cols_categorical[col] = list(np.sort(data_train_final[col].unique()))
    else:
        cols_numerical[col] = len(np.unique(data_train_final[col]))

print("The list of columns which represent a target repayment variable \n", cols_target)
print("The list of dummy variable columns \n", cols_dummy)
print("The list of categorical columns \n", cols_categorical)
print("The list of numerical columns \n", cols_numerical)

print("The distrbution of the target variable by the status of the checking account \n", \
    pd.crosstab(data_train_final["Status_checking_account"], data_train_final["Target"], normalize=False))

plt.hist(data_train_final["Duration_month"][data_train_final["Target"] == 1], bins=50, label="Good loans")
plt.hist(data_train_final["Duration_month"][data_train_final["Target"] == 2], bins=50, label="Bad loans")
plt.title("The distribution of credit card loan duration by repayment type")
plt.legend()
plt.show()

print("The distrbution of the target variable by credit history \n", \
    pd.crosstab(data_train_final["Credit_history"], data_train_final["Target"], normalize=False))

plt.hist(data_train_final["Credit_amount"][data_train_final["Target"] == 1], bins=50, label="Good loans")
plt.hist(data_train_final["Credit_amount"][data_train["Target"] == 2], bins=50, label="Bad loans")
plt.title("The distribution of credit card loan amount by repayment type")
plt.legend()
plt.show()

print("The distribution of the target variable by the presence of savings accounts/bonds \n", \
    pd.crosstab(data_train_final["Savings_account/bonds"], data_train_final["Target"], normalize=False))

print("The distribution of the target variable by the duration of present employment \n", \
    pd.crosstab(data_train_final["Present_employment"], data_train_final["Target"], normalize=False))

print("The distribution of the target variable by the presence of other debtors/guarantors \n", \
    pd.crosstab(data_train_final[10], data_train_final["Target"], normalize=False))

plt.hist(data_train_final["Age"][data_train_final["Target"] == 1], bins=50, label="Good loans")
plt.hist(data_train_final["Age"][data_train_final["Target"] == 2], bins=50, label="Bad loans")
plt.title("The distribution of credit card debtor's age by repayment type")
plt.legend()
plt.show()

print("Pearson correlation coefficient \n", \
    stats.pearsonr(data_train_final["Age"], data_train_final["Target"]))


