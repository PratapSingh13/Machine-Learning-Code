# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
df_train = pd.read_csv("train.csv", index_col="Id", na_values="NA")
train_size = len(df_train)

df_test = pd.read_csv("test.csv", index_col="Id", na_values="NA")
df = df_train.append(df_test)
df.drop("SalePrice", axis=1, inplace=True)


print("There are {} rows in train.csv".format(len(df_train)))
print("There are {} rows in test.csv".format(len(df_test)))


# helper method to obtain all numerical/categorical features of a data frame
def get_features(df, feature_type):
    if feature_type == "num":
        return list(df.select_dtypes(include=["float", "int"]).columns)
    elif feature_type == "cat":
        return list(df.select_dtypes(include=["object"]).columns)
    else:
        raise ValueError("feature_type must be 'num' (numerical) or 'cat' (categorical).")
        
        
num_features = get_features(df, "num")
columns_with_na_values = df.columns[df.isnull().any()]
num_features_without_na = [x for x in num_features if x not in columns_with_na_values]

print("Numerical features without missing values in train.csv and test.csv:")
print(num_features_without_na)

sns.regplot(x="GrLivArea", y="SalePrice", data=df_train);


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

d

