import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./Linear-Regression/House_dataset.csv")

# select important features
df = df[["area","bedrooms","price"]]

# shuffle data So we get different data in both training & testing dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# check if null value is present
is_null = df.isnull().any().any()
print(is_null) # False

# EDA

# check if house size have linear relationship with house price
# plt.scatter(df["area"],df["price"])
# plt.xlabel("house size")
# plt.ylabel("house price")
# plt.show()

# correlation_coefficient = df["area"].corr(df["price"], method="pearson")
# print(correlation_coefficient) # 0.53 this means Moderate linear relationship.

# check how house price influences for no. of bedrooms
# plt.bar(df["bedrooms"],df["price"])
# plt.xlabel("bedrooms")
# plt.ylabel("house price")
# plt.show()

# feature scaling - standardization
df["area"] = (df["area"] - df["area"].mean()) / df["area"].std()
df["bedrooms"] = (df["bedrooms"] - df["bedrooms"].mean()) / df["bedrooms"].std()

# print(df["area"].mean())
# print(df["bedrooms"].mean()) 
# print(df["area"].std())
# print(df["bedrooms"].std())

# split data into training and testing
train_df = df.iloc[0:431,:]
test_df = df.iloc[431:,:]

# actual implementation of algorithm 



