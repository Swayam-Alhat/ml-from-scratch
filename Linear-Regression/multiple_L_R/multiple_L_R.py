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

learning_rate = 0.000001
m1 = 0
m2 = 0
b = 0
N = len(train_df)

epoch = 1000


for i in range(epoch):

    # predict y for each x
    # it returns new series object
    predicted_y = (m1 * train_df["area"]) + (m2 * train_df["bedrooms"]) + b

    # calculate error
    errors = train_df["price"] - predicted_y

    # calculate MSE
    MSE = (sum(errors ** 2)) / len(errors)

    # calculate gradient
    grad_m1 = (-2 / N) * (sum(errors * train_df["area"]))
    grad_m2 = (-2 / N) * (sum(errors * train_df["bedrooms"]))
    grad_b = (-2 / N) * sum(errors)

    # update m1, m2 and b
    m1 = m1 - learning_rate * grad_m1
    m2 = m2 - learning_rate * grad_m2
    b = b - learning_rate * grad_b

print(MSE)
