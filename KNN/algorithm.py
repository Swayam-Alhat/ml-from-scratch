import pandas as pd
import numpy as np

df = pd.read_csv("./Logistic-Regression/diabetes-dataset.csv")

# remove unwanted features
df = df.drop(["Pregnancies","DiabetesPedigreeFunction","BMI"], axis=1)

# removed rows of each columns whose values were 0
df = df[df["Glucose"] != 0]
df = df[df["BloodPressure"] != 0]
df = df[df["SkinThickness"] != 0]

# df[df['Glucose'] != 0] — the inner part df['Glucose'] != 0 returns True/False for each row. Then df[...] keeps only the rows where it's True. So you're basically saying "give me only rows where Glucose is not zero.


training_data = df.iloc[:373,:]
testing_data = df.iloc[373:, :]

# feature scaling on training data (Standardization)

# save mean and std of training data.
# because we should not use mean and std of already scaled training data
mean_x1 = training_data["Glucose"].mean()
mean_x2 = training_data["BloodPressure"].mean()
mean_x3 = training_data["SkinThickness"].mean()
mean_x4 = training_data["Insulin"].mean()
mean_x5 = training_data["Age"].mean()

std_x1 = training_data["Glucose"].std()
std_x2 = training_data["BloodPressure"].std()
std_x3 = training_data["SkinThickness"].std()
std_x4 = training_data["Insulin"].std()
std_x5 = training_data["Age"].std()

# feature scaling on training data
training_data["Glucose"] = (training_data["Glucose"] - mean_x1) / std_x1
training_data["BloodPressure"] = (training_data["BloodPressure"] - mean_x2) / std_x2
training_data["SkinThickness"] = (training_data["SkinThickness"] - mean_x3) / std_x3
training_data["Insulin"] = (training_data["Insulin"] - mean_x4) / std_x4
training_data["Age"] = (training_data["Age"] - mean_x5) / std_x5

# As KNN does not have a training phase, we directly start prediction
# Before prediction, we should scale testing data using same mean and std of training data
testing_data["Glucose"] = (testing_data["Glucose"] - mean_x1) / std_x1
testing_data["BloodPressure"] = (testing_data["BloodPressure"] - mean_x2) / std_x2
testing_data["SkinThickness"] = (testing_data["SkinThickness"] - mean_x3) / std_x3
testing_data["Insulin"] = (testing_data["Insulin"] - mean_x4) / std_x4
testing_data["Age"] = (testing_data["Age"] - mean_x5) / std_x5

# prediction

k = 5

# function to calculate euclidean distance
def calculate_euclidean(test_data):
    # y represent feature values of training data
    # x represent feature values of testing data
    y1 = training_data["Glucose"].to_numpy()
    y2 = training_data["BloodPressure"].to_numpy()
    y3 = training_data["SkinThickness"].to_numpy()
    y4 = training_data["Insulin"].to_numpy()
    y5 = training_data["Age"].to_numpy()

    # distance calculation
    euclidean_distance = ((test_data[0] - y1)**2) + ((test_data[1] - y2)**2) + ((test_data[2] - y3)**2) + ((test_data[3] - y4)**2) + ((test_data[4] - y5)**2)

    # final euclidean distance
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance

# prediction function
def prediction(test_data):
    # get euclidean distance
    euclidean_distance = calculate_euclidean(test_data)

    indices = np.argsort(euclidean_distance)[:k]
    # np.argsort returns indices that would sort the array in ascending order, so [:k] directly gives you the k nearest neighbors' indices. Cleaner and correct


    # find those nearest data points and get their class label
    class_labels = []

    for idx in indices:
        data_point = training_data.iloc[idx,:].to_numpy()
        class_labels.append(data_point[-1])
    
    # Get unique values and their occurence count
    values, counts = np.unique(np.array(class_labels) , return_counts=True)

    # get index of frequent class
    frequent_class_idx = np.argmax(counts)

    # return most frequent value
    return values[frequent_class_idx]
    
# Prediction process

prediction_arr = []
actual_arr = []

for i in range(len(testing_data)):
    # convert sample into array
    test_data = testing_data.iloc[i,:].to_numpy()
    # perform prediction
    prediction_value = prediction(test_data)

    # append prediction
    prediction_arr.append(prediction_value)

    # append actual value
    actual_arr.append(test_data[-1])



print(np.array(prediction_arr) == np.array(actual_arr))
print(sum(np.array(prediction_arr) == np.array(actual_arr)))
print(len(actual_arr))


