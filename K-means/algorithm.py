import pandas as pd
import numpy as np
import math

df = pd.read_csv("./k-means/Mall_Customers.csv")

# remove unwanted features
df = df.drop(["Gender", "CustomerID"], axis=1)

# removed rows of each columns whose values were 0
df = df[df["Age"] != 0]
df = df[df["Annual Income (k$)"] != 0]
df = df[df["Spending Score (1-100)"] != 0]

# df[df['Age'] != 0] — the inner part df['Age'] != 0 returns True/False for each row. Then df[...] keeps only the rows where it's True. So you're basically saying "give me only rows of df where Age is not zero

# Remove outlier using IQR

def is_integer(val):
    # because is_integer() always work with float value
    return float(val).is_integer()

# function to get Q1 and Q3
def get_Q_value(idx_position , column):

    # check if idx_position is complete index
    is_pure_idx = is_integer(idx_position)

    if is_pure_idx:
        # return value at that index as Q
        return column.iloc[int(idx_position)]
    else:
        # remove number after decimal and convert into int
        idx = int(idx_position)
        # return average of both values
        return (column.iloc[idx] + column.iloc[idx + 1]) / 2


# get acceptable range for column
def get_acceptable_range(column_name):
    
        # column data
        column = df[column_name]

        # sort values
        column = column.sort_values()

        # get index position which represent first 25% data
        idx_position_Q1 = 0.25 * (len(column) - 1)

        # get index position which represent first 75% data
        idx_position_Q3 = 0.75 * (len(column) - 1)

        # get Q values
        Q1 = get_Q_value(idx_position_Q1 , column)
        Q3 = get_Q_value(idx_position_Q3, column)

        # Get IQR
        IQR = Q3 - Q1

        lower_fence = Q1 - (1.5 * IQR)
        upper_fence = Q3 + (1.5 * IQR)

        # normal points range for that column
        acceptable_range = [lower_fence, upper_fence]
        # values outside this range are outliers

        return acceptable_range

range_col1 = get_acceptable_range("Age")
range_col2 = get_acceptable_range("Annual Income (k$)")
range_col3 = get_acceptable_range("Spending Score (1-100)")

# we should filter entire df
# That is remove entire rows which contains outlier

df = df[(df["Age"] >= range_col1[0]) & (df["Age"] <= range_col1[1])]
# This returns only rows of whole df[condition...] whose values are True (i.e  matches condition)

df = df[(df["Annual Income (k$)"] >= range_col2[0]) & (df["Annual Income (k$)"] <= range_col2[1])]

df = df[(df["Spending Score (1-100)"] >= range_col3[0]) & (df["Spending Score (1-100)"] <= range_col3[1])]

# Removed Outliers from df


# find optimal value of k using elbow method
# ...
# code to find optimal k using elbow method
#...

# Code for k-means

clusters = {}
k = 3

def k_means(k, clusters):
     
    # If its first iteration, then assign random points from dataset to k centroid
    # empty dictionary evaluates to false
    if not clusters:
         
        #  get random points from dataset & convert it into numpy array
        k_random_points = df.sample(n=k).to_numpy()

        # assign k random points to k clusters
        for i in range(k):
            clusters[f"cluster{i+1}"] = {"centroid" : k_random_points[i], "points":[]}
        
    else:
        # update centroid values
        for key in clusters:
            
            # convert into numpy array
            points = np.array(clusters[key]["points"])

            avg = np.mean(points, axis=0)
            # axis = 0 means operates columnwise. avg is single array like [0,0,0]

            # assign avg as centroid value
            clusters[key] = {"centroid" : avg, "points":clusters[key]["points"]}

    
    # create array of data points. empty array will be filled with data points
    data_points = []
    for i in range(k):
        data_points.append([])
    
    # if cluster are 2, then data points contains 2 arrays. 1st array is data points near to cluster1 & 2nd array is data points near to cluster2

    # euclideans for each points
    euclidean_distance_arr = []

    # iterate through all data points to calculate distance
    for i in range(len(df)):

        # for each point, this array will contains euclidean for k cluster
        euclidean = []
        for key in clusters:
            # calculate euclidean
            euclidean_distance = math.sqrt(sum((clusters[key]["centroid"] - df.iloc[i].to_numpy())**2))

            # add euclidean in euclidean array
            euclidean.append(euclidean_distance)
        
        # add euclidean of each k cluster for this point in array
        euclidean_distance_arr.append(euclidean)
    
    # find min distance and assign point as per
    for i in range(len(euclidean_distance_arr)):
        # find the index of min distance of clusters for current point, so we get that cluster
        min_distance_idx = np.argmin(np.array(euclidean_distance_arr[i]))

        # add current data point in clusters_data-point
        data_points[min_distance_idx].append(df.iloc[i].to_numpy())
    
    # stopping condition
    # check if old data points of k cluster are same as new data points of k cluster

    is_same = []
    for i in range(k):
        if np.array_equal(np.array(clusters[f"cluster{i+1}"]["points"]) , np.array(data_points[i])):
            is_same.append(True)
        else:
            is_same.append(False)
    
    # check if all cluster's data points remain unchanged
    if np.all(np.array(is_same)):
        # return cluster dictionary
        return clusters
    else:
        # assign points
        for i in range(k):
            clusters[f"cluster{i+1}"]["points"] = data_points[i]
        
        # after assugning points, again run k-means
        return k_means(k, clusters)

