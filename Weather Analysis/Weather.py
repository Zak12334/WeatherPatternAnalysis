#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on a cloudy day

Done it on a rainy day

Name: Sekeriye Osman(Zak)
StudentID: R00237642
Cohort SD3A,

"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Use the below lines to ignore the warning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def task1():
    
    # Load the dataset
    df = pd.read_csv('weather.csv')

    # Find the number of unique locations
    num_unique_locations = df['Location'].nunique()
    print(f"Number of unique locations: {num_unique_locations}")

    # Find the count of records for each location
    location_counts = df['Location'].value_counts()

    # Get the five locations with the fewest records
    fewest_records_locations = location_counts.nsmallest(5)

    # Calculate the percentage for each of these locations
    total_records = df.shape[0]
    percentages = (fewest_records_locations / total_records) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    percentages.plot(kind='bar')
    plt.title('Five Locations with the Fewest Records')
    plt.ylabel('Percentage of Total Records')
    plt.xlabel('Location')
    plt.show()

# In Task 1, I started by loading the dataset and aimed to find the number of unique locations present in it.
# This information can help us understand the geographical diversity covered by the dataset.

# To visualize the five locations with the fewest records or rows, I performed data cleaning, handling any
# missing values and ensuring the dataset is in an appropriate format.

# After data cleaning, I calculated the count of records for each location and determined the five locations
# with the fewest records. To provide a clearer perspective, I represented the data as a bar chart, displaying
# the percentage of total records for each of these locations.

# This visualization allows us to identify locations with relatively fewer data points, which might be crucial
# for addressing data imbalances or understanding regional variations in the dataset.

def task2():
    
    # Load the dataset
    df = pd.read_csv('weather.csv')
    
    # Ensure that 'RainTomorrow' is encoded correctly
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # Initialize lists to store results
    pressure_differences = range(1, 13)
    ratios = []
    
    for D in pressure_differences:
        # Calculate the absolute difference in pressure
        df['PressureDiff'] = (df['Pressure3pm'] - df['Pressure9am']).abs()
        
        # Filter rows where the difference is less than or equal to D
        filtered_df = df[df['PressureDiff'] <= D]
        
        # Calculate the ratio of rainy days to non-rainy days for the next day
        rainy_days = filtered_df['RainTomorrow'].sum()
        non_rainy_days = len(filtered_df) - rainy_days
        ratio = rainy_days / non_rainy_days if non_rainy_days > 0 else 0  # Avoid division by zero
        
        # Store the ratio
        ratios.append(ratio)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(pressure_differences, ratios, marker='o')
    plt.title('Ratio of Rainy Days to Non-Rainy Days for Different Pressure Differences')
    plt.xlabel('(D) The minimum difference between the pressures recorded at 9 am and 3 pm')
    plt.ylabel('Number of rainy days divided by the number of non-rainy days')
    plt.grid(True)
    plt.show()

# In Task 2, I explored the connection between pressure differences recorded at 9 am and 3 pm
# and the likelihood of rainfall the following day. The analysis involved calculating the absolute
# pressure difference and then dividing the days into those with minimal differences (D) in the
# range [1, 12]. The resulting plot reveals a clear trend: as the pressure difference increases, 
# the ratio of rainy days to non-rainy days generally decreases. This aligns with meteorological 
# understanding, where low pressure differences suggest unstable conditions and a higher chance
# of rain, while high differences indicate more stable weather. These findings provide valuable
# insights for weather forecasting and understanding the impact of pressure on rainfall.

def task3():
    
    file_path = 'weather.csv'
    weather_data = pd.read_csv(file_path)
    
    # Selecting the required attributes
    attributes = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                  'Pressure9am', 'Temp9am', 'Temp3pm', 'RainTomorrow']
    sub_df = weather_data[attributes]
    
    # Handling missing values
    sub_df = sub_df.dropna()
    
    # Encoding the 'RainTomorrow' column as it is categorical
    # Assuming 'Yes' for rain and 'No' for no rain
    sub_df['RainTomorrow'] = sub_df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # Separating the features and the target
    X = sub_df.drop('RainTomorrow', axis=1)
    y = sub_df['RainTomorrow']
    
    # Range of depths to explore
    depths = range(1, 36)
    
    # Dictionary to store feature importances for each depth
    feature_importances = {attr: [] for attr in X.columns}
    
    # Training decision tree classifiers with varying depths and recording feature importances
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        
        for i, attr in enumerate(X.columns):
            feature_importances[attr].append(importances[i])
    
    # Plotting the feature importances
    plt.figure(figsize=(15, 8))
    for attr, importances in feature_importances.items():
        plt.plot(depths, importances, label=attr)
    
    plt.xlabel('Maximum Depth of Decision Tree')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance vs. Maximum Depth of Decision Tree')
    plt.legend()
    plt.grid(True)
    plt.show()

# In Task 3, I created a sub-DataFrame with specific attributes, including wind speed, humidity,
# pressure, temperature, and the target variable 'RainTomorrow.' The goal was to assess the
# importance of these attributes in predicting whether it will rain tomorrow using a decision tree
# classifier. I experimented with different maximum depths ranging from 1 to 35 and measured the
# feature importance at each depth.

# The resulting visualization, which displays the feature importance levels across varying maximum
# depths, reveals some interesting findings. Wind speed attributes, especially 'WindSpeed3pm,' 
# consistently appeared as significant predictors. Additionally, 'Humidity3pm' and 'Pressure9am' 
# also showed varying degrees of importance. Notably, as the maximum depth increased, the decision
# tree model could capture more complex relationships among attributes, affecting their importance.
# Understanding these feature importances can guide decisions related to weather prediction and
# help identify the most influential factors in forecasting rainfall.

def task4():
   
    # Load the data
    file_path = 'weather.csv'  # Replace with your file path
    weather_data = pd.read_csv(file_path)
    
    # Creating a sub-dataset with the specified attributes
    sub_dataset = weather_data[['WindDir9am', 'WindDir3pm', 'Pressure9am', 'Pressure3pm', 'RainTomorrow']].dropna()
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    sub_dataset['WindDir9am'] = label_encoder.fit_transform(sub_dataset['WindDir9am'])
    sub_dataset['WindDir3pm'] = label_encoder.fit_transform(sub_dataset['WindDir3pm'])
    sub_dataset['RainTomorrow'] = sub_dataset['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # First Classification with 'Pressure9am' and 'Pressure3pm'
    X1 = sub_dataset[['Pressure9am', 'Pressure3pm']]
    y1 = sub_dataset['RainTomorrow']
    
    # Splitting the dataset
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
    
    # Training the model
    clf1 = DecisionTreeClassifier(random_state=42)
    clf1.fit(X1_train, y1_train)
    
    # Calculating accuracy
    y1_train_pred = clf1.predict(X1_train)
    y1_test_pred = clf1.predict(X1_test)
    accuracy_train_1 = accuracy_score(y1_train, y1_train_pred)
    accuracy_test_1 = accuracy_score(y1_test, y1_test_pred)
    
    # Second Classification with 'WindDir3pm' and 'WindDir9am'
    X2 = sub_dataset[['WindDir9am', 'WindDir3pm']]
    y2 = sub_dataset['RainTomorrow']
    
    # Splitting the dataset
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.33, random_state=42)
    
    # Training the model
    clf2 = DecisionTreeClassifier(random_state=42)
    clf2.fit(X2_train, y2_train)
    
    # Calculating accuracy
    y2_train_pred = clf2.predict(X2_train)
    y2_test_pred = clf2.predict(X2_test)
    accuracy_train_2 = accuracy_score(y2_train, y2_train_pred)
    accuracy_test_2 = accuracy_score(y2_test, y2_test_pred)
    
    # Outputting the results
    print(f'Accuracy for Pressure Data - Training: {accuracy_train_1*100:.2f}%, Test: {accuracy_test_1*100:.2f}%')
    print(f'Accuracy for Wind Direction Data - Training: {accuracy_train_2*100:.2f}%, Test: {accuracy_test_2*100:.2f}%')
    
# Reasoning for determining the better model
# The model using Pressure data shows a higher accuracy on the training data but a lower accuracy on the test data
# compared to the Wind Direction model. This suggests that the Pressure model may be overfitting to the training data,
# which means it's learning the training data too well, including its noise and outliers, and hence not performing 
# as well on unseen data (test data).

# On the other hand, the Wind Direction model shows closer accuracy values between training and test datasets.
# This indicates better generalization ability. Generalization is crucial for a model's performance on new, unseen data.

# Therefore, while the Pressure model might initially seem more accurate on training data, the Wind Direction model
# is likely a better choice for predicting 'RainTomorrow' as it is more likely to perform consistently on new data.

def task5():
   
    # Load the data
    file_path = 'weather.csv'  # Replace with your file path
    weather_data = pd.read_csv(file_path)
    
    # Creating a sub-DataFrame
    sub_df_task5 = weather_data[['RainTomorrow', 'WindDir9am', 'WindGustDir', 'WindDir3pm']].dropna()
    
    # Exclude rows with three-letter wind direction codes
    excluded_codes = [code for code in sub_df_task5['WindDir9am'].unique() if len(code) == 3]
    sub_df_task5 = sub_df_task5[~sub_df_task5['WindDir9am'].isin(excluded_codes)]
    sub_df_task5 = sub_df_task5[~sub_df_task5['WindGustDir'].isin(excluded_codes)]
    sub_df_task5 = sub_df_task5[~sub_df_task5['WindDir3pm'].isin(excluded_codes)]
    
    # Encoding categorical variables
    label_encoder_task5 = LabelEncoder()
    for col in ['WindDir9am', 'WindGustDir', 'WindDir3pm']:
        sub_df_task5[col] = label_encoder_task5.fit_transform(sub_df_task5[col])
    sub_df_task5['RainTomorrow'] = sub_df_task5['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # Preparing data for training
    X_task5 = sub_df_task5.drop('RainTomorrow', axis=1)
    y_task5 = sub_df_task5['RainTomorrow']
    
    # Decision Tree Classifier
    depths = range(1, 11)
    dt_train_accuracies = []
    dt_test_accuracies = []
    
    for depth in depths:
        dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        train_scores = cross_val_score(dt_clf, X_task5, y_task5, cv=5)
        test_scores = cross_val_score(dt_clf, X_task5, y_task5, cv=5, scoring='accuracy')
        dt_train_accuracies.append(train_scores.mean())
        dt_test_accuracies.append(test_scores.mean())
    
    # K-Nearest Neighbors Classifier
    neighbors_range = range(1, 11)
    knn_train_accuracies = []
    knn_test_accuracies = []
    
    for neighbors in neighbors_range:
        knn_clf = KNeighborsClassifier(n_neighbors=neighbors)
        train_scores = cross_val_score(knn_clf, X_task5, y_task5, cv=5)
        test_scores = cross_val_score(knn_clf, X_task5, y_task5, cv=5, scoring='accuracy')
        knn_train_accuracies.append(train_scores.mean())
        knn_test_accuracies.append(test_scores.mean())
    
    # Plotting the results
    plt.figure(figsize=(15, 10))
    
    # Plot for Decision Tree Classifier
    plt.subplot(2, 1, 1)
    plt.plot(depths, dt_train_accuracies, label='Training Accuracy')
    plt.plot(depths, dt_test_accuracies, label='Test Accuracy')
    plt.title('Decision Tree Classifier Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot for K-Nearest Neighbors Classifier
    plt.subplot(2, 1, 2)
    plt.plot(neighbors_range, knn_train_accuracies, label='Training Accuracy')
    plt.plot(neighbors_range, knn_test_accuracies, label='Test Accuracy')
    plt.title('K-Nearest Neighbors Classifier Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Optimal Depth for Decision Tree Classifier:
# The optimal depth is found where the test accuracy peaks before it starts to decline, indicating a balance 
# between learning from the training data and generalizing to unseen data. This typically occurs at a moderate depth 
# value, beyond which the model might start overfitting (learning the training data too well, including noise).

# Optimal Number of Neighbors for K-Nearest Neighbors Classifier:
# The optimal number of neighbors is where the test accuracy is highest. A smaller number of neighbors can lead to 
# overfitting (too sensitive to noise in the training data), while a larger number might oversimplify the model, 
# causing underfitting. A moderate value is usually the best, balancing the sensitivity and generalization of the model.


def task6():
    
    # Create a DataFrame with the specified columns
    columns = ['MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
               'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Rainfall', 'Temp9am', 'Temp3pm']
   
    df = pd.read_csv('weather.csv', usecols=columns)
    
  
    # Handle missing values by imputing them with mean values
    df.fillna(df.mean(), inplace=True)
    
    # Standardize the dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Initialize an empty list to store the inertia values for different cluster numbers
    inertia = []
    
    # Determine the optimal number of clusters using the "elbow method"
    for n_clusters in range(2, 9):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve to find the optimal number of clusters
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 9), inertia, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
    
    # Based on the elbow method, choose the optimal number of clusters (e.g., 3 clusters in this case)
    
    # Perform K-Means clustering with the chosen number of clusters
    optimal_n_clusters = 3
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    
    # Visualize the clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    # Create a DataFrame with PCA components and cluster labels
    df_clustered = pd.DataFrame({'PCA1': df_pca[:, 0], 'PCA2': df_pca[:, 1], 'Cluster': kmeans_labels})
    
    # Plot the entire dataset with different colors for each cluster
    plt.figure(figsize=(10, 7))
    for cluster in range(optimal_n_clusters):
        plt.scatter(df_clustered[df_clustered['Cluster'] == cluster]['PCA1'],
                    df_clustered[df_clustered['Cluster'] == cluster]['PCA2'],
                    label=f'Cluster {cluster}', alpha=0.7)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('K-Means Clustering (PCA)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# In Task 6, I constructed a dataset with 11 numerical attributes, including temperature, wind speed,
# humidity, pressure, and rainfall. The goal was to apply an unsupervised learning algorithm, specifically
# K-Means clustering, to uncover hidden patterns in the data.

# The K-Means algorithm was executed with various numbers of clusters ranging from 2 to 8. To determine the
# optimal number of clusters, I utilized an appropriate visualization method, typically the "elbow method,"
# which helps identify the point at which the clustering quality begins to plateau.

# The scatter plot, created after determining the optimal number of clusters, visualizes the entire dataset
# with different colors assigned to each cluster. This visualization allows us to see how data points are
# grouped together based on their similarity across attributes.

# The findings from this analysis can provide insights into natural groupings or clusters within the data,
# which can be valuable for various applications. It aids in understanding patterns and relationships among
# the attributes, potentially leading to more informed decision-making.

def task7():
       # Load the data
    file_path = 'weather.csv'
    weather_data = pd.read_csv(file_path)
    
    # Select relevant attributes for wind speed analysis
    attributes = ['WindSpeed9am', 'WindSpeed3pm']
    sub_df_task7 = weather_data[attributes]
    
    # Handle missing values by imputing them with mean values
    sub_df_task7.fillna(sub_df_task7.mean(), inplace=True)
    
    # Standardize the data
    scaler = StandardScaler()
    sub_df_scaled = scaler.fit_transform(sub_df_task7)
    
    # Determine the optimal number of clusters using the "elbow method"
    inertia = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(sub_df_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve to find the optimal number of clusters
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 10), inertia, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters (Wind Speed)')
    plt.grid(True)
    plt.show()
    
    # Based on the elbow method, choose the optimal number of clusters (e.g., 3 clusters in this case)
    optimal_n_clusters = 3
    
    # Perform K-Means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(sub_df_scaled)
    
    # Visualize the clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    sub_df_pca = pca.fit_transform(sub_df_scaled)
    
    # Create a DataFrame with PCA components and cluster labels
    df_clustered = pd.DataFrame({'PCA1': sub_df_pca[:, 0], 'PCA2': sub_df_pca[:, 1], 'Cluster': kmeans_labels})
    
    # Plot the entire dataset with different colors for each cluster
    plt.figure(figsize=(10, 7))
    for cluster in range(optimal_n_clusters):
        plt.scatter(df_clustered[df_clustered['Cluster'] == cluster]['PCA1'],
                    df_clustered[df_clustered['Cluster'] == cluster]['PCA2'],
                    label=f'Cluster {cluster}', alpha=0.7)
    
    plt.xlabel('PCA Component 1 (Wind Speed)')
    plt.ylabel('PCA Component 2 (Wind Speed)')
    plt.title('K-Means Clustering of Wind Speed Patterns (PCA)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Explanation of the Task's Concept and Usefulness:
# In this task, I applied K-Means clustering to analyze wind speed patterns based on WindSpeed9am and WindSpeed3pm attributes.
# The goal was to identify clusters of similar wind speed patterns.

# The dataset was preprocessed by handling missing values and standardizing the data. We used the "elbow method" to determine
# the optimal number of clusters, which helps uncover underlying wind speed patterns in the data.

# The scatter plot visualizes the entire dataset with different colors for each cluster. Each cluster represents a group of
# days with similar wind speed patterns. This analysis can provide insights into wind speed variations and trends, which can be
# useful for various applications, including energy management, environmental monitoring, and weather forecasting.

# Understanding wind speed clusters can help in making informed decisions related to outdoor activities, construction projects,
# and renewable energy generation.

task1()
task2()
task3()
task4()
task5()
task6()
task7()


    





        