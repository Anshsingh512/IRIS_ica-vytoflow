import pandas as pd
import numpy as np
import scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = pd.read_csv("IRIS.csv")

# Display the first few rows and basic statistics of the dataset
print(iris.head())
print(iris.describe())  

# Print unique target labels
print("Target Labels", iris["species"].unique())

# Visualize the dataset using Plotly Express
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Split the dataset into features (x) and target labels (y)
x = iris.drop("species", axis=1)
y = iris["species"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train a KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Take user input for new data point
user_input = input("Enter values for sepal length, sepal width, petal length, and petal width (comma-separated): ")
try:
    # Convert user input string into a list of float values
    x_new = [float(i) for i in user_input.split(',')]
    
    # Make prediction based on user input
    prediction = knn.predict([x_new])
    print("Prediction: {}".format(prediction))
except ValueError:
    print("Invalid input. Please enter comma-separated float values.")
