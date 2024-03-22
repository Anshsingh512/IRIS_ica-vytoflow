Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements. Now assume that you have the measurements of the iris flowers according to their species,
and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.


import pandas as pd
import numpy as np
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




So this is how you can train a machine learning model for the task of Iris classification using Python. 
    Iris Classification is one of the most popular case studies among the data science community. 


    this project is developed by ansh singh of vytoflow tech
    all copyright claim is to vytoflow . It comes under MIT license thus it is open for any changes and suggestion you can copy the code and run this in google colab


    .. Further improvements - 1. website of same using streamlin or flask can be made 
    2. tensor flow , open cv , etc. libraries can be used to further improve this project on image recognition basis . 


    
