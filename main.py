from download_dataset import download_dataset

import os

# handle and visualize data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# sklearn models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

def main():
    # Download latest version of crypto dataset
    dataset_path = download_dataset()

    dataset_list = os.listdir(dataset_path)

    print("Path to dataset files:", dataset_path)
    print(dataset_list)

    for csv in dataset_list:
        train_model(os.path.join(dataset_path, csv))

def train_model(data_path):
    data = pd.read_csv(data_path)

    data.info()
    data.head(10)

    data = data.drop(columns= ['ticker', 'date'], axis = 1)
    target = data['close']
    features = data.drop(columns= ['close'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=46)

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)

    # Evaluation
    print('Accuracy: ', linear.score(X_test, y_test) * 100)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Visualization
    a = X_test.open
    b = y_test
    c = X_test.open
    d = y_pred
    plt.figure(dpi=80)
    plt.scatter(a, b)
    plt.scatter(c, d)
    plt.legend(["Test", "Predicted"])
    plt.xlabel("open")
    plt.ylabel("close")
    plt.show()

if __name__ == '__main__':
    main()
