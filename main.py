from download_dataset import download_dataset

# handle and visualize data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# sklearn models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

def main():
    # Download latest version of crypto dataset
    path = download_dataset()

    print("Path to dataset files:", path)

if __name__ == '__main__':
    main()
