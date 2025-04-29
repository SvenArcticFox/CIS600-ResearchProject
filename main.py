import os
import threading

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

# get dataset
import kagglehub

def download_dataset():
    return kagglehub.dataset_download("svaningelgem/crypto-currencies-daily-prices")

def main():
    # Download latest version of crypto dataset
    dataset_path = download_dataset()

    dataset_list = os.listdir(dataset_path)
    dataset_list.sort()

    print("Path to dataset files:", dataset_path)
    print("Files in dataset:", dataset_list)

    if not os.path.exists("./figures"):
        os.mkdir("figures")

    for csv in dataset_list:
        train_models(os.path.join(dataset_path, csv))

def train_models(data_path):
    data = pd.read_csv(data_path)

    # split the path to get the ticker name from the filename
    ticker_name = os.path.split(data_path)[-1].split('.')[0]
    print('Working on', ticker_name)

    figure_save_path = os.path.join("./figures", ticker_name)

    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    data.info()
    data.head(10)

    data = data.drop_duplicates()

    data = data.drop(columns= ['ticker', 'date'], axis = 1)
    target = data['close']
    features = data.drop(columns= ['close'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=46)

    def train_linear_regression():
        linear = LinearRegression(n_jobs=-1) # If you get an error, remove n_jobs parameter
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)

        # Evaluation
        evaluation_file = open(os.path.join(figure_save_path, "linear_regression_evaluation.txt"), "w")

        evaluation_file.write('Linear Regression Evaluation of ' + str(ticker_name) + '\n')
        evaluation_file.write('Accuracy: ' + str(linear.score(X_test, y_test) * 100) + '\n')
        evaluation_file.write('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        evaluation_file.write('Root Mean Squared Error: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                              + '\n')

        evaluation_file.close()

        # print('Linear Regression Evaluation of ', ticker_name)
        # print('Accuracy: ', linear.score(X_test, y_test) * 100)
        # print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
        # print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

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
        plt.title(ticker_name)
        plt.savefig(os.path.join(figure_save_path, "linear_regression.png"))
        plt.close()

    def train_lasso():
        lasso = Lasso(max_iter = 25000, alpha = 0.2)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)

        # Evaluation
        evaluation_file = open(os.path.join(figure_save_path, "lasso_evaluation.txt"), "w")

        evaluation_file.write('Lasso Evaluation of ' + str(ticker_name) + '\n')
        evaluation_file.write('Accuracy: ' + str(lasso.score(X_test, y_test) * 100) + '\n')
        evaluation_file.write('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        evaluation_file.write('Root Mean Squared Error: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                              + '\n')

        evaluation_file.close()

        # print('Lasso Evaluation of ', ticker_name)
        # print('Accuracy: ', lasso.score(X_test, y_test) * 100)
        # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



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
        plt.title(ticker_name)
        plt.savefig(os.path.join(figure_save_path, "lasso.png"))
        plt.close()

    def train_ridge():
        ridge = Ridge(max_iter=10, alpha=0.05)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        # Evaluation
        evaluation_file = open(os.path.join(figure_save_path, "ridge_evaluation.txt"), "w")

        evaluation_file.write('Ridge Evaluation of ' + str(ticker_name) + '\n')
        evaluation_file.write('Accuracy: ' + str(ridge.score(X_test, y_test) * 100) + '\n')
        evaluation_file.write('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        evaluation_file.write('Root Mean Squared Error: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                              + '\n')

        evaluation_file.close()

        # print('Ridge Evaluation of ', ticker_name)
        # print('Accuracy: ', ridge.score(X_test, y_test) * 100)
        # print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
        # print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

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
        plt.title(ticker_name)
        plt.savefig(os.path.join(figure_save_path, "ridge.png"))
        plt.close()

    def train_random_forest():
        rf = RandomForestRegressor(n_estimators=100, random_state=46, n_jobs=-1) # If you get an error,
                                                                                 # remove n_jobs parameter
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Evaluation
        evaluation_file = open(os.path.join(figure_save_path, "random_forest_evaluation.txt"), "w")

        evaluation_file.write('Random Forest Evaluation of ' + str(ticker_name) + '\n')
        evaluation_file.write('Accuracy: ' + str(rf.score(X_test, y_test) * 100) + '\n')
        evaluation_file.write('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        evaluation_file.write('Root Mean Squared Error: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                              + '\n')

        evaluation_file.close()

        # print('Random Forest Evaluation of ', ticker_name)
        # print('Accuracy: ', rf.score(X_test, y_test) * 100)
        # print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
        # print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

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
        plt.title(ticker_name)
        plt.savefig(os.path.join(figure_save_path, "random_forest.png"))
        plt.close()

    def train_stacking():
        # Base models
        ridge = Ridge(alpha=0.1)
        rf = RandomForestRegressor(n_estimators=100, random_state=46)
        # Final Estimator
        final_estimator = LinearRegression()

        stacking = StackingRegressor(
            estimators= [('ridge', ridge), ('rf', rf)],
            final_estimator= final_estimator,
            passthrough= True,
            n_jobs= -1
        ) # If you get an error, remove n_jobs parameter

        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)

        # Evaluation
        evaluation_file = open(os.path.join(figure_save_path, "stacking_evaluation.txt"), "w")

        evaluation_file.write('Stacking Evaluation of ' + str(ticker_name) + '\n')
        evaluation_file.write('Accuracy: ' + str(stacking.score(X_test, y_test) * 100) + '\n')
        evaluation_file.write('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        evaluation_file.write('Root Mean Squared Error: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                              + '\n')

        evaluation_file.close()

        # print('Stacking Evaluation of ', ticker_name)
        # print('Accuracy: ', stacking.score(X_test, y_test) * 100)
        # print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
        # print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

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
        plt.title(ticker_name)
        plt.savefig(os.path.join(figure_save_path, "stacking.png"))
        plt.close()

    # train_linear_regression()
    # print()
    # train_lasso()
    # print()
    # train_ridge()
    # print()
    # train_random_forest()
    # print()
    # train_stacking()
    # print()

    linear_thread = threading.Thread(train_linear_regression())
    lasso_thread = threading.Thread(train_lasso())
    ridge_thread = threading.Thread(train_ridge())
    rf_thread = threading.Thread(train_random_forest())
    stacking_thread = threading.Thread(train_stacking())

    linear_thread.start()
    lasso_thread.start()
    ridge_thread.start()
    rf_thread.start()
    stacking_thread.start()

    linear_thread.join()
    lasso_thread.join()
    ridge_thread.join()
    rf_thread.join()
    stacking_thread.join()
    print()


if __name__ == '__main__':
    main()
