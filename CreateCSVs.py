import csv
import os
import glob

def main():
    os.chdir('./figures')
    dirs = os.listdir()

    for dir in dirs:
        if os.path.isdir(dir):
            makeCSV(dir)

def makeCSV(dir):
    os.chdir(dir)
    files = glob.glob("*.txt")

    with open(dir + '_evaluations.csv', mode='w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)

        header = ['Algorithm', 'Accuracy', 'Mean Absolute Error', 'Root Mean Square Error']
        csvWriter.writerow(header)

        for txtFile in files:
            if txtFile == 'linear_regression_evaluation.txt':
                algorithmName = 'Linear Regression'
            elif txtFile == 'decision_tree_evaluation.txt':
                algorithmName = 'Decision Tree'
            elif txtFile == 'ridge_evaluation.txt':
                algorithmName = 'Ridge Regression'
            elif txtFile == 'stacking_evaluation.txt':
                algorithmName = 'Stacking'
            elif txtFile == 'random_forest_evaluation.txt':
                algorithmName = 'Random Forest'
            elif txtFile == 'lasso_evaluation.txt':
                algorithmName = 'Lasso Regression'
            elif txtFile == 'gradient_boosting_evaluation.txt':
                algorithmName = 'Gradient Boosting'
            elif txtFile == 'bagging_evaluation.txt':
                algorithmName = 'Bagging'
            else:
                algorithmName = txtFile

            lines = open(txtFile, mode='r').readlines()

            # Get the numbers from the txt file using this splitting expression
            accuracy = float(lines[1].split('\n')[0].split(':')[1].split(' ')[1])
            meanAbsoluteError = float(lines[2].split('\n')[0].split(':')[1].split(' ')[1])
            rootMeanSquaredError = float(lines[3].split('\n')[0].split(':')[1].split(' ')[1])

            row = [algorithmName, accuracy, meanAbsoluteError, rootMeanSquaredError]

            csvWriter.writerow(row)

    print('Successfully wrote CSV Evaluation for', dir)
    os.chdir('..')


if __name__ == '__main__':
    main()