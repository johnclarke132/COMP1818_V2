import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, \
    confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import numpy as np


def spam_SVM():
    # Load data from csv
    data = pd.read_csv("dataset.csv").astype("U")

    # Split data into text and ham/spam label
    X = data["email_text"].values
    y = data["label"].values

    # Randomly split data into test and training sets with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Converting text String to Integer value
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    # Apply SVC algorithm
    model = SVC(kernel="rbf", random_state=0, probability=True)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))  # Return accuracy score

    # Fine tune parameters (improve accuracy) using GridSearchCV function
    tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    tuned_model = GridSearchCV(svm.SVC(probability=True), tuned_parameters)
    tuned_model.fit(X_train, y_train)
    print(tuned_model.best_params_)  # Return best parameters from GridSearchCV
    print(tuned_model.score(X_test, y_test))  # Return accuracy score for tuned model

    # Make predictions
    y_pred = tuned_model.predict(X_test)
    y_pred_prob = tuned_model.predict_proba(X_test)
    spam_probs = y_pred_prob[:, 1]

    # Confusion matrix
    confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(tuned_model, X_test, y_test, cmap=plt.cm.Blues)
    plt.show()
    print(classification_report(y_test, y_pred))

    # Scores
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1_score: {f1}")
    print(f"Recall: {recall}")
    print(f"MCC: {mcc}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_SVM()
