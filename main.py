import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def spam_SVM():
    # Load data from csv
    data = pd.read_csv("spam.csv")

    # Split data into text and ham/spam label
    X = data["text"].values
    y = data["label_num"].values

    # Randomly split data into test and training sets with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Converting text String to Integer value
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    # Apply SVC algorithm
    model = SVC(kernel="rbf", random_state=0, probability=True)
    model.fit(X_train, y_train)
    # print(model.score(X_test, y_test))  # Return accuracy score

    # Fine tune parameters (improve accuracy) using GridSearchCV function
    tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    tuned_model = GridSearchCV(svm.SVC(probability=True), tuned_parameters)
    tuned_model.fit(X_train, y_train)
    # print(tuned_model.score(X_test, y_test))  # Return accuracy score for tuned model

    # Make predictions
    y_pred = tuned_model.predict(X_test)
    # y_pred_prob = tuned_model.predict_proba(X_test)
    # spam_probs = y_pred_prob[:, 1]

    # Scores
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1_score: {f1}")
    print(f"MCC: {mcc}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_SVM()
