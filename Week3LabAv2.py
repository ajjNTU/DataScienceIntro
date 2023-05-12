import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, precision_recall_fscore_support, \
    confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset from the provided URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
data = pd.read_csv(url, compression='gzip', header=None)

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column contains the target variable

# Initialize the stratified shuffle split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=152)

# Perform the stratified split
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of classifiers and their names
classifiers = [
    (KNeighborsClassifier(n_neighbors=1), "1-Nearest Neighbors"),
    (KNeighborsClassifier(), "5-Nearest Neighbors"),
    (RandomForestClassifier(random_state=42), "Random Forest (default depth)"),
    (RandomForestClassifier(random_state=42, max_depth=10), "Random Forest (depth=10)"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree (default depth)"),
    (DecisionTreeClassifier(random_state=42, max_depth=10), "Decision Tree (depth=10)"),
    (LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression"),
    (GaussianNB(), "Gaussian Naive Bayes"),
    (MLPClassifier(random_state=42), "Multi-layer Perceptron")
]

# Create a DataFrame to store the results
results = pd.DataFrame(columns=["Classifier", "MCC", "Accuracy", "Precision", "Recall", "F1-score"])

# Train, predict and store the metrics for each classifier
for clf, name in classifiers:
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Store the results
    results = pd.concat([results, pd.DataFrame({"Classifier": [name],
                                                "MCC": [mcc],
                                                "Accuracy": [accuracy],
                                                "Precision": [precision],
                                                "Recall": [recall],
                                                "F1-score": [f1_score]})], ignore_index=True)

    # Plot confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # confusion matrix visualization
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.title(f"{clf}")
    plt.show()

# Assign weights to each metric
weights = {"MCC": 1, "Accuracy": 1, "Precision": 2, "Recall": 2, "F1-score": 2}

# Calculate the weighted average of the metrics for each classifier
results['Weighted Average'] = (results['MCC'] * weights['MCC'] +
                               results['Accuracy'] * weights['Accuracy'] +
                               results['Precision'] * weights['Precision'] +
                               results['Recall'] * weights['Recall'] +
                               results['F1-score'] * weights['F1-score']) / sum(weights.values())

# Sort the results based on the weighted average
results = results.sort_values(by='Weighted Average', ascending=False)

# Display the results
print(results.to_string())