from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline


import gzip
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


# Load data file
with gzip.open("data/taiwan-bankruptcy-data.json.gz","r") as read_file:
    taiwan_data = json.load(read_file)
    
print(type(taiwan_data))




# Extract the key names from <code>taiwan_data</code>


taiwan_data_keys = taiwan_data.keys()
print(taiwan_data_keys)


# Calculate how many companies are in `taiwan_data`

n_companies = len(taiwan_data["observations"])
print(n_companies)


# Calculate the number of features associated with each company and assign the result to `n_features`.

n_features = len(taiwan_data["observations"][0])
print(n_features)

taiwan_data["observations"][0]


# Create a `wrangle` function that takes as input the path of a compressed JSON file and returns the file's contents as a DataFrame. 

def wrangle(filename):
    with gzip.open(filename, "r") as f:
        data = json.load(f)
    df = pd.DataFrame().from_dict(data["observations"]).set_index("id")
    return df


df = wrangle("data/taiwan-bankruptcy-data.json.gz")
print("df shape:", df.shape)
df.head()



df.info()

nans_by_col = ...
print("nans_by_col shape:", nans_by_col.shape)
nans_by_col.head()


# Plot class balance
df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    xlabel="Bankrupt",
    ylabel="Frequency",
    title="Class Balance"
);



target = "bankrupt"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_train.shape)
print("y_test shape:", y_train.shape)


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()



clf = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state=42)
)
clf.fit(X_train_over, y_train_over)



cv_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
print(cv_scores)


# Create a dictionary <code>params</code> with the range of hyperparameters 

params = {
    "simpleimputer__strategy": ["mean", "median"],
    "randomforestclassifier__n_estimators": range(25, 100, 25),
    "randomforestclassifier__max_depth": range(10,50,10)
}
params


# Create a <code>GridSearchCV</code> named `model` that includes your classifier and hyperparameter grid.

model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model


model.fit(X_train_over, y_train_over)


# Extract the cross-validation results from your model, and load them into a DataFrame named <code>cv_results</code>.


cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(5)


# Extract the best hyperparameters from your model and assign them to <code>best_params</code>. 


best_params = model.best_params_
print(best_params)


# Test the quality of model by calculating accuracy scores for the training and test data.


acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Model Training Accuracy:", round(acc_train, 4))
print("Model Test Accuracy:", round(acc_test, 4))


# Plot a confusion matrix


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);



from sklearn.metrics import classification_report
class_report = classification_report(y_test, model.predict(X_test))
print(class_report)


# Create a horizontal bar chart with the 10 most important features for model.


features = X_train_over.columns
importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");
# Don't delete the code below 👇
plt.savefig("images/5-5-17.png", dpi=150)


# Save model
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)


# Import your module
from my_predictor_assignment import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()