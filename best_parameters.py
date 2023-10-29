import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np


# Load the dataset
data = pd.read_csv('three_hundred.csv')

# Preprocess the data
data = data.dropna()

# Convert temperature and humidity string ranges to numerical values
data['Temperature (°C)'] = data['Temperature (°C)'].apply(lambda x: np.mean([int(val) for val in x.split('-')]))
data['Humidity (%)'] = data['Humidity (%)'].apply(lambda x: np.mean([int(val) for val in x.split('-')]))

# Encode the 'Impact' column
label_encoder = LabelEncoder()
data['Impact'] = label_encoder.fit_transform(data['Impact'])

# Extract features and target variable
X = data[['Temperature (°C)', 'Humidity (%)']]
y = data['Impact']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the parameter values that should be searched
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_gb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'max_depth': [1, 2, 3, 4, 5],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

param_grid_knn = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance']
}

param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100, 1000]
}

param_grid_lr = {
    'C': [0.1, 1, 10, 100, 1000]
}

# Instantiate the grid search models
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, cv=3, n_jobs=-1)
grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid_gb, cv=3, n_jobs=-1)
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=3, n_jobs=-1)
grid_search_svm = GridSearchCV(estimator=SVC(), param_grid=param_grid_svm, cv=3, n_jobs=-1)
grid_search_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid_lr, cv=3, n_jobs=-1)

# Fit the grid search models to the training data
grid_search_rf.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)

# Get the best parameters for each model
best_params_rf = grid_search_rf.best_params_
best_params_gb = grid_search_gb.best_params_
best_params_knn = grid_search_knn.best_params_
best_params_svm = grid_search_svm.best_params_
best_params_lr = grid_search_lr.best_params_

# Print the best parameters for each model
print("Best parameters for RandomForestClassifier: ", best_params_rf)
print("Best parameters for GradientBoostingClassifier: ", best_params_gb)
print("Best parameters for KNeighborsClassifier: ", best_params_knn)
print("Best parameters for SVC: ", best_params_svm)
print("Best parameters for LogisticRegression: ", best_params_lr)

