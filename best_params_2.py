import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-Validation Strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

# Instantiate the grid search models with RandomizedSearchCV
grid_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid_rf, cv=cv, n_jobs=-1, n_iter=50)
grid_search_gb = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=param_grid_gb, cv=cv, n_jobs=-1, n_iter=50)
grid_search_knn = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=param_grid_knn, cv=cv, n_jobs=-1, n_iter=50)
grid_search_svm = RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid_svm, cv=cv, n_jobs=-1, n_iter=50)
grid_search_lr = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=param_grid_lr, cv=cv, n_jobs=-1, n_iter=50)

# Fit the grid search models to the training data
grid_search_rf.fit(X_train_scaled, y_train)
grid_search_gb.fit(X_train_scaled, y_train)
grid_search_knn.fit(X_train_scaled, y_train)
grid_search_svm.fit(X_train_scaled, y_train)
grid_search_lr.fit(X_train_scaled, y_train)

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

# Instantiate and train the stacking classifier
base_classifiers = [('rf', RandomForestClassifier(**best_params_rf)), ('gb', GradientBoostingClassifier(**best_params_gb)), ('knn', KNeighborsClassifier(**best_params_knn)), ('svm', SVC(**best_params_svm)), ('lr', LogisticRegression(**best_params_lr))]
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())
stacking_classifier.fit(X_train_scaled, y_train)

# Make predictions with the stacking classifier
y_pred_stacking = stacking_classifier.predict(X_test_scaled)

# Evaluate the stacking classifier
accuracy = accuracy_score(y_test, y_pred_stacking)
print("Accuracy of Stacking Classifier on Test Data: ", accuracy)
print(classification_report(y_test, y_pred_stacking))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stacking))
