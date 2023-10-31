import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

# Feature Engineering
# Interaction feature
data['Temp x Humidity'] = data['Temperature (°C)'] * data['Humidity (%)']

# Polynomial feature
data['Temp^2'] = data['Temperature (°C)'] ** 2

# Binning feature
data['Temp_bin'] = pd.cut(data['Temperature (°C)'], bins=[data['Temperature (°C)'].min(), 20, 30, data['Temperature (°C)'].max()], labels=['Low', 'Medium', 'High'])

# Convert categorical features to numerical ones
data['Temp_bin'] = label_encoder.fit_transform(data['Temp_bin'])

# Extract features and target variable
X = data[['Temperature (°C)', 'Humidity (%)', 'Temp x Humidity', 'Temp^2', 'Temp_bin']]
y = data['Impact']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the parameter values that should be searched
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Instantiate the grid
rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)

# Fit the grid with data
rf_grid.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", rf_grid.best_params_)

# Use the best parameters to initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=rf_grid.best_params_['n_estimators'], 
                            max_depth=rf_grid.best_params_['max_depth'], 
                            min_samples_split=rf_grid.best_params_['min_samples_split'], 
                            min_samples_leaf=rf_grid.best_params_['min_samples_leaf'], 
                            random_state=42)

# Train the classifiers
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest MSE:", rf_mse)
print("Random Forest Report:", classification_report(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))