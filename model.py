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

# Extract features and target variable
X = data[['Temperature (°C)', 'Humidity (%)']]
y = data['Impact']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the classifiers
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
knn = KNeighborsClassifier()
svm = SVC()
lr = LogisticRegression()

# Train the classifiers
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Make predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
lr_pred = lr.predict(X_test)

# Initialize the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stacking Ensemble
stacking = StackingClassifier(estimators=[('rf', rf), ('gb', gb)], final_estimator=knn, cv=kf)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)

#stacking ensemble for svc and linear regression
stacking_svc_lr = StackingClassifier(estimators=[('svm', svm), ('lr', lr)], final_estimator=knn, cv=kf)
stacking_svc_lr.fit(X_train, y_train)
stacking_svc_lr_predict = stacking_svc_lr.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
svc_lr_accuracy = accuracy_score(y_test, stacking_svc_lr_predict)

print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("kNN Accuracy:", knn_accuracy)
print("Svm accuracy: ", svm_accuracy)
print("linear regression: ", lr_accuracy)
print("Stacking Ensemble Accuracy:", stacking_accuracy)
print("svc and lr stacking ensemble accuracy: ", svc_lr_accuracy)

# Calculate MSE
rf_mse = mean_squared_error(y_test, rf_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
svm_mse = mean_squared_error(y_test, svm_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
stacking_mse = mean_squared_error(y_test, stacking_pred)
svc_lr_mse = mean_squared_error(y_test, stacking_svc_lr_predict)

print("Random Forest MSE:", rf_mse)
print("Gradient Boosting MSE:", gb_mse)
print("kNN MSE:", knn_mse)
print("Stacking Ensemble MSE:", stacking_mse)
print("svm mse: ", svm_mse)
print("linear regression mse: ", lr_mse)
print("svc and lr stacking ensemble mse: ", svc_lr_mse)

print("Random Forest Report:", classification_report(y_test, rf_pred))
print("Gradient Boosting Report:", classification_report(y_test, gb_pred))
print("kNN Report:", classification_report(y_test, knn_pred))
print("SVM Report:", classification_report(y_test, svm_pred))
print("linear regression Report:", classification_report(y_test, lr_pred))
print("Stacking Ensemble Report:", classification_report(y_test, stacking_pred))
print("Stacking Ensemble(svc and lr) Report:", classification_report(y_test, stacking_svc_lr_predict))


print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Gradient Boosting Confusion Matrix:\n", confusion_matrix(y_test, gb_pred))
print("kNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("SVM confusion matrix:", confusion_matrix(y_test, svm_pred))
print("linear regression confusion matrix:", confusion_matrix(y_test, lr_pred))
print("Stacking Ensemble Confusion Matrix:\n", confusion_matrix(y_test, stacking_pred))
print("Stacking Ensemble svc and lr confusion matrix:\n", confusion_matrix(y_test, stacking_svc_lr_predict))