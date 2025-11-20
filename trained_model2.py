import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('Chronic_Kidney_Disease_data.csv')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop irrelevant columns
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ensure target variable is encoded to {0, 1}
df['classification'] = df['classification'].replace({2: 1})

# Separate features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Scale numerical features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance dataset using SMOTE
print("Before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train.value_counts())  # Check if balanced

# Hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100],  # Reduce trees
    'max_depth': [5, 10],  # Reduce depth to prevent overfitting
    'min_samples_split': [10, 20],  # Increase min samples split
    'min_samples_leaf': [5, 10],  # Increase min leaf
    'max_features': ['sqrt'],
    'bootstrap': [True],
    'class_weight': ['balanced']
}

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist,
                                   n_iter=10, cv=stratified_kfold, n_jobs=-1, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

# Best model
best_rf_model = random_search.best_estimator_

# Save model
pickle.dump(best_rf_model, open('model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Check training and test accuracy
y_train_pred = best_rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

y_test_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
