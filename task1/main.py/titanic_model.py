import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv(r"C:\Users\91917\OneDrive\Desktop\codesoft intern\TITANIC SURVIVAL PREDICTION\Titanic-Dataset.csv")
print("Sample Data:\n", data.head())

# Data Cleaning
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Feature Selection and Encoding
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical to numeric
y = data['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 1. Survival Rate by Gender
sns.barplot(data=data, x='Sex', y='Survived')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

# 2. Survival Rate by Passenger Class
sns.barplot(data=data, x='Pclass', y='Survived')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# 3. Age Distribution for Survived vs Not Survived
plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution: Survived vs. Not Survived')
plt.show()

# 4. Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()
