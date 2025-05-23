import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\91917\OneDrive\Desktop\codesoft intern\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv", encoding='ISO-8859-1')

# Copy dataset
data = df.copy()

# Drop rows with missing ratings
data = data.dropna(subset=['Rating'])

# Clean 'Year': extract year from string
data['Year'] = data['Year'].astype(str).str.extract('(\d{4})')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Clean 'Duration': extract numeric minutes
data['Duration'] = data['Duration'].str.extract('(\d+)')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')

# Fill missing numeric values with median
data['Duration'] = data['Duration'].fillna(data['Duration'].median())
data['Year'] = data['Year'].fillna(data['Year'].median())

# Fill missing categorical values with 'Unknown'
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    data[col] = data[col].fillna('Unknown')

# Process Genre (multi-label binarization)
data['Genre'] = data['Genre'].fillna('Unknown')
data['Genre'] = data['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(data['Genre'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=data.index)

# Encode Director and Actors
label_encoders = {}
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Combine all features
features = pd.concat([
    data[['Year', 'Duration', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']],
    genre_df
], axis=1)

# Target variable
target = data['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features.columns

plt.figure(figsize=(12, 8))
plt.title("Top 20 Feature Importances - Movie Rating Prediction")
plt.bar(range(20), importances[indices][:20], align="center")
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.show()
