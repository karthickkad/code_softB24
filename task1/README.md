# 🚢 Titanic Survival Prediction

This project uses machine learning to predict whether a passenger on the Titanic survived, based on personal attributes like age, sex, ticket class, and fare. It is built using Python, pandas, scikit-learn, and visualization libraries such as seaborn and matplotlib.

---

## 📁 Dataset

- Dataset file: `Titanic-Dataset.csv`
- Common features:
  - `Pclass` (Passenger class)
  - `Sex`
  - `Age`
  - `SibSp` (Number of siblings/spouses aboard)
  - `Parch` (Number of parents/children aboard)
  - `Fare`
  - `Embarked` (Port of Embarkation)
- Target variable: `Survived` (0 = No, 1 = Yes)

---

## 🔧 Features

- Data preprocessing:
  - Handling missing values
  - Encoding categorical variables
- Model:
  - Random Forest Classifier
- Evaluation:
  - Accuracy score calculation
- Visualizations:
  - Survival rates by gender
  - Survival rates by passenger class
  - Age distribution of survivors vs non-survivors
  - Feature importance analysis

---

## 📊 Output

- Model accuracy printed in the terminal
- Interactive graphs showing survival patterns and feature impact

---

## 🛠 Installation

1. Make sure Python 3.x is installed.
2. Install required packages:

```bash
pip install -r requirements.txt
