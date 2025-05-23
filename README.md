# 🎬 Movie Rating Prediction with Python

This project predicts the IMDb rating of Indian movies based on features such as **genre**, **director**, **star cast**, **runtime**, **meta score**, and **gross earnings**. It uses regression modeling to analyze historical data and estimate future movie ratings.

---

## 📌 Project Goals

- Analyze historical movie data
- Engineer relevant features
- Build and evaluate regression models
- Understand the key drivers of movie ratings

---

## 📁 Dataset

- **Source**: IMDb India dataset
- **Format**: CSV
- **Columns**:
  - `Genre`: Movie genre (e.g., Action, Drama)
  - `Director`: Name of the movie's director
  - `Star`: Lead actor/actress
  - `Runtime`: Duration of the movie (in minutes)
  - `Meta_score`: Score from Metacritic
  - `Gross`: Box office gross earnings
  - `IMDB_Rating`: Target variable (movie rating)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 How to Run

1. **Clone the repository** or download the code.
2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script**:

    ```bash
    python movie_rating_prediction.py
    ```

---

## 🔍 Features

- Missing value handling
- Label encoding for categorical features
- Linear Regression model training
- Model evaluation using Mean Squared Error and R² Score
- Feature importance visualization

---

## 📊 Model Output

- **Mean Squared Error (MSE)**: How far predictions are from actual values.
- **R² Score**: How well the model explains the variability in ratings.
- **Feature Importance**: Bar chart showing which features matter most.

---

## 💡 Future Improvements

- Use advanced models like Random Forest, XGBoost, or Neural Networks
- Handle multi-genre encoding
- Add NLP-based features from movie overviews
- Deploy model as a web app using Streamlit or Flask

---

## 📬 Contact

For questions or feedback, feel free to reach out.

---
