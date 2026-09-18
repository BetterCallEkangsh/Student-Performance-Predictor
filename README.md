# 🎓 Student Performance Predictor — End-to-End ML Project

An end-to-end machine learning project that predicts a student's **math score** from demographic and academic features. It covers the full lifecycle: EDA, modular training pipeline, model selection with hyperparameter tuning, and a Flask web app for real-time predictions.

---

## 📌 Problem Statement

How do factors like parental education, lunch type, and test preparation affect a student's performance? Given a student's profile and their reading and writing scores, this project predicts the **math score** (a regression task).

## 📊 Dataset

`stud.csv` has 1,000 records with the following columns:

| Column | Type | Description |
|---|---|---|
| `gender` | categorical | Student's gender |
| `race_ethnicity` | categorical | Group A–E |
| `parental_level_of_education` | categorical | Highest education level of parent(s) |
| `lunch` | categorical | Standard or free/reduced |
| `test_preparation_course` | categorical | Completed or none |
| `reading_score` | numerical | Score in reading |
| `writing_score` | numerical | Score in writing |
| `math_score` | numerical | **Target variable** |

## 🧱 Project Structure

```
.
├── app.py                        # Flask application
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/stud.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Reads data, train/test split
│   │   ├── data_transformation.py# Imputation, scaling, one-hot encoding
│   │   └── model_trainer.py      # Trains, tunes and selects best model
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py   # Loads artifacts and serves predictions
│   ├── exception.py              # Custom exception with file/line details
│   ├── logger.py                 # Timestamped file logging
│   └── utils.py                  # save/load objects, model evaluation
├── templates/
│   ├── index.html
│   └── home.html
├── artifacts/                    # Generated: train/test data, model.pkl, preprocessor.pkl
└── logs/                         # Generated: run logs
```

## ⚙️ How It Works

1. **Data Ingestion** — loads `stud.csv`, saves a raw copy, and creates an 80/20 train–test split.
2. **Data Transformation** — builds a `ColumnTransformer`:
   - Numerical (`reading_score`, `writing_score`): median imputation → `StandardScaler`
   - Categorical (5 columns): most-frequent imputation → `OneHotEncoder` → `StandardScaler(with_mean=False)`
3. **Model Training** — trains and tunes seven regressors with `GridSearchCV` (cv=3):
   - Linear Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, CatBoost
4. **Model Selection** — picks the model with the best **R²** on the test set and saves it as `artifacts/model.pkl`. Training fails if no model reaches R² ≥ 0.6.
5. **Prediction Pipeline** — the Flask app collects form inputs, applies the saved preprocessor, and returns the predicted math score.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies (also installs the local package via `-e .`)
pip install -r requirements.txt
```

### Train the model

```bash
python src/components/data_ingestion.py
```

This runs ingestion → transformation → training and writes `model.pkl` and `preprocessor.pkl` to `artifacts/`.

### Run the web app

```bash
python app.py
```

Open `http://127.0.0.1:5000/` and go to `/predictdata` to enter a student's details and get a predicted math score.

## 🛠️ Tech Stack

- **Language:** Python
- **Data & ML:** pandas, NumPy, scikit-learn, XGBoost, CatBoost
- **Visualisation (EDA):** Matplotlib, Seaborn
- **Web:** Flask
- **Tooling:** Git/GitHub, custom logging & exception handling, `setuptools` packaging

## 🗺️ Roadmap

- [x] Project setup, logging, and exception handling
- [x] EDA and model training notebooks
- [x] Data ingestion, transformation, and model trainer components
- [x] Hyperparameter tuning
- [x] Prediction pipeline and Flask app
- [ ] Deployment on AWS Elastic Beanstalk
- [ ] Docker + AWS EC2 / ECR deployment
- [ ] Azure container deployment

## 🙌 Acknowledgements

Built while following Krish Naik's *End-to-End Machine Learning Project* course. Dataset: Students Performance in Exams.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).