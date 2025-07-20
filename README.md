
# 🩺 Diabetes Prediction Using Random Forest

This is a Flask-based web app that predicts whether a patient is diabetic using the Pima Indians Diabetes dataset.

🎥 **Project Demo Video**: [Watch Here](https://drive.google.com/file/d/1dTCiiZXX9ywT-J9-7nUVvBi7DnfSpZGB/view?usp=drive_link)

---

## 💡 Features

- Predicts diabetes based on medical data
- Trained with Random Forest, Logistic Regression & Neural Network
- Uses SMOTE to balance dataset
- Displays prediction result and model accuracy
- Easy-to-use Flask frontend

---

## 🛠 Tech Stack

- **Frontend:** HTML, CSS
- **Backend:** Python, Flask
- **Machine Learning Models:**  
  - Random Forest  
  - Logistic Regression  
  - Neural Network
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `imblearn`

---

## 📁 Folder Structure

```
Diabetes-Prediction-Project/
│
├── predictive_diabetes/            # Main project package
│   ├── dataset/                    # (Optional) Dataset directory
│   └── models/                     # Saved models and results
│       ├── accuracies.pkl
│       ├── diabetes_model.pkl
│       ├── logistic_regression_model.pkl
│       ├── model_accuracy.txt
│       ├── neural_network_model.pkl
│       ├── random_forest_model.pkl
│       └── scaler.pkl
│
├── static/                         # CSS styling
│   ├── style.css
│   └── styles.css
│
├── templates/                      # HTML templates
│   ├── index.html
│   ├── prediction_form.html
│   └── result.html
│
├── app.py                          # Flask application entry point
└── requirements.txt                # Python dependencies
```

---

## 🚀 Run Locally

Follow these steps to run the project on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/Divya22022004/Diabetes-Prediction-Project.git
cd Diabetes-Prediction-Project
```

### 2. Create & Activate Virtual Environment

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Now open your browser and visit:  
👉 [http://localhost:5000](http://localhost:5000)
