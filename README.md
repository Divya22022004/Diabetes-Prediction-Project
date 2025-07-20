# 🩺 Diabetes Prediction Using Random Forest

A **Flask-based web application** that predicts whether a person is diabetic using the **Pima Indians Diabetes Dataset**. The application leverages multiple machine learning models and provides accurate predictions with visual results.

---

## 🎥 Demo

▶️ **[Watch Demo Video](https://drive.google.com/file/d/1dTCiiZXX9ywT-J9-7nUVvBi7DnfSpZGB/view?usp=drive_link)**

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

<details>
<summary>Click to expand</summary>

```plaintext
PREDICTIVE_DIABETES/
│
├── predictive_diabetes/            # Main project package
│   ├── dataset/                    # Dataset files (optional)
│   └── models/                     # Trained model files
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
├── app.py                          # Main Flask application
└── requirements.txt                # Python dependencies
